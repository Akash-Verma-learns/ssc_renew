"""
Routes
------
All FastAPI endpoints. Wired into api.py via app.include_router(router).

Feedback + Learning additions:
  POST /rfps/{id}/clauses/{type}/feedback   — submit/update feedback (any user)
  GET  /rfps/{id}/clauses/{type}/feedback   — view feedback for one clause
  GET  /rfps/{id}/feedback                  — summary across all 10 clauses
  DEL  /rfps/{id}/clauses/{type}/feedback   — retract own feedback
  GET  /feedback/insights                   — aggregated learning signal (admin)
  GET  /feedback/log                        — full audit log (admin)
  POST /feedback/synthesise                 — trigger LLM rule synthesis (admin)
  GET  /feedback/rules                      — list all learned rules (admin)
  PATCH /feedback/rules/{id}               — activate/deactivate/edit rule (admin)
  GET  /feedback/examples                   — list few-shot examples (admin)
  PATCH /feedback/examples/{id}            — adjust usefulness score (admin)
"""

import uuid
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import (
    get_db, User, RFP, ClauseResult, Comment,
    ClauseFeedback, LearningExample, LearnedRule,
    TQEvaluation, TQScoreItem,
)
from auth import (
    verify_password, create_token, hash_password,
    get_current_user, require_admin,
    require_tq_access, require_tq_or_admin,
)

router = APIRouter()

UPLOAD_DIR = Path("./uploads")
OUTPUT_DIR = Path("./outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

_OFFERING_SOLUTIONS_PATH = Path(__file__).parent / "offering_solutions.json"
try:
    with open(_OFFERING_SOLUTIONS_PATH, "r", encoding="utf-8") as f:
        OFFERING_SOLUTIONS: dict = json.load(f)
except FileNotFoundError:
    OFFERING_SOLUTIONS = {}

VALID_CLAUSE_TYPES = {
    "liability", "insurance", "scope", "payment",
    "deliverables", "personnel", "ld", "penalties",
    "termination", "eligibility",
}
VALID_AGREEMENTS  = {"agree", "too_high", "too_low", "incorrect"}
VALID_RISK_LEVELS = {"HIGH", "MEDIUM", "LOW", "ACCEPTABLE", "NEEDS_REVIEW"}


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic schemas
# ══════════════════════════════════════════════════════════════════════════════

class LoginRequest(BaseModel):
    email: str
    password: str

class CommentRequest(BaseModel):
    clause_type: str
    comment_text: str

class CreateUserRequest(BaseModel):
    name: str
    email: str
    password: str
    role: str = "reviewer"

class FeedbackRequest(BaseModel):
    """
    agreement:
      agree      — system assessment is correct for this offering/solution
      too_high   — system over-stated the risk for this type of work
      too_low    — system under-stated the risk for this type of work
      incorrect  — reasoning is wrong (explain in feedback_comment)

    suggested_risk_level (optional):
      HIGH | MEDIUM | LOW | ACCEPTABLE | NEEDS_REVIEW

    feedback_comment (optional):
      Free text. If provided on a non-agree submission, a LearningExample
      is automatically created and injected into future LLM prompts for
      matching offering/solution/clause combinations.
    """
    agreement: str
    suggested_risk_level: Optional[str] = None
    feedback_comment: Optional[str] = None

class SynthesiseRequest(BaseModel):
    clause_type: str
    offering: str
    solution: str
    force: bool = False   # bypass minimum-feedback guard (for testing/demos)

class RuleUpdateRequest(BaseModel):
    is_active: Optional[bool] = None
    rule_text: Optional[str]  = None   # admin manual override of synthesised text

class ExampleUpdateRequest(BaseModel):
    is_active:        Optional[bool] = None
    usefulness_score: Optional[int]  = None   # 1 = normal, 2 = high, 3 = essential


# ══════════════════════════════════════════════════════════════════════════════
# Serialisers
# ══════════════════════════════════════════════════════════════════════════════
def _parse_fin_methodology(ev) -> dict | None:
        try:
            raw = ev.financial_methodology_json
            if raw:
                return json.loads(raw)
        except Exception:
            pass
        return None

def user_to_dict(u: User) -> dict:
    return {
        "id": u.id, "name": u.name, "email": u.email, "role": u.role,
        "created_at": u.created_at.isoformat() if u.created_at else None,
    }

def clause_to_dict(c: ClauseResult) -> dict:
    return {
        "clause_text":             c.clause_text,
        "clause_reference":        c.clause_reference,
        "page_no":                 c.page_no,
        "risk_level":              c.risk_level,           # raw system output
        "risk_description":        c.risk_description,
        "auto_remark":             c.auto_remark,
        "needs_exception_approval": c.needs_exception,
        "needs_eqcr":              c.needs_eqcr,
        "deviation_suggested":     c.deviation_suggested,
        # Feedback-engine adjustments
        "adjusted_risk_level":     c.adjusted_risk_level,
        "effective_risk_level":    c.adjusted_risk_level or c.risk_level,  # use this in UI
        "adjustment_reason":       c.adjustment_reason,
        "adjustment_confidence":   float(c.adjustment_confidence) if c.adjustment_confidence else None,
        "feedback_count":          c.feedback_count or 0,
        # Learning
        "learned_rule_applied":    c.learned_rule_applied or False,
    }

def feedback_to_dict(fb: ClauseFeedback) -> dict:
    return {
        "id": fb.id, "rfp_id": fb.rfp_id, "clause_type": fb.clause_type,
        "user_id": fb.user_id,
        "user_name": fb.user.name if fb.user else "Unknown",
        "user_role": fb.user.role if fb.user else "",
        "agreement": fb.agreement,
        "suggested_risk_level": fb.suggested_risk_level,
        "feedback_comment": fb.feedback_comment,
        "system_risk_level": fb.system_risk_level,
        "offering": fb.offering, "solution": fb.solution, "bu": fb.bu,
        "created_at": fb.created_at.isoformat() if fb.created_at else None,
    }

def rule_to_dict(r: LearnedRule) -> dict:
    try:    threshold_notes = json.loads(r.threshold_notes_json or "{}")
    except: threshold_notes = {}
    return {
        "id": r.id, "clause_type": r.clause_type,
        "offering": r.offering, "solution": r.solution,
        "rule_text": r.rule_text, "threshold_notes": threshold_notes,
        "key_differences": r.key_differences, "confidence": r.confidence,
        "feedback_count_at_gen": r.feedback_count_at_gen,
        "is_active": r.is_active,
        "generated_at": r.generated_at.isoformat() if r.generated_at else None,
    }

def example_to_dict(e: LearningExample) -> dict:
    return {
        "id": e.id, "feedback_id": e.feedback_id,
        "clause_type": e.clause_type, "offering": e.offering,
        "solution": e.solution, "bu": e.bu,
        "clause_snippet": e.clause_snippet,
        "system_risk_level": e.system_risk_level,
        "correct_risk_level": e.correct_risk_level,
        "reviewer_reason": e.reviewer_reason,
        "usefulness_score": e.usefulness_score, "is_active": e.is_active,
        "created_at": e.created_at.isoformat() if e.created_at else None,
    }

def _parse_offering_solutions(offering_str: str, solutions_str: str):
    def _p(raw):
        if not raw: return []
        raw = str(raw).strip()
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if x and str(x).strip()]
            except Exception: pass
        return [raw]
    return _p(offering_str), _p(solutions_str)

def rfp_to_dict(rfp: RFP, include_clauses=False, viewer_role="reviewer") -> dict:
    offerings, solutions = _parse_offering_solutions(rfp.offering or "", rfp.solutions or "")
    d = {
        "id": rfp.id,
        "opportunity_name": rfp.opportunity_name or "",
        "client_name":      rfp.client_name or "",
        "bu":               rfp.bu or "",
        "bu_code":          rfp.classification or "",
        "state":            rfp.state or "",
        "country":          rfp.country or "",
        "offerings":        offerings,
        "solutions":        solutions,
        "offering":         offerings[0] if offerings else (rfp.offering or ""),
        "solution":         solutions[0] if solutions else (rfp.solutions or ""),
        "offering_raw":     rfp.offering or "",
        "solutions_raw":    rfp.solutions or "",
        "file_name":        rfp.file_name or "",
        "job_id":           rfp.job_id or "",
        "status":           rfp.status or "queued",
        "progress":         rfp.progress or 0,
        "current_step":     rfp.current_step or "",
        "error_message":    rfp.error_message,
        "created_at":       rfp.created_at.isoformat() if rfp.created_at else None,
    }
    d["uploaded_by_name"] = (rfp.uploaded_by_user.name if rfp.uploaded_by_user else "") \
        if viewer_role == "admin" else None
    if include_clauses:
        d["clauses"] = {c.clause_type: clause_to_dict(c) for c in rfp.clause_results}
    return d


# ══════════════════════════════════════════════════════════════════════════════
# Background pipeline task — learning-aware
# ══════════════════════════════════════════════════════════════════════════════

"""
run_pipeline_task — replacement for the function in routes.py
-------------------------------------------------------------
Paste this function over the existing run_pipeline_task definition in routes.py.
Everything else in routes.py stays the same.

KEY CHANGES vs the original
----------------------------
1.  extract_all_clauses no longer receives `db`.
    The extractor now opens its own fresh session per clause (see core/extractor.py).
    Passing `db` was the root cause of the SSL-drop cascade.

2.  get_learned_rule and get_adjustment each use a fresh short-lived session
    opened and closed inside a try/finally. If any one query fails, only that
    query's session is affected; the main session is untouched.

3.  The main session (`db`) is ONLY used for:
      - reading the RFP row
      - updating rfp.status / rfp.progress
      - writing ClauseResult rows
      - committing at the very end
    It is NEVER held open during any Ollama call.

4.  Every db.commit() that writes RFP state is wrapped in its own try/except
    so a transient DB blip cannot prevent clause results from being saved.

5.  The final failure handler re-opens a fresh session if the main one is dead,
    so the "failed" status is always written even after a catastrophic error.
"""

from pathlib import Path

from core.document_parser import parse_document_simple as parse_document
def run_pipeline_task(rfp_id: int, file_path: str, job_id: str):
    from database import SessionLocal, RFP, ClauseResult, LearnedRule

    # ── helpers ────────────────────────────────────────────────────────────────

    def _fresh_db():
        """Return a brand-new session.  Caller must close() in a finally block."""
        return SessionLocal()

    def _safe_commit(db, label=""):
        try:
            db.commit()
        except Exception as e:
            print(f"[Pipeline] commit failed ({label}): {e}")
            try:
                db.rollback()
            except Exception:
                pass

    # ── open main session ──────────────────────────────────────────────────────
    db = _fresh_db()

    try:
        rfp = db.query(RFP).filter(RFP.id == rfp_id).first()
        if not rfp:
            print(f"[Pipeline] RFP {rfp_id} not found — aborting.")
            return

        # ── status helper (uses main session) ─────────────────────────────────
        def update(status, progress, step):
            try:
                rfp.status       = status
                rfp.progress     = progress
                rfp.current_step = step
                db.commit()
            except Exception as e:
                print(f"[Pipeline] status update failed: {e}")
                try:
                    db.rollback()
                except Exception:
                    pass

        update("processing", 10, "Parsing document")

        from routes import _parse_offering_solutions   # local import — avoids circular
        offerings, solutions = _parse_offering_solutions(rfp.offering or "", rfp.solutions or "")
        primary_offering = offerings[0] if offerings else ""
        primary_solution = solutions[0] if solutions else ""

        from core.parser       import parse_document
        from core.vector_store import ingest_chunks
        from core.extractor    import extract_all_clauses
        from rules.risk_engine import evaluate_clause, RiskResult

        doc_name = Path(file_path).name
        chunks   = parse_document(file_path)
        print(f"[Pipeline] {len(chunks)} chunks extracted")

        update("processing", 35, "Ingesting into vector store")
        ingest_chunks(chunks, doc_id=doc_name)

        # ── Metadata extraction ────────────────────────────────────────────────
        # Uses its own requests call (no DB held open).
        update("processing", 45, "Extracting document metadata")
        try:
            from core.metadata_extractor import extract_metadata
            meta     = extract_metadata(doc_name)
            opp_name = meta.get("opportunity_name")
            cli_name = meta.get("client_name")

            if opp_name and len(opp_name.strip()) > 5:
                rfp.opportunity_name = opp_name.strip()
                print(f"[Pipeline] opportunity_name set: {rfp.opportunity_name!r}")
            else:
                print(f"[Pipeline] opportunity_name not extracted (got: {opp_name!r})")

            if cli_name and len(cli_name.strip()) > 2:
                rfp.client_name = cli_name.strip()
                print(f"[Pipeline] client_name set: {rfp.client_name!r}")
            else:
                print(f"[Pipeline] client_name not extracted (got: {cli_name!r})")

            _safe_commit(db, "metadata")
        except Exception as meta_exc:
            print(f"[Pipeline] Metadata extraction skipped: {meta_exc}")
            try:
                db.rollback()
            except Exception:
                pass

        # ── Clause extraction ──────────────────────────────────────────────────
        # IMPORTANT: do NOT pass `db` here.
        # extract_all_clauses v2 opens its own fresh session per clause lookup,
        # so no DB connection is ever held open during an Ollama call.
        update("processing", 50, "Extracting clauses (with learning context)")

        extraction_results = {}
        try:
            extraction_results = extract_all_clauses(
                doc_name=doc_name,
                model="llama3.2",
                offering=primary_offering,
                solution=primary_solution,
                # db intentionally omitted — extractor manages its own sessions
            )
        except Exception as ext_exc:
            print(f"[Pipeline] Clause extraction error: {ext_exc}")
            # Continue — we'll save whatever partial results we have

        # ── Learned-rule lookup ────────────────────────────────────────────────
        # Fresh session per clause type — closed immediately after the query.
        update("processing", 70, "Evaluating risk + checking learned rules")

        CLAUSE_ORDER = [
            "liability", "insurance", "scope", "payment", "deliverables",
            "personnel", "ld", "penalties", "termination", "eligibility",
        ]

        from rules.learning_store import get_learned_rule
        learned_rule_ids: dict = {}
        for ct in CLAUSE_ORDER:
            _db2 = _fresh_db()
            try:
                rule_text = get_learned_rule(ct, primary_offering, primary_solution, _db2)
                if rule_text:
                    rule_row = _db2.query(LearnedRule).filter(
                        LearnedRule.clause_type == ct,
                        LearnedRule.is_active   == True,
                    ).first()
                    if rule_row:
                        learned_rule_ids[ct] = rule_row.id
                        print(f"[Pipeline] Learned rule active: {ct} / {primary_offering}")
            except Exception as lr_err:
                print(f"[Pipeline] Learned rule lookup failed ({ct}): {lr_err}")
            finally:
                try:
                    _db2.close()
                except Exception:
                    pass

        # ── Risk evaluation + feedback adjustment ──────────────────────────────
        update("processing", 82, "Saving results")

        for ct in CLAUSE_ORDER:
            ext  = extraction_results.get(ct, {})
            exd  = ext.get("extracted", {})

            try:
                risk = evaluate_clause(ct, exd)
            except Exception as re_err:
                risk = RiskResult(
                    clause_name=ct,
                    risk_level="NEEDS_REVIEW",
                    risk_description=f"Evaluation failed: {re_err}",
                    auto_remark="",
                )

            # Feedback adjustment — fresh session, closed immediately
            adj_level = adj_reason = adj_conf = None
            adj_count = 0
            _db3 = _fresh_db()
            try:
                from rules.feedback_engine import get_adjustment
                adj = get_adjustment(ct, primary_offering, primary_solution, risk.risk_level, _db3)
                adj_level  = adj["adjusted_risk_level"] if adj["applied"] else None
                adj_reason = adj["reason"]              if adj["applied"] else None
                adj_conf   = str(adj["confidence"])     if adj["applied"] else None
                adj_count  = adj["feedback_count"]
            except Exception as adj_err:
                print(f"[Pipeline] Adjustment lookup failed ({ct}): {adj_err}")
            finally:
                try:
                    _db3.close()
                except Exception:
                    pass

            cr = ClauseResult(
                rfp_id=rfp_id,
                clause_type=ct,
                clause_text=exd.get("clause_text") or exd.get("summary", ""),
                clause_reference=exd.get("clause_reference", ""),
                page_no=str(exd.get("page_no", "") or ""),
                risk_level=risk.risk_level,
                risk_description=risk.risk_description,
                auto_remark=exd.get("auto_remark", ""),
                needs_exception=bool(exd.get("needs_exception_approval", False)),
                needs_eqcr=bool(exd.get("needs_eqcr", False)),
                deviation_suggested=exd.get("deviation_suggested", "") or "",
                adjusted_risk_level=adj_level,
                adjustment_reason=adj_reason,
                adjustment_confidence=adj_conf,
                feedback_count=adj_count,
                learned_rule_applied=ct in learned_rule_ids,
                learned_rule_id=learned_rule_ids.get(ct),
            )
            db.add(cr)

        # ── DOCX output ────────────────────────────────────────────────────────
        from pathlib import Path as _Path
        OUTPUT_DIR = _Path("./outputs")
        OUTPUT_DIR.mkdir(exist_ok=True)

        try:
            from output.writer import build_table_rows, fill_ssc1_table
            table_rows = build_table_rows({
                ct: {
                    "extracted": extraction_results.get(ct, {}).get("extracted", {}),
                    "risk": None,
                }
                for ct in CLAUSE_ORDER
            })
            fill_ssc1_table(
                table_rows,
                "document_for_format.docx",
                str(OUTPUT_DIR / f"{job_id}_ssc1.docx"),
                rfp_name=rfp.opportunity_name or job_id,
            )
        except Exception as docx_err:
            print(f"[Pipeline] DOCX output skipped: {docx_err}")

        # ── Final commit ───────────────────────────────────────────────────────
        rfp.status       = "completed"
        rfp.progress     = 100
        rfp.current_step = "Done"
        db.commit()
        print(f"[Pipeline] RFP {rfp_id} completed.")

    except Exception as e:
        print(f"[Pipeline] RFP {rfp_id} FAILED: {e}")
        # Try to mark failed on the main session; if that's dead, open a fresh one.
        for _session in [db, _fresh_db()]:
            try:
                try:
                    _session.rollback()
                except Exception:
                    pass
                _session.query(RFP).filter(RFP.id == rfp_id).update({
                    "status":        "failed",
                    "error_message": str(e),
                    "current_step":  "Failed",
                })
                _session.commit()
                break   # succeeded — stop trying
            except Exception as db_err:
                print(f"[Pipeline] Could not write failed status: {db_err}")
            finally:
                if _session is not db:   # don't double-close the main session
                    try:
                        _session.close()
                    except Exception:
                        pass
    finally:
        try:
            db.close()
        except Exception:
            pass

# ══════════════════════════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/auth/login")
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Invalid email or password")
    return {
        "access_token": create_token(user.id),
        "token_type": "bearer",
        "user": user_to_dict(user),
    }

@router.get("/auth/me")
def me(current_user: User = Depends(get_current_user)):
    return user_to_dict(current_user)

@router.get("/offering-solutions")
def get_offering_solutions(current_user: User = Depends(get_current_user)):
    return OFFERING_SOLUTIONS


# ══════════════════════════════════════════════════════════════════════════════
# RFP ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/rfps")
def list_rfps(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rfps = db.query(RFP).order_by(RFP.created_at.desc()).all()
    return [rfp_to_dict(r, viewer_role=current_user.role) for r in rfps]


@router.post("/rfps/upload")
async def upload_rfp(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    form = await request.form()

    # Find the file field regardless of what name Lovable used
    file_obj = form.get("file") or form.get("rfp_file") or form.get("document")
    if file_obj is None:
        for k in form.keys():
            v = form[k]
            if hasattr(v, "filename") and v.filename:
                file_obj = v
                break
    if file_obj is None:
        raise HTTPException(400, "No file found in upload.")

    ext = Path(file_obj.filename).suffix.lower()
    if ext not in (".pdf", ".docx", ".doc"):
        raise HTTPException(400, f"Only PDF and DOCX supported.")

    form_lower = {
        k.lower().replace(" ", "_").replace("-", "_"): str(form.get(k)).strip()
        for k in form.keys()
        if not hasattr(form.get(k), "filename")
    }

    def _f(*names, default=""):
        for name in names:
            v = form.get(name)
            if v is not None and not hasattr(v, "filename") and str(v).strip():
                return str(v).strip()
            norm = name.lower().replace(" ", "_").replace("-", "_")
            if norm in form_lower and form_lower[norm]:
                return form_lower[norm]
        return default

    def _pa(raw):
        if not raw: return []
        raw = str(raw).strip()
        if raw.startswith("["):
            try:
                return [str(x).strip() for x in json.loads(raw) if x and str(x).strip()]
            except Exception: pass
        return [raw] if raw else []

    opportunity_name = _f("opportunity_name", "opportunityName", "Opportunity Name", "title")
    client_name = _f("client_name", "clientName", "Client Name", "client")
    bu      = _f("bu", "name_of_bu", "nameOfBu", "business_unit", "businessUnit")
    state   = _f("state", "State", "location")
    country = _f("country", "Country", "nation")
    bu_code = _f("bu_code", "buCode", "classification", "Classification") or "TRF"

    raw_off = _f("offerings_json", "offeringsJson", "offerings", "Offerings")
    raw_sol = _f("solutions_json", "solutionsJson", "solutions", "Solutions", "solution")
    resolved_off = _pa(raw_off)
    resolved_sol = _pa(raw_sol)
    for i in range(1, 6):
        o = _f(f"offering_{i}", f"offering{i}")
        s = _f(f"solution_{i}", f"solution{i}")
        if o and o not in resolved_off: resolved_off.append(o)
        if s and s not in resolved_sol:  resolved_sol.append(s)
    resolved_off = [x for x in resolved_off if x][:5]
    resolved_sol = [x for x in resolved_sol  if x][:5]

    job_id    = str(uuid.uuid4())[:8]
    save_path = UPLOAD_DIR / f"{job_id}{ext}"
    with open(save_path, "wb") as fp:
        fp.write(await file_obj.read())

    rfp = RFP(
        opportunity_name=opportunity_name or "",
        client_name=client_name, bu=bu, classification=bu_code,
        state=state, country=country,
        offering=json.dumps(resolved_off, ensure_ascii=False),
        solutions=json.dumps(resolved_sol, ensure_ascii=False),
        file_name=file_obj.filename, job_id=job_id,
        status="queued", progress=0, uploaded_by=current_user.id,
    )
    db.add(rfp)
    db.commit()
    db.refresh(rfp)

    background_tasks.add_task(run_pipeline_task, rfp.id, str(save_path), job_id)
    return {"job_id": job_id, "rfp_id": rfp.id, "status": "queued"}


@router.get("/rfps/{rfp_id}/status")
def get_status(
    rfp_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rfp = db.query(RFP).filter(RFP.id == rfp_id).first()
    if not rfp: raise HTTPException(404, "RFP not found")
    return {
        "rfp_id": rfp.id, "status": rfp.status,
        "progress": rfp.progress, "current_step": rfp.current_step,
        "error_message": rfp.error_message,
    }


@router.get("/rfps/{rfp_id}")
def get_rfp(
    rfp_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rfp = db.query(RFP).filter(RFP.id == rfp_id).first()
    if not rfp: raise HTTPException(404, "RFP not found")
    return rfp_to_dict(rfp, include_clauses=True, viewer_role=current_user.role)


@router.post("/rfps/{rfp_id}/complete")
def mark_complete(
    rfp_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    rfp = db.query(RFP).filter(RFP.id == rfp_id).first()
    if not rfp: raise HTTPException(404, "RFP not found")
    rfp.status = "completed"
    db.commit()
    return {"status": "completed"}


@router.patch("/rfps/{rfp_id}")
async def update_rfp(
    rfp_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    rfp = db.query(RFP).filter(RFP.id == rfp_id).first()
    if not rfp: raise HTTPException(404, "RFP not found")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Request body must be valid JSON")

    def _pa(raw):
        if not raw: return []
        if isinstance(raw, list): return [str(x).strip() for x in raw if x]
        raw = str(raw).strip()
        if raw.startswith("["):
            try: return [str(x).strip() for x in json.loads(raw) if x]
            except Exception: pass
        return [raw] if raw else []

    for field in ["opportunity_name", "client_name", "bu", "state", "country"]:
        if field in body:
            setattr(rfp, field, str(body[field]).strip())
    if "bu_code" in body or "classification" in body:
        rfp.classification = str(body.get("bu_code") or body.get("classification", "")).strip()
    new_off = _pa(body.get("offerings_json") or body.get("offerings") or body.get("offering"))
    new_sol = _pa(body.get("solutions_json") or body.get("solutions") or body.get("solution"))
    if new_off: rfp.offering  = json.dumps(new_off[:5], ensure_ascii=False)
    if new_sol: rfp.solutions = json.dumps(new_sol[:5], ensure_ascii=False)

    db.commit()
    db.refresh(rfp)
    return rfp_to_dict(rfp, viewer_role=current_user.role)


@router.get("/rfps/{rfp_id}/download")
def download_rfp(
    rfp_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rfp = db.query(RFP).filter(RFP.id == rfp_id).first()
    if not rfp: raise HTTPException(404, "RFP not found")
    out = OUTPUT_DIR / f"{rfp.job_id}_ssc1.docx"
    if not out.exists(): raise HTTPException(404, "Output not ready yet.")
    return FileResponse(
        str(out),
        filename=f"SSC1_Review_{rfp.opportunity_name[:30]}.docx",
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


# ══════════════════════════════════════════════════════════════════════════════
# COMMENTS
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/rfps/{rfp_id}/comments")
def get_comments(
    rfp_id: int,
    clause: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    q = db.query(Comment).filter(Comment.rfp_id == rfp_id)
    if clause:
        q = q.filter(Comment.clause_type == clause)
    return [
        {
            "id": c.id, "clause_type": c.clause_type, "user_id": c.user_id,
            "user_name": c.user.name if c.user else "Unknown",
            "user_role": c.user.role if c.user else "",
            "comment_text": c.comment_text,
            "created_at": c.created_at.isoformat() if c.created_at else None,
        }
        for c in q.order_by(Comment.created_at.asc()).all()
    ]


@router.post("/rfps/{rfp_id}/comments")
def post_comment(
    rfp_id: int,
    body: CommentRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not db.query(RFP).filter(RFP.id == rfp_id).first():
        raise HTTPException(404, "RFP not found")
    c = Comment(
        rfp_id=rfp_id, clause_type=body.clause_type,
        user_id=current_user.id, comment_text=body.comment_text,
    )
    db.add(c)
    db.commit()
    db.refresh(c)
    return {
        "id": c.id, "clause_type": c.clause_type, "user_id": c.user_id,
        "user_name": current_user.name, "user_role": current_user.role,
        "comment_text": c.comment_text, "created_at": c.created_at.isoformat(),
    }


@router.delete("/rfps/{rfp_id}/comments/{comment_id}")
def delete_comment(
    rfp_id: int, comment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    c = db.query(Comment).filter(
        Comment.id == comment_id, Comment.rfp_id == rfp_id,
    ).first()
    if not c: raise HTTPException(404, "Comment not found")
    if current_user.role != "admin" and c.user_id != current_user.id:
        raise HTTPException(403, "Not allowed")
    db.delete(c)
    db.commit()
    return {"deleted": True}


# ══════════════════════════════════════════════════════════════════════════════
# FEEDBACK — submit, view, retract
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/rfps/{rfp_id}/clauses/{clause_type}/feedback")
def submit_feedback(
    rfp_id: int,
    clause_type: str,
    body: FeedbackRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Submit or update feedback for one clause. One record per user per clause
    per RFP (upsert — re-submitting overwrites the previous response).

    If feedback_comment is provided on a non-agree submission, a
    LearningExample is created automatically and will be injected into
    future LLM prompts for matching offering/solution/clause combinations.
    """
    if clause_type not in VALID_CLAUSE_TYPES:
        raise HTTPException(400, "Unknown clause type.")
    if body.agreement not in VALID_AGREEMENTS:
        raise HTTPException(400, f"Invalid agreement value: {body.agreement}")
    if body.suggested_risk_level and body.suggested_risk_level not in VALID_RISK_LEVELS:
        raise HTTPException(400, f"Invalid risk level: {body.suggested_risk_level}")

    rfp = db.query(RFP).filter(RFP.id == rfp_id).first()
    if not rfp: raise HTTPException(404, "RFP not found")

    cr = db.query(ClauseResult).filter(
        ClauseResult.rfp_id == rfp_id,
        ClauseResult.clause_type == clause_type,
    ).first()
    system_risk = cr.risk_level if cr else "UNKNOWN"

    offerings, solutions = _parse_offering_solutions(rfp.offering or "", rfp.solutions or "")
    off = offerings[0] if offerings else ""
    sol = solutions[0] if solutions else ""

    # Upsert — one feedback per user per clause per RFP
    existing = db.query(ClauseFeedback).filter(
        ClauseFeedback.rfp_id == rfp_id,
        ClauseFeedback.clause_type == clause_type,
        ClauseFeedback.user_id == current_user.id,
    ).first()

    if existing:
        existing.agreement            = body.agreement
        existing.suggested_risk_level = body.suggested_risk_level
        existing.feedback_comment     = body.feedback_comment
        existing.system_risk_level    = system_risk
        existing.offering             = off
        existing.solution             = sol
        existing.bu                   = rfp.bu or ""
        existing.created_at           = datetime.utcnow()
        db.commit()
        db.refresh(existing)
        fb = existing
        action = "updated"
    else:
        fb = ClauseFeedback(
            rfp_id=rfp_id,
            clause_result_id=cr.id if cr else None,
            clause_type=clause_type,
            user_id=current_user.id,
            offering=off, solution=sol, bu=rfp.bu or "",
            agreement=body.agreement,
            suggested_risk_level=body.suggested_risk_level,
            feedback_comment=body.feedback_comment,
            system_risk_level=system_risk,
        )
        db.add(fb)
        db.commit()
        db.refresh(fb)
        action = "created"

    # Auto-create LearningExample from high-quality feedback (comment + disagreement)
    if body.feedback_comment and body.agreement != "agree":
        try:
            from rules.learning_store import create_learning_example
            snippet = cr.clause_text[:300] if cr and cr.clause_text else ""
            if create_learning_example(fb.id, db, clause_snippet=snippet):
                print(f"[Feedback] Learning example created: {clause_type} / {off}")
        except Exception as le:
            print(f"[Feedback] Learning example skipped: {le}")


    # ── Live risk-level update ─────────────────────────────────────────────
    # Recompute adjusted_risk_level for all matching RFPs so the review page
    # reflects feedback immediately without needing a re-upload.
    try:
        from rules.feedback_engine import get_adjustment as _gadj
        _norm_u = lambda t: (t or "").strip().upper()
        _matched_rfps = db.query(RFP).filter(RFP.status == "completed").all()
        _updated = 0
        for _r in _matched_rfps:
            _ro, _rs = _parse_offering_solutions(_r.offering or "", _r.solutions or "")
            _r_off = _ro[0] if _ro else ""
            _r_sol = _rs[0] if _rs else ""
            _no = _norm_u(_r_off); _to = _norm_u(off)
            if not (_to in _no or _no in _to or not _to):
                continue
            _cr2 = db.query(ClauseResult).filter(
                ClauseResult.rfp_id == _r.id,
                ClauseResult.clause_type == clause_type,
            ).first()
            if not _cr2:
                continue
            _adj2 = _gadj(clause_type, _r_off, _r_sol, _cr2.risk_level, db)
            _cr2.adjusted_risk_level   = _adj2["adjusted_risk_level"] if _adj2["applied"] else None
            _cr2.adjustment_reason     = _adj2["reason"]              if _adj2["applied"] else None
            _cr2.adjustment_confidence = str(_adj2["confidence"])     if _adj2["applied"] else None
            _cr2.feedback_count        = _adj2["feedback_count"]
            _updated += 1
        if _updated:
            db.commit()
            print(f"[Feedback] Live risk update: {clause_type} → {_updated} clause result(s) updated")
    except Exception as _ue:
        print(f"[Feedback] Live risk update skipped: {_ue}")
    # ── end live risk-level update ─────────────────────────────────────────

    print(f"[Feedback] {action.upper()} — {current_user.email!r} "
          f"rfp={rfp_id} {clause_type!r} {body.agreement!r} "
          f"offering={off!r} solution={sol!r}")

    return {**feedback_to_dict(fb), "action": action}


@router.get("/rfps/{rfp_id}/clauses/{clause_type}/feedback")
def get_clause_feedback(
    rfp_id: int,
    clause_type: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Admins see all feedback for the clause.
    Reviewers see only their own.
    Both see the aggregate breakdown.
    """
    if clause_type not in VALID_CLAUSE_TYPES:
        raise HTTPException(400, "Unknown clause type.")

    q = db.query(ClauseFeedback).filter(
        ClauseFeedback.rfp_id == rfp_id,
        ClauseFeedback.clause_type == clause_type,
    )
    if current_user.role != "admin":
        q = q.filter(ClauseFeedback.user_id == current_user.id)
    feedbacks = q.order_by(ClauseFeedback.created_at.asc()).all()

    all_fb = db.query(ClauseFeedback).filter(
        ClauseFeedback.rfp_id == rfp_id,
        ClauseFeedback.clause_type == clause_type,
    ).all()
    from collections import Counter
    breakdown = dict(Counter(f.agreement for f in all_fb))

    return {
        "clause_type": clause_type, "rfp_id": rfp_id,
        "total_feedback": len(all_fb),
        "breakdown": breakdown,
        "my_feedback": next(
            (feedback_to_dict(f) for f in feedbacks if f.user_id == current_user.id), None
        ),
        "all_feedback": [feedback_to_dict(f) for f in feedbacks],
    }


@router.get("/rfps/{rfp_id}/feedback")
def get_rfp_feedback_summary(
    rfp_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Summary across all 10 clauses — useful for a review completion tracker."""
    if not db.query(RFP).filter(RFP.id == rfp_id).first():
        raise HTTPException(404, "RFP not found")

    all_fb = db.query(ClauseFeedback).filter(ClauseFeedback.rfp_id == rfp_id).all()
    from collections import defaultdict, Counter
    by_clause = defaultdict(list)
    for fb in all_fb:
        by_clause[fb.clause_type].append(fb)

    summary = {}
    for ct, fbs in by_clause.items():
        my = next((f for f in fbs if f.user_id == current_user.id), None)
        summary[ct] = {
            "total": len(fbs),
            "breakdown": dict(Counter(f.agreement for f in fbs)),
            "my_agreement": my.agreement if my else None,
            "my_suggested_level": my.suggested_risk_level if my else None,
        }

    my_done = {fb.clause_type for fb in all_fb if fb.user_id == current_user.id}
    return {
        "rfp_id": rfp_id,
        "total_clauses": len(VALID_CLAUSE_TYPES),
        "my_feedback_count": len(my_done),
        "my_clauses_done": sorted(my_done),
        "clauses_pending": sorted(VALID_CLAUSE_TYPES - my_done),
        "by_clause": summary,
    }


@router.delete("/rfps/{rfp_id}/clauses/{clause_type}/feedback")
def delete_my_feedback(
    rfp_id: int,
    clause_type: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    fb = db.query(ClauseFeedback).filter(
        ClauseFeedback.rfp_id == rfp_id,
        ClauseFeedback.clause_type == clause_type,
        ClauseFeedback.user_id == current_user.id,
    ).first()
    if not fb: raise HTTPException(404, "No feedback found to retract")
    db.delete(fb)
    db.commit()
    return {"deleted": True}


# ══════════════════════════════════════════════════════════════════════════════
# FEEDBACK INSIGHTS + AUDIT LOG (admin only)
# ══════════════════════════════════════════════════════════════════════════════


@router.get("/feedback/summary")
def feedback_summary(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """
    Returns the four headline stats shown in the Insights panel.
    """
    from rules.feedback_engine import get_feedback_insights
    total = db.query(ClauseFeedback).count()
    insights = get_feedback_insights(db)
    strong = sum(1 for i in insights if i["has_strong_signal"])
    rules_count    = db.query(LearnedRule).filter(LearnedRule.is_active == True).count()
    examples_count = db.query(LearningExample).filter(LearningExample.is_active == True).count()
    return {
        "total_feedback":  total,
        "strong_signal":   strong,
        "active_rules":    rules_count,
        "active_examples": examples_count,
    }


@router.get("/feedback/insights")
def feedback_insights(
    clause_type: Optional[str] = Query(None),
    offering: Optional[str]    = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """
    Aggregated view of where feedback consensus exists.
    has_strong_signal=True means the feedback engine WILL apply an adjustment
    to the next RFP analysed for that offering/solution/clause combination.
    """
    from rules.feedback_engine import get_feedback_insights, CONSENSUS_THRESHOLD, MIN_FEEDBACK
    insights = get_feedback_insights(db)
    if clause_type:
        insights = [i for i in insights if i["clause_type"] == clause_type]
    if offering:
        insights = [i for i in insights if offering.upper() in i["offering"]]
    return {
        "settings": {
            "min_feedback_required": MIN_FEEDBACK,
            "consensus_threshold_pct": int(CONSENSUS_THRESHOLD * 100),
        },
        "total_groups": len(insights),
        "insights": insights,
    }


@router.get("/feedback/log")
def feedback_log(
    clause_type: Optional[str] = Query(None),
    offering: Optional[str]    = Query(None),
    user_id: Optional[int]     = Query(None),
    limit: int                 = Query(100, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Full paginated audit log with user attribution."""
    q = db.query(ClauseFeedback)
    if clause_type: q = q.filter(ClauseFeedback.clause_type == clause_type)
    if offering:    q = q.filter(ClauseFeedback.offering.ilike(f"%{offering}%"))
    if user_id:     q = q.filter(ClauseFeedback.user_id == user_id)
    return [
        feedback_to_dict(fb)
        for fb in q.order_by(ClauseFeedback.created_at.desc()).limit(limit).all()
    ]


# ══════════════════════════════════════════════════════════════════════════════
# LEARNED RULES — synthesise + manage
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/feedback/synthesise")
def synthesise_rule(
    body: SynthesiseRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """
    Trigger LLM rule synthesis for a specific (offering, solution, clause_type).

    The model reads all accumulated feedback + reviewer comments and rewrites
    the evaluation criteria for that context. The result is stored as a
    LearnedRule and applied automatically on all future pipeline runs for
    matching offering/solution combinations.

    Requires at least 5 feedback entries (use force=True to bypass).
    """
    from rules.learning_store import synthesise_rule as _synth
    return _synth(
        clause_type=body.clause_type,
        offering=body.offering,
        solution=body.solution,
        db=db,
        force=body.force,
    )


@router.get("/feedback/rules")
def list_rules(
    clause_type: Optional[str] = Query(None),
    offering: Optional[str]    = Query(None),
    active_only: bool          = Query(False),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """List all synthesised learned rules."""
    q = db.query(LearnedRule)
    if clause_type: q = q.filter(LearnedRule.clause_type == clause_type)
    if offering:    q = q.filter(LearnedRule.offering.ilike(f"%{offering.upper()}%"))
    if active_only: q = q.filter(LearnedRule.is_active == True)
    return [
        rule_to_dict(r)
        for r in q.order_by(LearnedRule.generated_at.desc()).all()
    ]


@router.patch("/feedback/rules/{rule_id}")
def update_rule(
    rule_id: int,
    body: RuleUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """
    Activate or deactivate a learned rule, or manually override its text.
    Deactivating is reversible — the rule is kept for history.
    """
    r = db.query(LearnedRule).filter(LearnedRule.id == rule_id).first()
    if not r: raise HTTPException(404, "Rule not found")
    if body.is_active is not None: r.is_active = body.is_active
    if body.rule_text:             r.rule_text  = body.rule_text
    db.commit()
    db.refresh(r)
    return rule_to_dict(r)


# ══════════════════════════════════════════════════════════════════════════════
# LEARNING EXAMPLES — view + manage
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/feedback/examples")
def list_examples(
    clause_type: Optional[str] = Query(None),
    offering: Optional[str]    = Query(None),
    active_only: bool          = Query(True),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """
    List curated few-shot examples currently being injected into LLM prompts.
    active_only=True (default) shows only examples currently in use.
    """
    q = db.query(LearningExample)
    if clause_type: q = q.filter(LearningExample.clause_type == clause_type)
    if offering:    q = q.filter(LearningExample.offering.ilike(f"%{offering.upper()}%"))
    if active_only: q = q.filter(LearningExample.is_active == True)
    return [
        example_to_dict(e)
        for e in q.order_by(
            LearningExample.usefulness_score.desc(),
            LearningExample.created_at.desc(),
        ).all()
    ]


@router.patch("/feedback/examples/{example_id}")
def update_example(
    example_id: int,
    body: ExampleUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """
    Promote a high-quality example (usefulness_score 1 → 2 or 3) so it
    appears earlier in prompts, or deactivate a poor one.
    """
    e = db.query(LearningExample).filter(LearningExample.id == example_id).first()
    if not e: raise HTTPException(404, "Example not found")
    if body.is_active is not None:
        e.is_active = body.is_active
    if body.usefulness_score is not None:
        if body.usefulness_score not in (1, 2, 3):
            raise HTTPException(400, "usefulness_score must be 1, 2, or 3")
        e.usefulness_score = body.usefulness_score
    db.commit()
    db.refresh(e)
    return example_to_dict(e)


# ══════════════════════════════════════════════════════════════════════════════
# USERS (admin only)
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/users")
def list_users(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    return [user_to_dict(u) for u in db.query(User).all()]


@router.post("/users")
def create_user(
    body: CreateUserRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(400, "Email already registered")
    if body.role not in ("admin", "reviewer", "tq_reviewer"):
        raise HTTPException(400, "Role must be admin or reviewer")
    u = User(
        name=body.name, email=body.email,
        password_hash=hash_password(body.password), role=body.role,
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return user_to_dict(u)


@router.delete("/users/{user_id}")
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    if user_id == current_user.id:
        raise HTTPException(400, "Cannot delete yourself")
    u = db.query(User).filter(User.id == user_id).first()
    if not u: raise HTTPException(404, "User not found")
    db.delete(u)
    db.commit()
    return {"deleted": True}

# ── DELETE any feedback record (admin) ────────────────────────────────────────
@router.delete("/feedback/reset-all")
def reset_all_feedback(
    confirm: str = Query(..., description="Must be 'yes-delete-everything'"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """
    Admin: wipe ALL feedback, learning examples, and learned rules globally.
    Requires ?confirm=yes-delete-everything as a safety guard.
    Use this to reset mock/demo data.
    """
    if confirm != "yes-delete-everything":
        raise HTTPException(400, "Pass ?confirm=yes-delete-everything to proceed.")

    examples_deleted = db.query(LearningExample).delete(synchronize_session=False)
    rules_deleted    = db.query(LearnedRule).delete(synchronize_session=False)
    feedback_deleted = db.query(ClauseFeedback).delete(synchronize_session=False)
    db.commit()

    print(f"[RESET] Deleted {feedback_deleted} feedback, "
          f"{examples_deleted} examples, {rules_deleted} rules")

    return {
        "feedback_deleted":  feedback_deleted,
        "examples_deleted":  examples_deleted,
        "rules_deleted":     rules_deleted,
    }
@router.delete("/feedback/{feedback_id}")
def admin_delete_feedback(
    feedback_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Admin: delete any single feedback record by ID."""
    fb = db.query(ClauseFeedback).filter(ClauseFeedback.id == feedback_id).first()
    if not fb:
        raise HTTPException(404, "Feedback not found")
    db.delete(fb)
    db.commit()
    return {"deleted": True, "feedback_id": feedback_id}


# ── RESET all feedback for one RFP (admin) ────────────────────────────────────

@router.delete("/rfps/{rfp_id}/feedback")
def reset_rfp_feedback(
    rfp_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Admin: wipe ALL feedback (+ learning examples) for one RFP. Useful for mock resets."""
    # Delete learning examples linked to this RFP's feedback
    fb_ids = [
        row.id for row in
        db.query(ClauseFeedback).filter(ClauseFeedback.rfp_id == rfp_id).all()
    ]
    if fb_ids:
        db.query(LearningExample).filter(
            LearningExample.feedback_id.in_(fb_ids)
        ).delete(synchronize_session=False)

    count = db.query(ClauseFeedback).filter(
        ClauseFeedback.rfp_id == rfp_id
    ).delete(synchronize_session=False)
    db.commit()
    return {"deleted": count, "rfp_id": rfp_id}


"""
PASTE THIS INTO routes.py
==========================

1. At the top of routes.py, update imports:

    from database import (
        get_db, User, RFP, ClauseResult, Comment,
        ClauseFeedback, LearningExample, LearnedRule,
        TQEvaluation, TQScoreItem,           # ADD
    )

    from auth import (
        verify_password, create_token, hash_password,
        get_current_user, require_admin,
        require_tq_access, require_tq_or_admin,  # ADD
    )

2. Replace rfp_to_dict() with the version below.

3. In create_user() update:
   OLD: if body.role not in ("admin", "reviewer"):
   NEW: if body.role not in ("admin", "reviewer", "tq_reviewer"):

4. Paste all functions below (serialisers + background task + routes).
"""

# ══════════════════════════════════════════════════════════════════════════════
# REPLACEMENT: rfp_to_dict — hides PQ clauses from tq_reviewer
# ══════════════════════════════════════════════════════════════════════════════

def rfp_to_dict(rfp, include_clauses=False, viewer_role="reviewer"):
    offerings, solutions = _parse_offering_solutions(rfp.offering or "", rfp.solutions or "")
    d = {
        "id": rfp.id,
        "opportunity_name": rfp.opportunity_name or "",
        "client_name":      rfp.client_name or "",
        "bu":               rfp.bu or "",
        "bu_code":          rfp.classification or "",
        "state":            rfp.state or "",
        "country":          rfp.country or "",
        "offerings":        offerings,
        "solutions":        solutions,
        "offering":         offerings[0] if offerings else (rfp.offering or ""),
        "solution":         solutions[0] if solutions else (rfp.solutions or ""),
        "offering_raw":     rfp.offering or "",
        "solutions_raw":    rfp.solutions or "",
        "file_name":        rfp.file_name or "",
        "job_id":           rfp.job_id or "",
        "status":           rfp.status or "queued",
        "progress":         rfp.progress or 0,
        "current_step":     rfp.current_step or "",
        "error_message":    rfp.error_message,
        "created_at":       rfp.created_at.isoformat() if rfp.created_at else None,
        "viewer_role":      viewer_role,
    }
    d["uploaded_by_name"] = (rfp.uploaded_by_user.name if rfp.uploaded_by_user else "") \
        if viewer_role == "admin" else None

    # tq_reviewer: no PQ clause details, only TQ
    if include_clauses and viewer_role != "tq_reviewer":
        d["clauses"] = {c.clause_type: clause_to_dict(c) for c in rfp.clause_results}
    else:
        d["clauses"] = {}

    return d


# ══════════════════════════════════════════════════════════════════════════════
# TQ BACKGROUND TASK
# ══════════════════════════════════════════════════════════════════════════════

TQ_UPLOAD_DIR = Path("./tq_uploads")
TQ_UPLOAD_DIR.mkdir(exist_ok=True)


import json
from pathlib import Path
from datetime import datetime as _dt
 
 
import json
from pathlib import Path
from datetime import datetime as _dt
 
# ─────────────────────────────────────────────────────────────────────────────
# TQ Serialisers  (replace the existing versions in routes.py)
# ─────────────────────────────────────────────────────────────────────────────
 
def tq_score_item_to_dict(item) -> dict:
    """Serialise one TQScoreItem row to the API response shape."""
    try:    strengths = json.loads(item.strengths_json or "[]")
    except: strengths = []
    try:    gaps = json.loads(item.gaps_json or "[]")
    except: gaps = []
    try:    discrepancies = json.loads(getattr(item, "discrepancies_json", "[]") or "[]")
    except: discrepancies = []
 
    raw_score      = item.score
    is_pending     = raw_score in ("-1", -1) or getattr(item, "requires_live_assessment", False)
    is_comparative = raw_score in ("-2", -2) or getattr(item, "requires_comparative_evaluation", False)
 
    score_val = None if (is_pending or is_comparative) else float(raw_score or 0)
    pct_val   = None if (is_pending or is_comparative) else float(item.score_percentage or 0)
 
    layer = getattr(item, "evaluation_layer", None) or "technical_document"
 
    return {
        "id":                              item.id,
        "item_code":                       item.item_code or "",
        "parameter":                       item.parameter or "",
        "max_marks":                       item.max_marks or 0,
        "score":                           score_val,
        "score_percentage":                pct_val,
        "justification":                   item.justification or "",
        "strengths":                       strengths,
        "gaps":                            gaps,
        "discrepancies":                   discrepancies,   # NEW
        "formula_hint":                    getattr(item, "formula_hint", "") or "",  # NEW
        "evidence_found":                  item.evidence_found or False,
        "is_sub_item":                     item.is_sub_item or False,
        "parent_parameter":                item.parent_parameter or "",
        "criteria_text":                   item.criteria_text or "",
        "sort_order":                      item.sort_order or 0,
        "evaluation_layer":                layer,
        "requires_live_assessment":        is_pending,
        "requires_comparative_evaluation": is_comparative,
    }
 
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Evaluation serialiser
# ─────────────────────────────────────────────────────────────────────────────
 
def tq_evaluation_to_dict(ev, include_scores: bool = True) -> dict:
    """Serialise a TQEvaluation row to the full API response shape."""
 
    def _g(attr, default=None):
        return getattr(ev, attr, default)
 
    qualification = None
    try:
        raw_q = _g("qualification_json")
        if raw_q: qualification = json.loads(raw_q)
    except Exception: pass
 
    financial_evaluation = None
    try:
        raw_fe = _g("financial_methodology_json")
        if raw_fe: financial_evaluation = json.loads(raw_fe)
    except Exception: pass
 
    # Global discrepancies (new field)
    global_discrepancies = []
    try:
        raw_gd = _g("global_discrepancies_json")
        if raw_gd: global_discrepancies = json.loads(raw_gd)
    except Exception: pass
 
    total_scored_raw = ev.total_scored or "0"
    total_pct_raw    = ev.total_percentage or "0"
    total_scored_val = float(total_scored_raw) if total_scored_raw not in ("-1", None) else 0.0
    total_pct_val    = float(total_pct_raw)    if total_pct_raw    not in ("-1", None) else 0.0
 
    d = {
        "id":                     ev.id,
        "rfp_id":                 ev.rfp_id,
        "proposal_file_name":     ev.proposal_file_name or "",
        "evaluation_title":       ev.evaluation_title or "Technical Evaluation",
        "grand_total_marks":      ev.grand_total_marks or 100,
        "technical_document_max": _g("scoreable_total", 0) or 0,
        "live_assessment_marks":  _g("live_assessment_marks", 0) or 0,
        "financial_marks":        _g("financial_marks", 0) or 0,
        "total_scored":           total_scored_val,
        "total_percentage":       total_pct_val,
        "final_score_formula":    _g("final_score_formula"),
        "financial_evaluation":   financial_evaluation,
        "qualification":          qualification,
        "schema_valid":           _g("schema_valid"),
        "schema_warning":         _g("schema_warning"),
        "global_discrepancies":   global_discrepancies,       # NEW
        "discrepancy_count":      len(global_discrepancies),  # NEW
        "status":                 ev.status or "queued",
        "progress":               ev.progress or 0,
        "current_step":           ev.current_step or "",
        "error_message":          ev.error_message,
        "uploaded_by_name":       ev.uploader.name if ev.uploader else "",
        "created_at":             ev.created_at.isoformat() if ev.created_at else None,
        "completed_at":           ev.completed_at.isoformat() if ev.completed_at else None,
    }
    if include_scores:
        d["scores"] = [tq_score_item_to_dict(s) for s in (ev.scores or [])]
    return d
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Background task
# ─────────────────────────────────────────────────────────────────────────────
 
TQ_UPLOAD_DIR = Path("./tq_uploads")
TQ_UPLOAD_DIR.mkdir(exist_ok=True)
 
 
def run_tq_evaluation_task(
    tq_eval_id:       int,
    rfp_doc_name:     str,
    proposal_path:    str,
    proposal_doc_name: str,
):
    """
    Background task: full TQ evaluation pipeline.
 
    Session lifecycle:
    - Short-lived sessions for progress updates (never held during Ollama calls)
    - Single final session for persisting all results
    """
    from database import SessionLocal, TQEvaluation, TQScoreItem
    from core.tq_extractor import run_tq_evaluation
 
    # ── Mark processing ───────────────────────────────────────────────────────
    db = SessionLocal()
    try:
        ev = db.query(TQEvaluation).filter(TQEvaluation.id == tq_eval_id).first()
        if not ev: return
        ev.status = "processing"
        db.commit()
    finally:
        db.close()
 
    # ── Progress callback (own short-lived session each call) ─────────────────
    def _progress(step: str, pct: int):
        _db = SessionLocal()
        try:
            _ev = _db.query(TQEvaluation).filter(TQEvaluation.id == tq_eval_id).first()
            if _ev:
                _ev.current_step = step
                _ev.progress     = pct
                _db.commit()
        except Exception:
            pass
        finally:
            try: _db.close()
            except Exception: pass
 
    # ── Run evaluation (no open DB session during Ollama calls) ──────────────
    result = None
    try:
        result = run_tq_evaluation(
            rfp_doc_name=rfp_doc_name,
            proposal_path=proposal_path,
            proposal_doc_name=proposal_doc_name,
            progress_callback=_progress,
        )
    except Exception as e:
        print(f"[TQ] run_tq_evaluation raised: {e}")
        _db = SessionLocal()
        try:
            _db.query(TQEvaluation).filter(TQEvaluation.id == tq_eval_id).update({
                "status": "failed", "error_message": str(e), "current_step": "Failed"
            })
            _db.commit()
        except Exception: pass
        finally:
            try: _db.close()
            except Exception: pass
        return
 
    # ── Persist results ───────────────────────────────────────────────────────
    db = SessionLocal()
    try:
        ev = db.query(TQEvaluation).filter(TQEvaluation.id == tq_eval_id).first()
        if not ev: return
 
        for i, scored in enumerate(result.get("scores", [])):
            layer      = scored.get("evaluation_layer", "technical_document")
            raw_score  = scored.get("score")
            is_pending = scored.get("requires_live_assessment", False) or raw_score is None
            is_comp    = scored.get("requires_comparative_evaluation", False)
 
            if is_comp:     stored_score = "-2"; stored_pct = "-2"
            elif is_pending or raw_score is None:
                             stored_score = "-1"; stored_pct = "-1"
            else:
                stored_score = str(raw_score)
                stored_pct   = str(scored.get("score_percentage", 0))
 
            item = TQScoreItem(
                evaluation_id    = tq_eval_id,
                item_code        = scored.get("item_code", ""),
                parameter        = scored.get("parameter", ""),
                max_marks        = scored.get("max_marks", 0),
                score            = stored_score,
                score_percentage = stored_pct,
                justification    = scored.get("justification", ""),
                strengths_json   = json.dumps(scored.get("strengths", [])),
                gaps_json        = json.dumps(scored.get("gaps", [])),
                evidence_found   = bool(scored.get("evidence_found", False)),
                is_sub_item      = bool(scored.get("is_sub_item", False)),
                parent_parameter = scored.get("parent_parameter", ""),
                criteria_text    = scored.get("criteria_text", ""),
                sort_order       = i,
                requires_live_assessment = is_pending,
            )
 
            # New fields — set safely (columns added by migration)
            def _set(obj, attr, val):
                try: setattr(obj, attr, val)
                except Exception: pass
 
            _set(item, "evaluation_layer",                layer)
            _set(item, "requires_comparative_evaluation", is_comp)
            _set(item, "discrepancies_json",
                 json.dumps(scored.get("discrepancies", [])))
            _set(item, "formula_hint",
                 scored.get("formula_hint", ""))
 
            db.add(item)
 
        # ── Update evaluation record ──────────────────────────────────────────
        ev.evaluation_title  = result.get("evaluation_title", "Technical Evaluation")
        ev.grand_total_marks = result.get("grand_total_marks", 100)
        ev.total_scored      = str(result.get("total_scored", 0))
        ev.total_percentage  = str(result.get("total_percentage", 0))
        ev.status            = "completed"
        ev.progress          = 100
        ev.current_step      = "Evaluation complete"
        ev.completed_at      = _dt.utcnow()
        ev.error_message     = result.get("error")
 
        def _sev(attr, val):
            try: setattr(ev, attr, val)
            except Exception: pass
 
        _sev("scoreable_total",       result.get("technical_document_max", 0))
        _sev("live_assessment_marks", result.get("live_assessment_marks", 0))
        _sev("financial_marks",       result.get("financial_marks", 0))
        _sev("schema_valid",          result.get("schema_valid", False))
        _sev("schema_warning",        result.get("schema_warning"))
        _sev("final_score_formula",   result.get("final_score_formula"))
        _sev("financial_methodology_json",
             json.dumps(result.get("financial_evaluation") or {}))
        _sev("qualification_json",
             json.dumps(result.get("qualification") or {}))
        _sev("global_discrepancies_json",           # NEW
             json.dumps(result.get("global_discrepancies") or []))
 
        db.commit()
 
        td_max = result.get("technical_document_max", ev.grand_total_marks)
        discs  = result.get("global_discrepancies") or []
        print(f"[TQ] Eval {tq_eval_id} persisted → "
              f"{ev.total_scored}/{td_max} ({ev.total_percentage}%) | "
              f"{len(discs)} discrepancy/ies")
 
    except Exception as e:
        print(f"[TQ] Persist error eval {tq_eval_id}: {e}")
        try:
            db.query(TQEvaluation).filter(TQEvaluation.id == tq_eval_id).update({
                "status": "failed", "error_message": str(e), "current_step": "Failed"
            })
            db.commit()
        except Exception: pass
    finally:
        try: db.close()
        except Exception: pass
 


# ══════════════════════════════════════════════════════════════════════════════
# TQ API ROUTES — paste below the existing USERS section in routes.py
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/rfps/{rfp_id}/tq/upload")
async def upload_tq_proposal(
    rfp_id: int,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_tq_or_admin),
):
    rfp = db.query(RFP).filter(RFP.id == rfp_id).first()
    if not rfp:
        raise HTTPException(404, "RFP not found")
    if rfp.status not in ("completed", "failed"):
        raise HTTPException(400, "RFP must finish processing before uploading a proposal.")
 
    form = await request.form()
    file_obj = (form.get("file") or form.get("proposal")
                or form.get("proposal_file") or form.get("document"))
    if file_obj is None:
        for k in form.keys():
            v = form[k]
            if hasattr(v, "filename") and v.filename:
                file_obj = v
                break
    if file_obj is None:
        raise HTTPException(400, "No proposal file in upload.")
 
    ext = Path(file_obj.filename).suffix.lower()
    if ext not in (".pdf", ".docx", ".doc"):
        raise HTTPException(400, "Only PDF and DOCX are supported.")
 
    proposal_uid      = str(uuid.uuid4())[:8]
    proposal_doc_name = f"proposal_{rfp.job_id}_{proposal_uid}{ext}"
    save_path         = TQ_UPLOAD_DIR / proposal_doc_name
 
    with open(save_path, "wb") as fp:
        fp.write(await file_obj.read())
 
    original_ext = Path(rfp.file_name).suffix if rfp.file_name else ".pdf"
    rfp_doc_name = f"{rfp.job_id}{original_ext}"
 
    ev = TQEvaluation(
        rfp_id=rfp_id,
        proposal_file_name=file_obj.filename,
        proposal_doc_name=proposal_doc_name,
        evaluation_title="Technical Evaluation",
        grand_total_marks=100,
        total_scored="0",
        total_percentage="0",
        scoreable_total=0,
        live_assessment_marks=0,
        schema_valid=False,
        status="queued",
        progress=0,
        current_step="Queued — waiting to start",
        uploaded_by=current_user.id,
    )
    db.add(ev)
    db.commit()
    db.refresh(ev)
 
    background_tasks.add_task(
        run_tq_evaluation_task,
        ev.id,
        rfp_doc_name,
        str(save_path),
        proposal_doc_name,
    )
 
    return {
        "tq_evaluation_id":   ev.id,
        "rfp_id":             rfp_id,
        "proposal_file_name": file_obj.filename,
        "status":             "queued",
        "message":            "Proposal uploaded. TQ evaluation started in background.",
    }
 
 
@router.get("/rfps/{rfp_id}/tq")
def get_tq_evaluations(
    rfp_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_tq_access),
):
    if not db.query(RFP).filter(RFP.id == rfp_id).first():
        raise HTTPException(404, "RFP not found")
    evaluations = (
        db.query(TQEvaluation)
        .filter(TQEvaluation.rfp_id == rfp_id)
        .order_by(TQEvaluation.created_at.desc())
        .all()
    )
    return {
        "rfp_id": rfp_id,
        "total_evaluations": len(evaluations),
        "evaluations": [tq_evaluation_to_dict(ev, include_scores=False)
                        for ev in evaluations],
        "latest": tq_evaluation_to_dict(evaluations[0]) if evaluations else None,
    }
 
 
def build_tq_status_response(ev) -> dict:
    qualification = None
    try:
        raw_q = getattr(ev, "qualification_json", None)
        if raw_q:
            qualification = json.loads(raw_q)
    except Exception:
        pass
    return {
        "id":                     ev.id,
        "status":                 ev.status,
        "progress":               ev.progress,
        "current_step":           ev.current_step,
        "error_message":          ev.error_message,
        "total_scored":           float(ev.total_scored or 0),
        "total_percentage":       float(ev.total_percentage or 0),
        "grand_total_marks":      ev.grand_total_marks,
        "technical_document_max": getattr(ev, "scoreable_total", 0) or 0,
        "live_assessment_marks":  getattr(ev, "live_assessment_marks", 0) or 0,
        "financial_marks":        getattr(ev, "financial_marks", 0) or 0,
        "final_score_formula":    getattr(ev, "final_score_formula", None),
        "schema_valid":           getattr(ev, "schema_valid", None),
        "qualification":          qualification,
        "financial_bid_opens":    (qualification or {}).get("financial_bid_opens"),
    }
 
 
@router.get("/rfps/{rfp_id}/tq/{evaluation_id}")
def get_tq_evaluation_detail(
    rfp_id: int,
    evaluation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_tq_access),
):
    ev = db.query(TQEvaluation).filter(
        TQEvaluation.id == evaluation_id,
        TQEvaluation.rfp_id == rfp_id,
    ).first()
    if not ev:
        raise HTTPException(404, "TQ evaluation not found")
    return tq_evaluation_to_dict(ev, include_scores=True)
 
 
@router.get("/rfps/{rfp_id}/tq/{evaluation_id}/status")
def get_tq_status(
    rfp_id: int,
    evaluation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_tq_access),
):
    ev = db.query(TQEvaluation).filter(
        TQEvaluation.id == evaluation_id,
        TQEvaluation.rfp_id == rfp_id,
    ).first()
    if not ev:
        raise HTTPException(404, "TQ evaluation not found")
    return {
        "id":                    ev.id,
        "status":                ev.status,
        "progress":              ev.progress,
        "current_step":          ev.current_step,
        "error_message":         ev.error_message,
        "total_scored":          float(ev.total_scored or 0),
        "total_percentage":      float(ev.total_percentage or 0),
        "grand_total_marks":     ev.grand_total_marks,
        "scoreable_total":       getattr(ev, "scoreable_total", 0) or 0,
        "live_assessment_marks": getattr(ev, "live_assessment_marks", 0) or 0,
        "schema_valid":          getattr(ev, "schema_valid", None),
    }
 
 
@router.patch("/rfps/{rfp_id}/tq/{evaluation_id}/scores/{score_id}")
async def override_tq_score(
    rfp_id: int,
    evaluation_id: int,
    score_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_tq_access),
):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")
 
    item = db.query(TQScoreItem).filter(
        TQScoreItem.id == score_id,
        TQScoreItem.evaluation_id == evaluation_id,
    ).first()
    if not item:
        raise HTTPException(404, "Score item not found")
 
    ev = db.query(TQEvaluation).filter(
        TQEvaluation.id == evaluation_id,
        TQEvaluation.rfp_id == rfp_id,
    ).first()
    if not ev:
        raise HTTPException(404, "Evaluation not found")
 
    new_score = float(body.get("score", item.score or 0))
    if new_score < 0 or new_score > (item.max_marks or 0):
        raise HTTPException(400, f"Score must be 0–{item.max_marks}")
 
    item.score            = str(round(new_score, 1))
    item.score_percentage = str(
        round((new_score / item.max_marks) * 100, 1) if item.max_marks else 0
    )
    item.requires_live_assessment = False  # override clears pending status
 
    override_reason = body.get("override_reason", "")
    if override_reason:
        item.justification = f"[MANUALLY OVERRIDDEN by {current_user.name}] {override_reason}"
 
    # Recompute evaluation total (only scored, non-pending items)
    db.flush()
    all_scores = db.query(TQScoreItem).filter(
        TQScoreItem.evaluation_id == evaluation_id
    ).all()
    new_total = sum(
        float(s.score or 0)
        for s in all_scores
        if s.score not in ("-1", None)
    )
    ev.total_scored     = str(round(new_total, 1))
    scoreable            = sum(s.max_marks for s in all_scores if s.score not in ("-1", None))
    ev.total_percentage = str(
        round((new_total / scoreable) * 100, 1) if scoreable else 0
    )
    db.commit()
 
    return {
        "updated_score":        tq_score_item_to_dict(item),
        "new_total_scored":     float(ev.total_scored),
        "new_total_percentage": float(ev.total_percentage),
        "grand_total_marks":    ev.grand_total_marks,
    }
 
 
@router.delete("/rfps/{rfp_id}/tq/{evaluation_id}")
def delete_tq_evaluation(
    rfp_id: int,
    evaluation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    ev = db.query(TQEvaluation).filter(
        TQEvaluation.id == evaluation_id,
        TQEvaluation.rfp_id == rfp_id,
    ).first()
    if not ev:
        raise HTTPException(404, "TQ evaluation not found")
    db.delete(ev)
    db.commit()
    return {"deleted": True, "evaluation_id": evaluation_id}
 
 
@router.get("/tq/evaluations")
def list_all_tq_evaluations(
    status: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    q = db.query(TQEvaluation)
    if status:
        q = q.filter(TQEvaluation.status == status)
    evs = q.order_by(TQEvaluation.created_at.desc()).limit(limit).all()
    return [tq_evaluation_to_dict(ev, include_scores=False) for ev in evs]