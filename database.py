"""
Database
--------
SQLAlchemy models + Postgres connection.
Tables are auto-created on first run.

Tables:
  users             — auth
  rfps              — uploaded tenders
  clause_results    — extracted + risk-evaluated clause data (SSC1/PQ)
  comments          — per-clause reviewer comments
  clause_feedback   — structured reviewer feedback
  learning_examples — curated few-shot examples for prompt injection
  learned_rules     — LLM-synthesised evaluation rules per offering/solution/clause
  tq_evaluations    — SSC2: one TQ evaluation per RFP + proposal upload
  tq_score_items    — SSC2: one scored criterion per evaluation

FIX NOTES (v4 — v20 extractor)
---------
- annexure_pages_json added to tq_score_items: stores page references from
  the proposal's compliance table response text (e.g. [87, 95, 452]).
  Used by future improvements for document verification and auditing.
- All previous v3 columns retained unchanged.
"""

import os
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Text,
    Boolean, DateTime, ForeignKey,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set in .env file")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=1800,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        raise
    finally:
        try:
            db.close()
        except Exception:
            pass


def get_fresh_db():
    """Non-generator session for background tasks — caller must close()."""
    return SessionLocal()


# ═════════════════════════════════════════════════════════════════════════════
# Tables
# ═════════════════════════════════════════════════════════════════════════════

class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    name          = Column(String(100), nullable=False)
    email         = Column(String(150), unique=True, index=True, nullable=False)
    password_hash = Column(String(200), nullable=False)
    role          = Column(String(20), default="reviewer")
    created_at    = Column(DateTime, default=datetime.utcnow)

    rfps      = relationship("RFP", back_populates="uploaded_by_user")
    comments  = relationship("Comment", back_populates="user")
    feedbacks = relationship("ClauseFeedback", back_populates="user")


class RFP(Base):
    __tablename__ = "rfps"

    id               = Column(Integer, primary_key=True, index=True)
    opportunity_name = Column(String(300), nullable=False)
    client_name      = Column(String(150))
    bu               = Column(String(150))
    classification   = Column(String(50))
    state            = Column(String(100))
    country          = Column(String(100))
    offering         = Column(String(500))
    solutions        = Column(String(500))
    file_name        = Column(String(300))
    job_id           = Column(String(50), unique=True, index=True)
    status           = Column(String(30), default="queued")
    progress         = Column(Integer, default=0)
    current_step     = Column(String(100), default="")
    error_message    = Column(Text, nullable=True)
    uploaded_by      = Column(Integer, ForeignKey("users.id"))
    created_at       = Column(DateTime, default=datetime.utcnow)

    uploaded_by_user = relationship("User", back_populates="rfps")
    clause_results   = relationship("ClauseResult", back_populates="rfp", cascade="all, delete")
    comments         = relationship("Comment", back_populates="rfp", cascade="all, delete")
    feedbacks        = relationship("ClauseFeedback", back_populates="rfp", cascade="all, delete")
    tq_evaluations   = relationship("TQEvaluation", back_populates="rfp", cascade="all, delete")


class ClauseResult(Base):
    __tablename__ = "clause_results"

    id                    = Column(Integer, primary_key=True, index=True)
    rfp_id                = Column(Integer, ForeignKey("rfps.id"), nullable=False)
    clause_type           = Column(String(50))
    clause_text           = Column(Text)
    clause_reference      = Column(String(200))
    page_no               = Column(String(20))
    risk_level            = Column(String(30))
    risk_description      = Column(Text)
    auto_remark           = Column(Text)
    needs_exception       = Column(Boolean, default=False)
    needs_eqcr            = Column(Boolean, default=False)
    deviation_suggested   = Column(Text)

    adjusted_risk_level   = Column(String(30), nullable=True)
    adjustment_reason     = Column(Text, nullable=True)
    adjustment_confidence = Column(String(10), nullable=True)
    feedback_count        = Column(Integer, default=0)

    learned_rule_applied  = Column(Boolean, default=False)
    learned_rule_id       = Column(Integer, ForeignKey("learned_rules.id"), nullable=True)

    rfp          = relationship("RFP", back_populates="clause_results")
    feedbacks    = relationship("ClauseFeedback", back_populates="clause_result",
                                foreign_keys="ClauseFeedback.clause_result_id")
    learned_rule = relationship("LearnedRule", foreign_keys=[learned_rule_id])


class Comment(Base):
    __tablename__ = "comments"

    id           = Column(Integer, primary_key=True, index=True)
    rfp_id       = Column(Integer, ForeignKey("rfps.id"), nullable=False)
    clause_type  = Column(String(50))
    user_id      = Column(Integer, ForeignKey("users.id"), nullable=False)
    comment_text = Column(Text, nullable=False)
    created_at   = Column(DateTime, default=datetime.utcnow)

    rfp  = relationship("RFP", back_populates="comments")
    user = relationship("User", back_populates="comments")


class ClauseFeedback(Base):
    __tablename__ = "clause_feedback"

    id                   = Column(Integer, primary_key=True, index=True)
    rfp_id               = Column(Integer, ForeignKey("rfps.id"), nullable=False)
    clause_result_id     = Column(Integer, ForeignKey("clause_results.id"), nullable=True)
    clause_type          = Column(String(50), nullable=False, index=True)
    user_id              = Column(Integer, ForeignKey("users.id"), nullable=False)

    offering             = Column(String(500))
    solution             = Column(String(500))
    bu                   = Column(String(150))

    agreement            = Column(String(20), nullable=False)
    suggested_risk_level = Column(String(30), nullable=True)
    feedback_comment     = Column(Text, nullable=True)
    system_risk_level    = Column(String(30), nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    rfp              = relationship("RFP", back_populates="feedbacks")
    user             = relationship("User", back_populates="feedbacks")
    clause_result    = relationship("ClauseResult", back_populates="feedbacks",
                                    foreign_keys=[clause_result_id])
    learning_example = relationship("LearningExample", back_populates="feedback",
                                    uselist=False)


class LearningExample(Base):
    __tablename__ = "learning_examples"

    id                 = Column(Integer, primary_key=True, index=True)
    feedback_id        = Column(Integer, ForeignKey("clause_feedback.id"), nullable=True)
    clause_type        = Column(String(50), nullable=False, index=True)
    offering           = Column(String(500), index=True)
    solution           = Column(String(500))
    bu                 = Column(String(150))

    clause_snippet     = Column(Text)
    system_risk_level  = Column(String(30))
    correct_risk_level = Column(String(30))
    reviewer_reason    = Column(Text)

    usefulness_score   = Column(Integer, default=1)
    is_active          = Column(Boolean, default=True)
    created_at         = Column(DateTime, default=datetime.utcnow)

    feedback = relationship("ClauseFeedback", back_populates="learning_example")


class LearnedRule(Base):
    __tablename__ = "learned_rules"

    id                    = Column(Integer, primary_key=True, index=True)
    clause_type           = Column(String(50), nullable=False, index=True)
    offering              = Column(String(500), index=True)
    solution              = Column(String(500))

    rule_text             = Column(Text, nullable=False)
    threshold_notes_json  = Column(Text)
    key_differences       = Column(Text)
    confidence            = Column(String(10))

    feedback_count_at_gen = Column(Integer, default=0)
    is_active             = Column(Boolean, default=True)
    generated_at          = Column(DateTime, default=datetime.utcnow)


# ═════════════════════════════════════════════════════════════════════════════
# SSC2 / TQ Tables
# ═════════════════════════════════════════════════════════════════════════════

class TQEvaluation(Base):
    """One TQ evaluation per RFP + proposal upload."""
    __tablename__ = "tq_evaluations"

    id                 = Column(Integer, primary_key=True, index=True)
    rfp_id             = Column(Integer, ForeignKey("rfps.id"), nullable=False)
    proposal_file_name = Column(String(300))
    proposal_doc_name  = Column(String(300))
    evaluation_title   = Column(String(300), default="Technical Evaluation")
    grand_total_marks  = Column(Integer, default=100)
    total_scored       = Column(String(20), default="0")
    total_percentage   = Column(String(20), default="0")

    # Denominator breakdown (v3)
    scoreable_total          = Column(Integer, default=0)
    live_assessment_marks    = Column(Integer, default=0)

    # Schema quality signal (v3)
    schema_valid             = Column(Boolean, default=False)
    schema_warning           = Column(Text, nullable=True)

    # Financial methodology (v3)
    financial_methodology_json = Column(Text, nullable=True)

    # Qualification gate result (v3)
    qualification_json       = Column(Text, nullable=True)

    # Financial marks count (v3)
    financial_marks          = Column(Integer, default=0)

    status             = Column(String(30), default="queued")
    progress           = Column(Integer, default=0)
    current_step       = Column(String(200), default="")
    error_message      = Column(Text, nullable=True)
    uploaded_by        = Column(Integer, ForeignKey("users.id"))
    created_at         = Column(DateTime, default=datetime.utcnow)
    completed_at       = Column(DateTime, nullable=True)

    rfp      = relationship("RFP", back_populates="tq_evaluations")
    uploader = relationship("User", foreign_keys=[uploaded_by])
    scores   = relationship(
        "TQScoreItem", back_populates="evaluation",
        cascade="all, delete",
        order_by="TQScoreItem.sort_order",
    )


class TQScoreItem(Base):
    """One scored criterion per TQEvaluation."""
    __tablename__ = "tq_score_items"

    id                       = Column(Integer, primary_key=True, index=True)
    evaluation_id            = Column(Integer, ForeignKey("tq_evaluations.id"), nullable=False)
    item_code                = Column(String(20))
    parameter                = Column(String(300))
    max_marks                = Column(Integer, default=0)
    score                    = Column(String(20), default="0")     # "-1" means pending
    score_percentage         = Column(String(20), default="0")
    justification            = Column(Text)
    strengths_json           = Column(Text, default="[]")
    gaps_json                = Column(Text, default="[]")
    evidence_found           = Column(Boolean, default=False)
    is_sub_item              = Column(Boolean, default=False)
    parent_parameter         = Column(String(300), default="")
    criteria_text            = Column(Text)
    sort_order               = Column(Integer, default=0)
    requires_live_assessment = Column(Boolean, default=False)

    # v4: annexure page references extracted from proposal compliance table
    # JSON array of integers: [87, 95, 303, 345, 452]
    annexure_pages_json      = Column(Text, default="[]")

    evaluation = relationship("TQEvaluation", back_populates="scores")


# ═════════════════════════════════════════════════════════════════════════════
# Init
# ═════════════════════════════════════════════════════════════════════════════

def init_db():
    Base.metadata.create_all(bind=engine)
    _safe_alter_columns()

    from auth import hash_password
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == "admin@grantthornton.in").first()
        if not existing:
            admin = User(
                name="Admin",
                email="admin@grantthornton.in",
                password_hash=hash_password("Admin@123"),
                role="admin",
            )
            db.add(admin)
            db.commit()
            print("[DB] Seeded default admin: admin@grantthornton.in / Admin@123")
        else:
            print("[DB] Tables ready.")
    finally:
        db.close()


def _safe_alter_columns():
    """Idempotent schema migrations — safe to run on every startup."""
    from sqlalchemy import text
    alterations = [
        # SSC1 / PQ
        "ALTER TABLE rfps ADD COLUMN IF NOT EXISTS country VARCHAR(100)",
        "ALTER TABLE clause_results ADD COLUMN IF NOT EXISTS adjusted_risk_level VARCHAR(30)",
        "ALTER TABLE clause_results ADD COLUMN IF NOT EXISTS adjustment_reason TEXT",
        "ALTER TABLE clause_results ADD COLUMN IF NOT EXISTS adjustment_confidence VARCHAR(10)",
        "ALTER TABLE clause_results ADD COLUMN IF NOT EXISTS feedback_count INTEGER DEFAULT 0",
        "ALTER TABLE clause_results ADD COLUMN IF NOT EXISTS learned_rule_applied BOOLEAN DEFAULT FALSE",
        "ALTER TABLE clause_results ADD COLUMN IF NOT EXISTS learned_rule_id INTEGER",
        "ALTER TABLE users ALTER COLUMN role TYPE VARCHAR(20)",

        # SSC2 / TQ evaluations — v1 columns
        "ALTER TABLE tq_evaluations ADD COLUMN IF NOT EXISTS completed_at TIMESTAMP",

        # SSC2 / TQ evaluations — v3 columns
        "ALTER TABLE tq_evaluations ADD COLUMN IF NOT EXISTS scoreable_total INTEGER DEFAULT 0",
        "ALTER TABLE tq_evaluations ADD COLUMN IF NOT EXISTS live_assessment_marks INTEGER DEFAULT 0",
        "ALTER TABLE tq_evaluations ADD COLUMN IF NOT EXISTS schema_valid BOOLEAN DEFAULT FALSE",
        "ALTER TABLE tq_evaluations ADD COLUMN IF NOT EXISTS schema_warning TEXT",
        "ALTER TABLE tq_evaluations ADD COLUMN IF NOT EXISTS financial_methodology_json TEXT",
        "ALTER TABLE tq_evaluations ADD COLUMN IF NOT EXISTS qualification_json TEXT",
        "ALTER TABLE tq_evaluations ADD COLUMN IF NOT EXISTS financial_marks INTEGER DEFAULT 0",
        "ALTER TABLE tq_evaluations ADD COLUMN IF NOT EXISTS final_score_formula TEXT",

        # SSC2 / TQ score items — v1 columns
        "ALTER TABLE tq_score_items ADD COLUMN IF NOT EXISTS criteria_text TEXT",
        "ALTER TABLE tq_score_items ADD COLUMN IF NOT EXISTS parent_parameter VARCHAR(300) DEFAULT ''",

        # SSC2 / TQ score items — v3 columns
        "ALTER TABLE tq_score_items ADD COLUMN IF NOT EXISTS requires_live_assessment BOOLEAN DEFAULT FALSE",
        "ALTER TABLE tq_score_items ADD COLUMN IF NOT EXISTS evaluation_layer VARCHAR(50) DEFAULT 'technical_document'",
        "ALTER TABLE tq_score_items ADD COLUMN IF NOT EXISTS requires_comparative_evaluation BOOLEAN DEFAULT FALSE",

        # SSC2 / TQ score items — v4 columns (v20 extractor)
        # Stores JSON array of page numbers from proposal compliance table
        # e.g. [87, 95, 303, 345, 452, 474] — annexures for this criterion
        "ALTER TABLE tq_score_items ADD COLUMN IF NOT EXISTS annexure_pages_json TEXT DEFAULT '[]'",
    ]
    try:
        with engine.connect() as conn:
            for stmt in alterations:
                try:
                    conn.execute(text(stmt))
                except Exception:
                    pass
            conn.commit()
    except Exception as e:
        print(f"[DB] _safe_alter_columns warning: {e}")