"""
core/knowledge_graph.py
========================
Corporate / Legal Knowledge Graph
===================================

Builds a NetworkX knowledge graph of entities and relationships
extracted from parsed document chunks. Used to:

  1. Disambiguate corporate acronyms (PMA, PMU, GTBL, NHB, MoFPI...)
  2. Relate contract terms to their legal implications
  3. Enrich LLM prompts with entity context
  4. Cluster semantically related clauses

Graph Schema
------------
Nodes:
  - Entity (org, person, program, act, clause)
  - Term (legal/technical abbreviation)
  - RiskConcept (liability, LD, termination...)

Edges:
  - DEFINED_AS (Term → definition text)
  - PART_OF    (program → parent org)
  - GOVERNS    (act → entity)
  - RELATES_TO (clause → risk concept)
  - SYNONYM_OF (abbreviation → full form)
  - IMPLIES    (condition → risk level)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    print("[KG] networkx not installed — pip install networkx")


# ─────────────────────────────────────────────────────────────────────────────
# Built-in domain ontology (GTBL / Indian government / RFP context)
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_ONTOLOGY = {
    # Organisational abbreviations
    "abbreviations": {
        "GTBL":    "Grant Thornton Bharat LLP (formerly Grant Thornton India LLP)",
        "GT":      "Grant Thornton",
        "GTIL":    "Grant Thornton International Limited",
        "MoFPI":   "Ministry of Food Processing Industries",
        "NHB":     "National Horticulture Board",
        "SFAC":    "Small Farmers' Agri-Business Consortium",
        "NITI":    "National Institution for Transforming India",
        "ADB":     "Asian Development Bank",
        "WB":      "World Bank",
        "UNDP":    "United Nations Development Programme",
        "AMRUT":   "Atal Mission for Rejuvenation and Urban Transformation",
        "PMGSY":   "Pradhan Mantri Gram Sadak Yojana",
        "PMAY":    "Pradhan Mantri Awas Yojana",
        "ULB":     "Urban Local Body",
        "PMC":     "Programme Management Consultancy",
        "PMU":     "Programme Management Unit",
        "PMA":     "Programme Management Agency",
        "SSC1":    "Strategy Steering Council — Pre-Qualification Risk Review",
        "SSC2":    "Strategy Steering Council — Technical Quality Review",
        "RFP":     "Request for Proposal",
        "RFQ":     "Request for Quotation",
        "EOI":     "Expression of Interest",
        "ToR":     "Terms of Reference",
        "BUL":     "Business Unit Leader",
        "EQCR":    "Engagement Quality Control Review",
        "LD":      "Liquidated Damages",
        "DPR":     "Detailed Project Report",
        "INR":     "Indian National Rupee",
        "Cr":      "Crore (10 million INR)",
        "FPO":     "Farmer Producer Organisation",
        "ABPU":    "Agri Business Promoting Unit",
        "PAN":     "Permanent Account Number (Indian tax ID)",
        "GST":     "Goods and Services Tax",
        "LLP":     "Limited Liability Partnership",
        "SPV":     "Special Purpose Vehicle",
        "JV":      "Joint Venture",
        "CV":      "Curriculum Vitae",
        "TL":      "Team Leader",
        "GIS":     "Geographic Information System",
        "ICT":     "Information and Communications Technology",
        "PPP":     "Public Private Partnership",
        "MIS":     "Management Information System",
        "SOP":     "Standard Operating Procedure",
        "KPI":     "Key Performance Indicator",
        "NPA":     "Non-Performing Asset",
        "CA":      "Chartered Accountant",
        "MBA":     "Master of Business Administration",
        "BTech":   "Bachelor of Technology",
        "MTech":   "Master of Technology",
        "PhD":     "Doctor of Philosophy",
        "ICAR":    "Indian Council of Agricultural Research",
        "NABARD":  "National Bank for Agriculture and Rural Development",
    },

    # Legal / contractual term definitions
    "legal_terms": {
        "limitation_of_liability": (
            "A clause capping the total financial exposure of a party. "
            "GTBL policy: cap must not exceed contract value. "
            "Risk: HIGH if uncapped or >contract value."
        ),
        "liquidated_damages": (
            "Pre-agreed penalty for delays or non-performance. "
            "GTBL thresholds: ≤10% → ACCEPTABLE, 10-20% → MEDIUM, ≥20% → HIGH."
        ),
        "termination_for_convenience": (
            "Client right to end contract without cause. "
            "Risk: HIGH if GTBL has no symmetric right. "
            "GTBL position: must have right to terminate for non-payment."
        ),
        "force_majeure": (
            "Excuses non-performance due to extraordinary events beyond party control."
        ),
        "indemnification": (
            "Obligation to compensate the other party for specified losses. "
            "Broad indemnities effectively create uncapped liability."
        ),
        "no_deviation_clause": (
            "Bidder must accept all RFP terms unconditionally. "
            "Risk: HIGH when combined with blacklisting/history declarations."
        ),
        "blacklisting_declaration": (
            "Requirement that bidder certifies it has not been debarred/blacklisted. "
            "GTBL was debarred Oct 2021–Sep 2024. "
            "Historical language ('has not been') conflicts with GTBL position."
        ),
        "co_insured": (
            "Client named as additional insured on contractor's insurance policy. "
            "Risk: HIGH — creates potential for large claims against GTBL."
        ),
        "milestone_payment": (
            "Payment tied to completion of defined deliverables. "
            "Preferred over deployment-based or time-based."
        ),
        "deemed_approval": (
            "Client approval assumed if no response within specified days. "
            "Protects GTBL from indefinite approval delays."
        ),
    },

    # Risk implications
    "risk_implications": {
        "uncapped_liability":    ("HIGH",      "Unlimited financial exposure for GTBL"),
        "unilateral_termination":("HIGH",      "Only client can terminate — GTBL locked in"),
        "ld_over_20pct":        ("HIGH",      "LD cap ≥20% of contract value"),
        "blacklisting_conflict": ("HIGH",      "Declaration conflicts with GTBL history"),
        "co_insured_client":    ("HIGH",      "Client as co-insured creates claim risk"),
        "no_invoice_cycle":     ("MEDIUM",    "Payment timing undefined"),
        "no_approval_timeline":  ("MEDIUM",    "Deliverable acceptance window undefined"),
        "ld_10_to_20pct":       ("MEDIUM",    "LD cap between 10% and 20%"),
        "replacement_lt_30days": ("MEDIUM",    "Personnel replacement period ≤30 days"),
        "liability_at_contract":  ("MEDIUM",   "Liability capped exactly at contract value"),
        "liability_below_contract":("ACCEPTABLE","Liability well-capped"),
        "ld_below_10pct":       ("ACCEPTABLE","LD within acceptable range"),
        "bilateral_termination": ("ACCEPTABLE","Symmetric rights for both parties"),
    },

    # Sector keywords → offerings mapping
    "sector_mappings": {
        "horticulture":  "AGRI & ALLIED",
        "food_park":     "AGRI & ALLIED",
        "agriculture":   "AGRI & ALLIED",
        "urban":         "URBAN INFRA",
        "smart_city":    "URBAN INFRA",
        "amrut":         "URBAN INFRA",
        "roads":         "TRANSPORT, LOGISTICS & INDUSTRIAL INFRA",
        "highway":       "TRANSPORT, LOGISTICS & INDUSTRIAL INFRA",
        "power":         "ENERGY & RENEWABLES",
        "solar":         "ENERGY & RENEWABLES",
        "health":        "HEALTH & HUMAN SERVICES",
        "education":     "EDUCATION",
        "digital":       "DIGIGOV",
        "e_governance":  "DIGIGOV",
        "msme":          "MSMES",
        "skill":         "RURAL DEVELOPMENT & SUSTAINABLE LIVELIHOODS",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KnowledgeGraph:
    """
    In-memory knowledge graph for a set of document chunks.
    Built once after parsing; queried during LLM prompt enrichment.
    """
    graph:     object = field(default=None)   # nx.DiGraph
    doc_name:  str    = ""
    entities:  Dict   = field(default_factory=dict)
    terms:     Dict   = field(default_factory=dict)

    def __post_init__(self):
        if HAS_NX:
            self.graph = nx.DiGraph()
        self._bootstrap_ontology()

    def _bootstrap_ontology(self):
        """Load built-in domain ontology into graph."""
        if not HAS_NX or self.graph is None:
            return

        G = self.graph

        # Add abbreviation nodes
        for abbr, full in DOMAIN_ONTOLOGY["abbreviations"].items():
            G.add_node(abbr, type="abbreviation", label=abbr)
            G.add_node(full, type="entity",       label=full)
            G.add_edge(abbr, full, relation="SYNONYM_OF")
            self.terms[abbr.lower()] = full

        # Add legal term nodes
        for term, definition in DOMAIN_ONTOLOGY["legal_terms"].items():
            node_id = f"TERM:{term}"
            G.add_node(node_id, type="legal_term", definition=definition, label=term)

        # Add risk implication edges
        for condition, (risk_level, description) in DOMAIN_ONTOLOGY["risk_implications"].items():
            G.add_node(f"RISK:{condition}", type="risk", level=risk_level,
                       description=description, label=condition)

    def build_from_chunks(self, chunks: list) -> "KnowledgeGraph":
        """
        Scan chunks and extract entities, relationships, and build graph.
        Returns self for chaining.
        """
        if not HAS_NX:
            return self

        for chunk in chunks:
            self._extract_from_chunk(chunk)

        return self

    def _extract_from_chunk(self, chunk):
        """Extract entities and relationships from a single chunk."""
        if not HAS_NX:
            return

        text = getattr(chunk, "text", "") or ""
        page = getattr(chunk, "page_no", 0)
        G    = self.graph

        # Detect known abbreviations in text
        for abbr in DOMAIN_ONTOLOGY["abbreviations"]:
            pattern = r"\b" + re.escape(abbr) + r"\b"
            if re.search(pattern, text):
                node_id = f"MENTION:{abbr}:p{page}"
                G.add_node(node_id, type="mention", abbr=abbr, page=page)
                G.add_edge(abbr, node_id, relation="MENTIONED_AT")

        # Extract organization names (rough heuristic: Title Case multi-word)
        orgs = re.findall(
            r"(?:Ministry|Department|Board|Authority|Corporation|Agency|"
            r"Limited|Ltd|LLP|Pvt|Government|Govt)\s+of\s+[A-Z][a-zA-Z\s,]+",
            text,
        )
        for org in orgs[:10]:
            org = org.strip()[:100]
            if org not in G:
                G.add_node(org, type="organisation", label=org)
            if self.doc_name:
                G.add_edge(self.doc_name, org, relation="INVOLVES")

        # Extract monetary values
        amounts = re.findall(
            r"(?:INR|Rs\.?|₹)\s*([\d,.]+)\s*(Cr(?:ore)?|Lakh|Million)?",
            text, re.I,
        )
        for amt_raw, unit in amounts[:5]:
            try:
                val  = float(amt_raw.replace(",", ""))
                unit = (unit or "").strip().lower()
                if unit.startswith("cr"):
                    val_cr = val
                elif unit.startswith("lakh"):
                    val_cr = val / 100
                else:
                    val_cr = val / 1e7
                self.entities.setdefault("amounts_cr", []).append(round(val_cr, 2))
            except ValueError:
                pass

        # Detect risk signals
        for condition, (level, desc) in DOMAIN_ONTOLOGY["risk_implications"].items():
            keywords = condition.replace("_", " ")
            if keywords in text.lower():
                G.add_edge(
                    f"RISK:{condition}",
                    self.doc_name or "document",
                    relation="DETECTED_IN",
                    page=page,
                )

    # ── Query helpers ─────────────────────────────────────────────────────────

    def expand_abbreviations(self, text: str) -> str:
        """Replace known abbreviations with 'ABBR (Full Form)' in text."""
        for abbr, full in DOMAIN_ONTOLOGY["abbreviations"].items():
            pattern = r"\b" + re.escape(abbr) + r"\b"
            # Only expand first occurrence
            text = re.sub(pattern, f"{abbr} ({full})", text, count=1)
        return text

    def get_context_for_clause(self, clause_type: str) -> str:
        """
        Return a concise graph-derived context string for a clause type.
        Injected into LLM prompts to improve extraction accuracy.
        """
        ctx_lines = [
            "=== KNOWLEDGE GRAPH CONTEXT ===",
            "",
        ]

        # Relevant abbreviations for this clause type
        relevant_abbrs = {
            "liability":    ["GTBL", "LLP", "EQCR", "BUL"],
            "insurance":    ["GTBL", "LLP"],
            "scope":        ["PMC", "PMU", "PMA", "DPR", "ToR", "TL", "GIS"],
            "payment":      ["INR", "Cr", "LLP", "GST"],
            "deliverables": ["ToR", "MIS", "KPI", "SOP"],
            "personnel":    ["TL", "CV", "GIS", "ICT"],
            "ld":           ["LD", "INR", "Cr"],
            "penalties":    ["LD", "INR", "Cr"],
            "termination":  ["GTBL", "BUL", "EQCR"],
            "eligibility":  ["GTBL", "LLP", "PAN", "GST", "JV", "SPV"],
        }

        abbrs = relevant_abbrs.get(clause_type, [])
        if abbrs:
            ctx_lines.append("ABBREVIATIONS IN THIS DOCUMENT:")
            for abbr in abbrs:
                full = DOMAIN_ONTOLOGY["abbreviations"].get(abbr, "")
                if full:
                    ctx_lines.append(f"  {abbr} = {full}")
            ctx_lines.append("")

        # Legal definition
        term_key = {
            "liability":    "limitation_of_liability",
            "ld":           "liquidated_damages",
            "termination":  "termination_for_convenience",
            "insurance":    "co_insured",
            "eligibility":  "blacklisting_declaration",
        }.get(clause_type)

        if term_key and term_key in DOMAIN_ONTOLOGY["legal_terms"]:
            ctx_lines.append("DEFINITION:")
            ctx_lines.append(f"  {DOMAIN_ONTOLOGY['legal_terms'][term_key]}")
            ctx_lines.append("")

        # Risk thresholds
        risk_map = {
            "liability":    ["uncapped_liability", "liability_at_contract", "liability_below_contract"],
            "ld":           ["ld_over_20pct", "ld_10_to_20pct", "ld_below_10pct"],
            "termination":  ["unilateral_termination", "bilateral_termination"],
            "insurance":    ["co_insured_client"],
            "eligibility":  ["blacklisting_conflict"],
            "payment":      ["no_invoice_cycle", "no_approval_timeline"],
            "personnel":    ["replacement_lt_30days"],
        }

        risks = risk_map.get(clause_type, [])
        if risks:
            ctx_lines.append("RISK THRESHOLDS:")
            for r in risks:
                if r in DOMAIN_ONTOLOGY["risk_implications"]:
                    level, desc = DOMAIN_ONTOLOGY["risk_implications"][r]
                    ctx_lines.append(f"  [{level}] {desc}")
            ctx_lines.append("")

        ctx_lines.append("=== END KNOWLEDGE GRAPH CONTEXT ===")
        return "\n".join(ctx_lines)

    def get_offering_hint(self, chunks: list) -> str:
        """
        Guess the likely Grant Thornton offering from document content.
        Used to pre-fill offering/solution if not provided by user.
        """
        text = " ".join(
            (getattr(c, "text", "") or "") for c in chunks[:50]
        ).lower()

        for keyword, offering in DOMAIN_ONTOLOGY["sector_mappings"].items():
            if keyword.replace("_", " ") in text or keyword in text:
                return offering

        return ""

    def to_summary(self) -> dict:
        """Return a serialisable summary of what the graph found."""
        summary = {
            "nodes":            0,
            "edges":            0,
            "organisations":    [],
            "amounts_cr":       self.entities.get("amounts_cr", []),
            "detected_risks":   [],
        }
        if not HAS_NX or self.graph is None:
            return summary

        G = self.graph
        summary["nodes"] = G.number_of_nodes()
        summary["edges"] = G.number_of_edges()

        summary["organisations"] = [
            n for n, d in G.nodes(data=True)
            if d.get("type") == "organisation"
        ][:10]

        summary["detected_risks"] = [
            {
                "condition": n.replace("RISK:", ""),
                "level":     d.get("level", ""),
                "desc":      d.get("description", ""),
            }
            for n, d in G.nodes(data=True)
            if d.get("type") == "risk" and G.out_degree(n) > 0
        ]

        return summary


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function used by pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_knowledge_graph(chunks: list, doc_name: str = "") -> KnowledgeGraph:
    """Build and return a KnowledgeGraph from parsed chunks."""
    kg = KnowledgeGraph(doc_name=doc_name)
    kg.build_from_chunks(chunks)
    return kg


def get_prompt_context(clause_type: str, kg: Optional[KnowledgeGraph] = None) -> str:
    """
    Return knowledge graph context string for injecting into LLM prompts.
    Falls back to static ontology if no graph provided.
    """
    if kg is not None:
        return kg.get_context_for_clause(clause_type)
    # Fallback: create minimal graph with just the ontology
    kg_static = KnowledgeGraph()
    return kg_static.get_context_for_clause(clause_type)
