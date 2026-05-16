"""
db_migration_tq_v6.py
======================
Adds columns required by the TQ v2 evaluation system.

Run ONCE before deploying the new tq_extractor:
    python db_migration_tq_v6.py

Safe to re-run — all statements use IF NOT EXISTS.

New columns
-----------
tq_evaluations:
    global_discrepancies_json   TEXT       -- JSON array of cross-criterion flags

tq_score_items:
    discrepancies_json          TEXT       -- JSON array of per-criterion flags
    formula_hint                VARCHAR(20)-- STEP|BAND|PER_UNIT|QUAL|BINARY|LLM
    extracted_value             TEXT       -- the fact that was extracted
    raw_evidence                TEXT       -- verbatim sentence from proposal
    pages_searched_json         TEXT       -- JSON list of page numbers searched
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set in .env")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

MIGRATIONS = [
    # tq_evaluations
    "ALTER TABLE tq_evaluations ADD COLUMN IF NOT EXISTS global_discrepancies_json TEXT",

    # tq_score_items
    "ALTER TABLE tq_score_items ADD COLUMN IF NOT EXISTS discrepancies_json TEXT",
    "ALTER TABLE tq_score_items ADD COLUMN IF NOT EXISTS formula_hint VARCHAR(20) DEFAULT ''",
    "ALTER TABLE tq_score_items ADD COLUMN IF NOT EXISTS extracted_value TEXT",
    "ALTER TABLE tq_score_items ADD COLUMN IF NOT EXISTS raw_evidence TEXT",
    "ALTER TABLE tq_score_items ADD COLUMN IF NOT EXISTS pages_searched_json TEXT",
]

with engine.connect() as conn:
    for stmt in MIGRATIONS:
        try:
            conn.execute(text(stmt))
            print(f"  OK  {stmt[:80]}")
        except Exception as e:
            print(f"  SKIP ({e}): {stmt[:80]}")
    conn.commit()

print("\nMigration tq_v6 complete.")
