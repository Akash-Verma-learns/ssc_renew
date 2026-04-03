"""
db_migration_tq_v5.py
======================
Adds the new columns required by the three-layer TQ evaluation.

Run once:
    python db_migration_tq_v5.py

Safe to re-run — all ALTER TABLE statements use IF NOT EXISTS.

New columns
-----------
tq_evaluations:
  financial_marks        INTEGER DEFAULT 0
  final_score_formula    TEXT
  qualification_json     TEXT      -- JSON: gate results + financial_bid_opens

tq_score_items:
  evaluation_layer                  VARCHAR(40) DEFAULT 'technical_document'
  requires_comparative_evaluation   BOOLEAN DEFAULT FALSE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set in .env")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

MIGRATIONS = [
    # ── tq_evaluations ────────────────────────────────────────────────────────
    "ALTER TABLE tq_evaluations ADD COLUMN IF NOT EXISTS financial_marks INTEGER DEFAULT 0",
    "ALTER TABLE tq_evaluations ADD COLUMN IF NOT EXISTS final_score_formula TEXT",
    "ALTER TABLE tq_evaluations ADD COLUMN IF NOT EXISTS qualification_json TEXT",

    # ── tq_score_items ────────────────────────────────────────────────────────
    "ALTER TABLE tq_score_items ADD COLUMN IF NOT EXISTS evaluation_layer VARCHAR(40) DEFAULT 'technical_document'",
    "ALTER TABLE tq_score_items ADD COLUMN IF NOT EXISTS requires_comparative_evaluation BOOLEAN DEFAULT FALSE",
]

with engine.connect() as conn:
    for stmt in MIGRATIONS:
        try:
            conn.execute(text(stmt))
            print(f"  OK: {stmt[:80]}")
        except Exception as e:
            print(f"  SKIP ({e}): {stmt[:80]}")
    conn.commit()

print("\nMigration complete.")
