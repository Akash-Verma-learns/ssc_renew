"""
core/proposal_analyzer_prompt_patch.py
=======================================
Drop-in replacement for _BULK_EXTRACT_PROMPT in proposal_analyzer.py.

WHY THE OLD PROMPT FAILS
──────────────────────────
The old prompt asks "extract the EXACT claimed value for each criterion."
When the top-scoring pages include the RFP's evaluation table (embedded in
the proposal as a compliance matrix or appendix), the LLM faithfully reports
"20 marks" as the value for Past Experience B, because that IS the exact
text on the page — it just happens to be the scoring rule, not the bidder's
actual claim.

THE FIX IN THE PROMPT
──────────────────────
1. Explicitly forbid extracting from scoring tables / evaluation criteria rows.
2. Require the LLM to identify WHERE in the proposal the value comes from
   (company profile? experience list? financial summary?).
3. Return null if the only evidence found is from criteria/scoring text.
4. Validate extracted values against expected unit/type.

INTEGRATION
───────────
In proposal_analyzer.py, replace the _BULK_EXTRACT_PROMPT constant with
BULK_EXTRACT_PROMPT from this module:

    from core.proposal_analyzer_prompt_patch import BULK_EXTRACT_PROMPT as _BULK_EXTRACT_PROMPT
"""

BULK_EXTRACT_PROMPT = """\
You are reading a vendor's technical PROPOSAL document submitted in response to an RFP.
Your job is to extract the value that the BIDDER ACTUALLY CLAIMS for each criterion.

══════════════════════════════════════════════════════════════════════
CRITICAL — WHAT TO EXTRACT vs WHAT TO IGNORE:

✅ EXTRACT values from:
   - Company profile / About Us section
   - Financial summaries (turnover tables, balance sheet summaries)
   - Experience lists / project lists / assignment tables
   - CV sections / team profiles
   - Compliance statements by the bidder ("We have X years…", "Our turnover is…")

❌ DO NOT EXTRACT from:
   - Scoring criteria rows ("5 marks for 1 project", "20 marks maximum")
   - RFP evaluation tables or compliance matrices showing the RFP's own criteria
   - Eligibility criteria tables ("Minimum 100 Cr", "At least 3 projects")
   - Any text that looks like "X marks for Y projects" or "Max marks: N"
   - Cover pages that list RFP criteria without bidder responses

If the only mention of a criterion on a page is inside a SCORING TABLE or
ELIGIBILITY CRITERIA TABLE, return found=false and value=null for that criterion.

EXPECTED VALUE FORMATS (use these as sanity checks):
   - Turnover / Financial: a number with unit (e.g. "884.49 Cr", "Rs. 450 lakhs")
   - Project count: a plain number or phrase (e.g. "26 projects", "more than 10")
   - Years of experience: a number with "years" (e.g. "15 years", "since 2005")
   - Personnel count: a number with unit (e.g. "250 professionals", "12 experts")
   - Binary presence: "Yes" / "No" / specific capability statement

A value like "20 marks" or "5 marks" or "maximum 20" is NEVER a valid bidder claim —
it is scoring language. Return null if that is all you find.
══════════════════════════════════════════════════════════════════════

CRITERIA TO EXTRACT:
{criteria_list}

PROPOSAL TEXT (most relevant pages):
{proposal_text}

Return ONLY valid JSON — no markdown, no preamble:
{{
  "extractions": [
    {{
      "parameter":    "<exact parameter name from input>",
      "found":        true/false,
      "value":        "<actual bidder-claimed value with unit, e.g. '884.49 Cr', '26 projects', 'more than 10 years' — OR null if not found>",
      "page":         <page number where found, or null>,
      "source_type":  "<where found: 'company_profile'|'financial_summary'|'experience_list'|'cv_section'|'compliance_statement'|'not_found'>"
    }}
  ]
}}

EXTRACTION GUIDE:
- Turnover/Financial: look for "average annual turnover", "INR X crores", "Rs. X Cr" in financial sections
- Experience (years): look for "X years of experience", "since YYYY", "established in YYYY" in company profiles
- Projects (count): look for "X projects", "X assignments", numbered project lists in experience sections
- Staff/Personnel: look for "X technically qualified", "X+ personnel", "staff of X" in company profiles
- BINARY presence: look for explicit capability claims in methodology or approach sections

If a value appears ONLY in a table that also contains "marks" or "points" columns,
that is a scoring table — do NOT extract from it.
"""
