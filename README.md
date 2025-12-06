<!--
  ==========================================================
     AI-POWERED CV RANKING SYSTEM
     Evidence-Based Evaluation Engine
     Version 1.0
  ==========================================================
-->

<div align="center">
  <h1 style="margin-bottom:0.2em;">AI-POWERED CV RANKING SYSTEM</h1>
  <h3 style="margin-top:0; margin-bottom:0.2em; font-weight:normal;">Evidence-Based Evaluation Engine</h3>
  <strong>Version 1.0</strong>
</div>

<br/>

<table>
<tr><td><b>Prepared By:</b></td><td>_________________________</td></tr>
<tr><td><b>Organization:</b></td><td>_________________________</td></tr>
<tr><td><b>Date:</b></td><td>__________________________</td></tr>
</table>

---

# TABLE OF CONTENTS

1. [System Overview](#1-system-overview)
2. [Complete Workflow](#2-complete-workflow)
3. [LLM Prompt Instructions](#3-llm-prompt-pdf-ready)
4. [Scoring & Ranking Criteria](#4-scoring--ranking-criteria-pdf-ready)
5. [Three-Tier Hallucination Check](#5-three-tier-hallucination-check)
6. [Model Selection & Scaling (500+ CVs)](#6-model-selection--scaling-500-cvs)
7. [Appendix A — Pydantic Schema](#7-appendix-a--pydantic-schema)
8. [Appendix B — Example Structured Output](#appendix-b--example-structured-output)

---

# 1. SYSTEM OVERVIEW

## 1.1 Objective

This system evaluates and ranks CVs using an evidence-based, hallucination-resistant AI pipeline, ensuring:

- **Precise skill matching**
- **Verified evidence extraction**
- **Transparent scoring**
- **Fair candidate ranking**
- **Reliable performance with 500+ CVs**

## 1.2 Architecture Diagram (Text Version)

```
+-------------------------+
|      PDF DOCUMENTS      |
| JD, CVs, Prompt, Criteria|
+-------------+-----------+
              |
              v
   +----------+-----------+
   |     TEXT EXTRACTION   |
   |  (PyMuPDFLoader)      |
   +----------+------------+
              |
              v
+-----------------+------------------+
|        PREPROCESSING                |
| Chunking | Embedding | Cleaning     |
+------------------+-----------------+
              |
              v
+-------------------------------------------+
| RETRIEVAL-AUGMENTED MATCHING (MPNet)      |
| Top-k relevant chunks per JD requirement   |
+-------------------+-----------------------+
              |
              v
   +----------+-----------+
   |   LLM EVALUATION     |
   | (Groq Tool-Use LLM)  |
   | Pydantic Structured  |
   +----------+-----------+
              |
              v
+-------------------------------+
|  HALLUCINATION VALIDATION     |
|  Exact | Fuzzy | Semantic     |
+-------------------------------+
              |
              v
   +---------+----------+
   |     RANKING ENGINE |
   | Final scoring      |
   +---------+----------+
```

---

# 2. COMPLETE WORKFLOW

## Stage 1 — Data Acquisition
- Load JD PDF
- Load Prompt Instructions
- Load Ranking Criteria
- Load CVs in batch mode (20–40 recommended)

## Stage 2 — Preprocessing
- Line cleanup
- Sentence/paragraph chunking
- Embeddings computed with: `all-mpnet-base-v2`

## Stage 3 — Retrieval-Augmented Evidence Extraction
- For every JD requirement:
  - Find top-3 semantically closest CV chunks
  - Pass only the retrieved subset to the LLM
  - LLM is prevented from hallucinating due to scoped context

## Stage 4 — LLM Structured Evaluation
- **LLM used:** `llama3-groq-8b-tool-use-preview` (tool calling enabled)
- **Output:**
  - Skill matches
  - Evidence from CV
  - Evidence from JD
  - Experience alignment
  - Final score & justification

## Stage 5 — Hallucination Detection Engine
- Three layers of validation:
  - **Exact match**
  - **Fuzzy match** (≥80% token-set ratio)
  - **Semantic similarity** ≥0.90
- **Hallucination Rate Formula:**
  ```
  hallucination_rate = (# hallucinated claims / total claims) * 100
  ```

## Stage 6 — Final Scoring & Ranking
- `adjusted_score = base_score * (1 - hallucination_rate/100)`
- **Ranking:** By adjusted score, tie-breakers applied

---

# 3. LLM PROMPT (PDF-READY)

**You MUST follow these instructions:**

1. Do not infer or assume any skills.
2. All evidence must be copy-paste from provided CV text.
3. Do not paraphrase evidence.
4. If no evidence exists, write exactly: “No supporting evidence found.”
5. Only output data using the `FinalStructuredOutput` schema.
6. The output must be strictly JSON.
7. You must base your evaluation only on:
   - Job Description text
   - Candidate CV text
   - Ranking Criteria text
8. No external knowledge is allowed.

---

# 4. SCORING & RANKING CRITERIA (PDF-READY)

**Score Distribution (100 Points Total):**

| CATEGORY           | POINTS | DESCRIPTION                   |
|--------------------|--------|-------------------------------|
| Skill Match        | 40     | JD skills vs CV evidence      |
| Experience Match   | 35     | Years, roles, domain          |
| Domain Knowledge   | 15     | Tools, platforms, technologies|
| Education & Extras | 10     | Degree, certifications        |

**Fit Category Thresholds:**

| CATEGORY   | SCORE RANGE |
|------------|-------------|
| High Fit   | 80–100      |
| Medium Fit | 60–79       |
| Low Fit    | 40–59       |
| No Fit     | <40         |

---

# 5. THREE-TIER HALLUCINATION CHECK

- **Tier 1: Exact**
  - Literal substring match (zero tolerance)
- **Tier 2: Fuzzy**
  - Ratio ≥ 80 (detects reordered words & partial matches)
- **Tier 3: Semantic**
  - Cosine similarity ≥ 0.90 (embedding model: `all-mpnet-base-v2`)

---

# 6. MODEL SELECTION & SCALING (500+ CVs)

- **Recommended batch size:** 20–40 CVs per batch
- **Parallel processes:** 8–12 concurrent CV evaluations
- **Database needed?** No, unless persistent FAISS storage is required

---

# 7. APPENDIX A — PYDANTIC SCHEMA

```python
class SkillMatch(BaseModel):
    skill_name: str
    evidence_from_cv: str
    evidence_from_jd: str

class FinalStructuredOutput(BaseModel):
    job_fit_category: Literal["High Fit","Medium Fit","Low Fit","No Fit"]
    skill_matches: List[SkillMatch]
    overall_match_score_100: int
    justification_summary: str
    skill_match_score_breakdown: str
    experience_match_breakdown: str
    additional_notes: Optional[str]
```

---

# Appendix B — Example Structured Output

```json
{
  "job_fit_category": "High Fit",
  "skill_matches": [
    {
      "skill_name": "Python",
      "evidence_from_cv": "Developed multiple Python-based data pipelines at Company X.",
      "evidence_from_jd": "Proficiency in Python required."
    }
  ],
  "overall_match_score_100": 92,
  "justification_summary": "Candidate demonstrates strong alignment with all required skills and experience.",
  "skill_match_score_breakdown": "Skill Match: 38/40, Experience: 33/35, Domain: 14/15, Education: 7/10",
  "experience_match_breakdown": "5+ years in relevant roles, direct domain experience.",
  "additional_notes": "Certified in relevant technologies."
}
```

---

*End of Document*
