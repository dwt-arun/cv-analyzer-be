import os, json
from copy import deepcopy
import logging
import numpy as np
import pandas as pd
from glob import glob
from rapidfuzz import fuzz
from sentence_transformers import util
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docx import Document

from extract import load_pdf_content
from schema import FinalStructuredOutput, recompute_total_score
from schema_extractor import extract_framework_schema
from llm import embedding_model, openai_client
from cv_lang_detect import process_cv_text


# ---------------------------------------------
# LOGGING CONFIGURATION
# ---------------------------------------------
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more verbosity
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# ---------------------------------------------
# CONFIG
# ---------------------------------------------
CV_FOLDER_PATH = "docs/CIO_cvs/"
PROMPT_FILE = "docs/prompt3/persona_CIO.pdf"
JD_FILE = "docs/jd/CIO_JD.pdf"
OUTPUT_STRUCTURE_FILE = "docs/scoring_framework_IT.pdf"
OUTPUT_DIR = "output/"
# LLM CONFIG
LLM_MODEL = "gpt-4o"
TEMPERATURE = 0

# MATCHING THRESHOLDS (MADE MORE FORGIVING)
SEMANTIC_THRESHOLD = 0.70      # was 0.85
FUZZY_THRESHOLD = 65           # was 80

# HOW MUCH CV CONTEXT TO FEED (MORE CONTEXT)
TOP_K_CV_CHUNKS = 60           # was 8


# ---------------------------------------------
# LOAD DOCUMENTS
# ---------------------------------------------
if PROMPT_FILE:
    prompt_instructions = load_pdf_content(PROMPT_FILE)
job_description = load_pdf_content(JD_FILE)
scoring_framework = load_pdf_content(OUTPUT_STRUCTURE_FILE)


# You can either glob or manually specify files
cv_files = ["docs/CIO_cvs/Lorenzo.pdf"]  #glob(os.path.join(CV_FOLDER_PATH, "*.pdf")) or: 
if not cv_files:
    raise Exception(f"No CVs found in: {CV_FOLDER_PATH}")


# ---------------------------------------------
# TEXT CHUNKING
# ---------------------------------------------

def convert_cv_json_to_text(cv_json: dict) -> str:
    """Convert structured CV JSON into a plain-text string."""
    sections = []

    if 'personalDetails' in cv_json:
        pd = cv_json['personalDetails']
        sections.append(f"Name: {pd.get('name', '')}")
        sections.append(f"Location: {pd.get('location', '')}")
        sections.append(f"Current Position: {pd.get('currentPosition', '')}")
        sections.append(f"Previous Companies: {', '.join(pd.get('previousCompanies', []))}")
        sections.append(f"Expertise: {', '.join(pd.get('expertise', []))}")
        sections.append(f"Qualifications: {', '.join(pd.get('qualifications', []))}")

    if 'summary' in cv_json:
        sections.append(f"Summary:\n{cv_json['summary']}")

    if 'experience' in cv_json:
        sections.append("Experience:")
        for exp in cv_json['experience']:
            sections.append(f"- {exp['position']} at {exp['company']} ({exp['duration']})")
            for r in exp.get('responsibilities', []):
                sections.append(f"  • {r}")

    if 'education' in cv_json:
        sections.append("Education:")
        for edu in cv_json['education']:
            sections.append(f"- {edu['degree']} in {edu['fieldOfStudy']} at {edu['institution']} ({edu['years']})")

    if 'topSkills' in cv_json:
        sections.append(f"Top Skills: {', '.join(cv_json['topSkills'])}")

    if 'languages' in cv_json:
        sections.append("Languages:")
        for lang in cv_json['languages']:
            sections.append(f"- {lang['language']}: {lang['proficiency']}")

    return "\n".join(sections)


def split_into_chunks(cv_data, is_json=False):
    """Split text (or JSON CV converted to text) into paragraph chunks."""
    if isinstance(cv_data, dict):  # JSON CV
        is_json = True
        cv_text = convert_cv_json_to_text(cv_data)
    else:
        cv_text = cv_data
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len
    )
    chunks = splitter.split_text(cv_text)
    return chunks if chunks else [cv_text.strip()]


def get_relevant_cv_snippets(job_description: str, cv_data, top_k: int = TOP_K_CV_CHUNKS) -> str:
    """
    Use semantic search to select the most relevant chunks from the CV.
    Handles both raw text and JSON CV.
    """
    if isinstance(cv_data, dict):  # JSON
        cv_text = convert_cv_json_to_text(cv_data)
    else:
        cv_text = cv_data

    cv_chunks = split_into_chunks(cv_text)
    if not cv_chunks:
        return cv_text

    jd_emb = embedding_model.encode(job_description, convert_to_tensor=True)
    cv_embs = embedding_model.encode(cv_chunks, convert_to_tensor=True)

    sims = util.cos_sim(jd_emb, cv_embs)[0]
    top_k = min(top_k, len(cv_chunks))
    top_indices = sims.topk(k=top_k).indices.tolist()
    selected_chunks = [cv_chunks[i] for i in sorted(set(top_indices))]

    return "\n\n".join(selected_chunks)


def check_hallucinations(structured_output, cv_data, jd_text, cv_chunks=None, jd_chunks=None):
    """
    Compare skill evidence against actual CV/JD text/chunks.
    Handles JSON CV input.
    """
    if isinstance(cv_data, dict):
        cv_text = convert_cv_json_to_text(cv_data)
    else:
        cv_text = cv_data

    if cv_chunks is None:
        cv_chunks = split_into_chunks(cv_text)
    if jd_chunks is None:
        jd_chunks = split_into_chunks(jd_text)

    hallucinations = 0
    total = len(structured_output.skill_matches)
    if total == 0:
        return 0.0

    for skill in structured_output.skill_matches:
        cv_ok = verify_evidence(skill.evidence_from_cv, cv_text, cv_chunks)
        if not cv_ok:
            hallucinations += 1

    return (hallucinations / total) * 100.0
def verify_evidence(claim, source_text, source_chunks):
    """Evidence is valid if any of the three tiers pass."""
    if not claim:
        return False
    if exact_match(claim, source_text):
        return True
    if fuzzy_match(claim, source_text):
        return True
    if semantic_match(claim, source_chunks):
        return True
    return False

def fix_hallucinations_manual(
    structured_output: FinalStructuredOutput,
    cv_data,
    jd_text: str,
    verify_evidence_func=verify_evidence,
    split_into_chunks_func=split_into_chunks,
) -> FinalStructuredOutput:
    """
    Remove unverifiable evidence from structured output.
    Accepts either CV string or JSON object.
    """
    if isinstance(cv_data, dict):
        cv_text = convert_cv_json_to_text(cv_data)
    else:
        cv_text = cv_data

    fixed_output = deepcopy(structured_output)
    cv_chunks = split_into_chunks_func(cv_text)
    jd_chunks = split_into_chunks_func(jd_text)

    verified_skills = []
    for sm in fixed_output.skill_matches:
        cv_ok = verify_evidence_func(sm.evidence_from_cv, cv_text, cv_chunks)
        jd_ok = verify_evidence_func(sm.evidence_from_jd, jd_text, jd_chunks)
        if cv_ok and jd_ok:
            verified_skills.append(sm)
    fixed_output.skill_matches = verified_skills

    for pillar in fixed_output.pillars:
        for dim in pillar.dimensions:
            cv_ok = verify_evidence_func(dim.evidence_from_cv, cv_text, cv_chunks)
            jd_ok = verify_evidence_func(dim.evidence_from_jd, jd_text, jd_chunks)
            if not cv_ok:
                old_score = dim.behavioral_anchor_score_1_5 or 3
                dim.behavioral_anchor_score_1_5 = max(2, old_score - 1)
                dim.comments = "Score adjusted due to unverifiable evidence."
                if not dim.evidence_from_cv.strip():
                    dim.evidence_from_cv = "No supporting evidence found"
            if not jd_ok:
                dim.evidence_from_jd = "No supporting evidence found"

    fixed_output.overall_match_score_100 = recompute_total_score(fixed_output)
    return fixed_output




# ---------------------------------------------
# 3-TIER EVIDENCE MATCHING
# ---------------------------------------------
def exact_match(claim, source):
    if not claim:
        return False
    return claim.lower().strip() in source.lower()


def fuzzy_match(claim, source):
    if not claim:
        return False
    score = fuzz.token_set_ratio(claim.lower(), source.lower())
    return score >= FUZZY_THRESHOLD


def semantic_match(claim, chunks):
    if not claim:
        return False
    claim_emb = embedding_model.encode(claim, convert_to_tensor=True)
    chunk_embs = embedding_model.encode(chunks, convert_to_tensor=True)
    similarity = util.cos_sim(claim_emb, chunk_embs)[0].max().item()
    return similarity >= SEMANTIC_THRESHOLD


def verify_evidence(claim, source_text, source_chunks):
    """Evidence is valid if any of the three tiers pass."""
    if not claim:
        return False
    if exact_match(claim, source_text):
        return True
    if fuzzy_match(claim, source_text):
        return True
    if semantic_match(claim, source_chunks):
        return True
    return False




def fix_hallucinations(structured_output, cv_text, jd_text):
    """
    Ask the LLM to clean hallucinated evidence.
    We run this at most ONCE per CV to avoid over-correction to 0.
    """
    correction_prompt = f"""
You previously produced this JSON evaluation:

{structured_output.model_dump_json(indent=2)}

We have checked your evidence against the original texts and found that some
evidence strings are NOT present in the CV or JD.

ORIGINAL JOB DESCRIPTION:
{jd_text}

ORIGINAL CANDIDATE CV:
{cv_text}

TASK:
1. Remove or correct any skill_matches whose evidence_from_cv or evidence_from_jd
   cannot be found in the original texts.
2. If you remove a skill_match, adjust the overall_match_score_100 and breakdowns.
3. Return a corrected FinalStructuredOutput JSON. Do NOT add new skills.
"""

    json_schema = FinalStructuredOutput.model_json_schema()


    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": correction_prompt}],
        temperature=TEMPERATURE,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "FinalStructuredOutput",
                "schema": json_schema,
            },
        },
    )

    raw = response.choices[0].message.content
    if not raw or not raw.strip():
        raise RuntimeError(
            "Model returned empty response in fix_hallucinations.\n"
            f"Full response object:\n{response}"
        )

    try:
        corrected_data = json.loads(raw)
    except Exception:
        logger.debug("===== DEBUG — MODEL RETURNED NON-JSON =====")
        logger.debug(raw)
        logger.debug("==========================================")

        raise

    return FinalStructuredOutput.model_validate(corrected_data)


# ---------------------------------------------
# PROMPT TEMPLATE
# ---------------------------------------------

    
prompt_template = """
### 🔒 STRICT SCORING & EVIDENCE RULES

1. All evidence MUST be **copy-pasted verbatim** from the candidate CV (for `evidence_from_cv`) and JD/framework (for context).
2. You MAY use interpretation to score (5/4/3/2), but **evidence must remain verbatim** — no paraphrasing or summaries.
3. Score each dimension based on how well the CV addresses the requirement in the JD/framework — but only use evidence explicitly found in the CV.
4. If no relevant evidence is found in the CV for a dimension, set:
   - `behavioral_anchor_score_1_5 = 1`
   - `evidence_from_cv = "No supporting evidence found"`
5. DO NOT assume or invent missing content. DO NOT give credit unless **explicitly stated** in the CV.

6. Include a `skill_matches[]` list (5–15 items):
   - Each item must represent a key skill or competency clearly stated in both the JD and the CV
   - Include a verbatim quote from the CV and a verbatim quote from the JD for each match
   - Do NOT infer or assume — only use what is explicitly written
7. For behavioral dimensions (e.g., empathy, resilience, collaboration), you may infer the trait from clear behavioral actions (e.g., “mentored team”, “navigated layoffs”, “retained talent”) even if the exact word is not used.

---

### 🧠 HOW TO USE THE INPUTS:

- **Base your scoring** on the SCORING FRAMEWORK and JOB DESCRIPTION
- **Only use copy-paste CV json** for all `evidence_from_cv` fields
- **Follow all rules** in the PROMPT INSTRUCTIONS below (applies to tone, scoring logic, and format)

---

### 🧾 PROMPT INSTRUCTIONS:
{prompt_instructions}

---

### 📄 JOB DESCRIPTION:
{job_description}

---

### 📊 SCORING FRAMEWORK:
{scoring_framework}

---

### 👤 CANDIDATE CV (JSON STRUCTURE):
The candidate CV is structured as JSON. Use only fields and values explicitly present in this data. Do not infer or fabricate content.

{candidate_cv}

---

### 🎯 TASK:

RETURN a fully populated `FinalStructuredOutput` JSON — **no explanation or extra text**.
"""



if not PROMPT_FILE:
    prompt_template = """
### EVALUATION INSTRUCTIONS

You are an expert hiring evaluator assessing a candidate's CV against a job description using the defined scoring framework, but don’t invent rules, don’t free-interpret the JD, and stick to the framework

Your task:
- Score each dimension in the scoring framework.
- Extract copy-paste evidence from the candidate CV and JD.
- Justify the score based on the evidence from CV.

### RULES FOR SCORING

1. **Evidence must be VERBATIM text** from the CV (no paraphrasing or assumptions).
2. If no clear evidence is present in the CV for a dimension, assign:
   - `behavioral_anchor_score_1_5 = 1`
   - `evidence_from_cv = "No supporting evidence found"`
3. **Scoring is based only on the CV**. The JD/framework provides context but not scoring weight.
4. Each dimension score must be justified in the `comments` field.
5. Include 5–15 skill_matches[] entries. Each must:
    "Name a key skill or competency"
    "Contain verbatim evidence from both the CV and the JD"
    "Show clear alignment (present or strongly implied in both)"
    "No assumptions — only use text from the original documents"

### SCORING SCALE (behavioral_anchor_score_1_5)

- **5 = Outstanding**: Clear, repeated, high-impact evidence in CV
- **4 = Strong**: Solid evidence, directly relevant
- **3 = Acceptable**: Some relevance, but limited scale/scope
- **2 = Weak**: Indirect or partial alignment
- **1 = Not Present**: No supporting evidence in CV

Derived Fields:
- `score_10_scale = behavioral_anchor_score_1_5 * 2`
- `weighted_score_out_of_100 = score_10_scale * (weight_percent / 100.0) * 10`

### FINAL OUTPUT FORMAT

You must return a complete `FinalStructuredOutput` JSON object that includes:
- pillars[] with their dimensions[]
- computed scores and weights
- total_pillar_score_out_of_100
- overall_match_score_100
- job_fit_category:
  - High Fit    => score ≥ 75 and no 1s in critical dimensions
  - Medium Fit  => 55–74 or strong potential
  - Low Fit     => 35–54 or multiple weak areas
  - No Fit      => <35 or many critical dimensions score 1
- justification_summary: include 5+ scored dimensions with rationales

### INPUTS
### 👤 CANDIDATE CV (JSON STRUCTURE):
The candidate CV is structured as JSON. Use only fields and values explicitly present in this data. Do not infer or fabricate content.

{candidate_cv}


#### JOB DESCRIPTION:
{job_description}

#### SCORING FRAMEWORK:
{scoring_framework}

#### CANDIDATE CV:
{candidate_cv}


### TASK

Return ONLY a valid `FinalStructuredOutput` JSON. No additional text.
"""



# ---------------------------------------------
# OpenAI LLM wrapper
# ---------------------------------------------
def call_llm_structured(input_data: dict) -> FinalStructuredOutput:
    """
    Calls OpenAI with JSON schema enforcement.
    Converts model output to FinalStructuredOutput.
    """
    user_prompt = prompt_template.format(**input_data)
    schema = FinalStructuredOutput.model_json_schema()

    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=TEMPERATURE,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "FinalStructuredOutput",
                "schema": schema
            }
        }
    )

    raw = response.choices[0].message.content
    if not raw or not raw.strip():
        raise RuntimeError(
            "⚠ Model returned empty response.\n"
            f"Raw object:\n{response}"
        )

    try:
        data = json.loads(raw)
    except Exception:
        logger.debug("===== DEBUG — MODEL RETURNED NON-JSON =====")
        logger.debug(raw)
        logger.debug("==========================================")

        raise
    for pillar in data['pillars']:
        for dim in pillar['dimensions']:
            if 'evidence_from_jd' not in dim:
                dim['evidence_from_jd'] = ""
            if 'comments' not in dim:
                dim['comments'] = ""
    return FinalStructuredOutput.model_validate(data)


# ---------------------------------------------
# EMBEDDING-BASED FALLBACK SCORE
# ---------------------------------------------
def compute_embedding_similarity_score(cv_text: str, jd_text: str) -> float:
    """
    Compute a coarse similarity score (0-100) from embeddings.
    Used as a fallback when LLM gives 0 or nonsense.
    """
    cv_emb = embedding_model.encode(cv_text, convert_to_tensor=True)
    jd_emb = embedding_model.encode(jd_text, convert_to_tensor=True)

    sim = util.cos_sim(cv_emb, jd_emb).item()  # -1..1 normally, ~0..1 here

    # Map similarity ~[0.3..0.9] to [0..100]
    # Clamp for safety
    sim_clamped = max(0.0, min(1.0, sim))
    base = 0.30
    span = 0.60  # 0.30 -> 0, 0.90 -> 100
    normalized = (sim_clamped - base) / span
    normalized = max(0.0, min(1.0, normalized))

    return round(normalized * 100.0, 2)

# ---------------------------------------------
# RESULT AGGREGATION AND EXPORT
# ---------------------------------------------

def get_next_batch_filename(folder_path, base_name="batch", extension=".csv"):
    existing_files = [
        f for f in os.listdir(folder_path)
        if f.startswith(base_name) and f.endswith(extension)
    ]

    # Extract batch numbers
    batch_numbers = []
    for f in existing_files:
        try:
            num = int(f.replace(base_name, "").replace(extension, "").strip("_"))
            batch_numbers.append(num)
        except ValueError:
            pass

    next_batch = max(batch_numbers, default=0) + 1
    filename = f"{base_name}{next_batch}{extension}"
    return os.path.join(folder_path, filename)


def export_candidate_report(structured, candidate_id, candidate_name, output_path):
    doc = Document()
    doc.add_heading(f'Pre-Assessment Report - {candidate_name}', 0)
    doc.add_paragraph(f'Candidate ID: {candidate_id}\n')
    doc.add_paragraph(f'Overall Score: {structured.get("overall_match_score_100", "NA")}')
    doc.add_paragraph(f'Fit Category: {structured.get("job_fit_category", "NA")}\n')

    doc.add_heading('COMPETENCY EVALUATION', level=1)

    for pillar in structured['pillars']:
        doc.add_heading(f"{pillar['pillar_name']}", level=2)
        table = doc.add_table(rows=1, cols=5)
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = 'Dimension'
        hdr_cells[1].text = 'Weight'
        hdr_cells[2].text = 'Score'
        hdr_cells[3].text = 'Score (10 pt)'
        hdr_cells[4].text = 'Weighted'
        hdr_cells[5].text = 'Evidence'
        hdr_cells[6].text = 'Rationale'

        for dim in pillar['dimensions']:
            row_cells = table.add_row().cells
            row_cells[0].text = dim['dimension_name']
            row_cells[1].text = str(dim['weight_percent'])
            row_cells[2].text = str(dim.get('behavioral_anchor_score_1_5', 'NA'))
            row_cells[3].text = dim.get('score_pt', 'NA')
            row_cells[4].text = dim.get('weighted', 'NA')
            row_cells[5].text = dim.get('evidence_from_cv', 'NA')
            row_cells[6].text = dim.get('comments', 'NA')
            # row_cells[4].text = dim.get('comments', '')
        doc.add_paragraph(f"{pillar['pillar_name']} subtotal: {pillar.get('total_pillar_score_out_of_100', 'NA')}")

    doc.add_paragraph()
    doc.add_heading('Final Assessment & Recommendation', level=1)
    doc.add_paragraph(f"Justification: {structured.get('justification_summary', '')}")

    # Optionally add skills, strengths, areas for development, etc.

    doc.save(output_path)
    logger.info(f"Saved DOCX for {candidate_name}: {output_path}")



def get_candidate_id(index):
    return f"CAND{str(index+1).zfill(3)}"

def get_short_name_from_filename(filename):
    # Remove path, extension, split by space/underscore/dash, take first two words
    base = os.path.splitext(os.path.basename(filename))[0]
    # Replace underscores/dashes with spaces, then split
    parts = base.replace("_", " ").replace("-", " ").split()
    return "_".join(parts[:2])
# ---------------------------------------------
# MAIN CV PROCESSING LOOP
# ---------------------------------------------
results = []

# for cv_path in cv_files:
    # logger.info(f"Processing CV: {cv_path}")
    # candidate_cv_full = load_pdf_content(cv_path)
candidate_cv_full = {
  "contact": {
    "blog": "www.eventmanager-online.com",
    "linkedin": "www.linkedin.com/in/fabian-siebler-4737832b"
  },
  "summary": "Technology and business executive with extensive experience in IT leadership, digital transformation, software development, and e-commerce operations. Proven track record as CIO, IT Director, and Founder, combining strategic IT management with hands-on product and software engineering expertise. Strong background in building and scaling digital businesses, leading IT organizations, and driving operational efficiencies across industries such as healthcare, logistics, and online retail.",
  "activity": [
    {
      "date": "07/05/2025",
      "action": "Dierk Thomas added candidate to 2nd Probat Univers"
    },
    {
      "date": "07/05/2025",
      "action": "Viewed by Dierk Thomas"
    }
  ],
  "education": [],
  "languages": [
    {
      "language": "Deutsch",
      "proficiency": "Native or Bilingual"
    },
    {
      "language": "Englisch",
      "proficiency": "Professional Working"
    }
  ],
  "topSkills": [
    "Projektmanagement",
    "IT-Strategie",
    "Management"
  ],
  "experience": [
    {
      "company": "EMO EventManager Online GmbH",
      "duration": "August 2017 - Present",
      "location": "Hamburg und Umgebung, Deutschland",
      "position": "Geschäftsführer",
      "responsibilities": []
    },
    {
      "company": "DaVita Deutschland AG",
      "duration": "February 2022 - Present",
      "location": "Hamburg, Deutschland",
      "position": "Chief Information Officer",
      "responsibilities": []
    },
    {
      "company": "DaVita Deutschland AG",
      "duration": "January 2020 - January 2022",
      "location": "Hamburg und Umgebung, Deutschland",
      "position": "Director IT",
      "responsibilities": []
    },
    {
      "company": "Jungheinrich PROFISHOP AG & Co. KG",
      "duration": "April 2016 - December 2019",
      "location": "",
      "position": "Leiter IT & Prozesse",
      "responsibilities": []
    },
    {
      "company": "Jungheinrich PROFISHOP AG & Co. KG",
      "duration": "March 2014 - April 2016",
      "location": "Hamburg",
      "position": "Technischer IT-Projektmanager E-Commerce",
      "responsibilities": []
    },
    {
      "company": "thinkfactories GmbH",
      "duration": "September 2009 - December 2017",
      "location": "",
      "position": "Inhaber",
      "responsibilities": []
    },
    {
      "company": "eventmanager-online.com",
      "duration": "May 2009 - December 2017",
      "location": "Hamburg",
      "position": "Gründer & Geschäftsführer",
      "responsibilities": []
    },
    {
      "company": "Gruner + Jahr",
      "duration": "May 2011 - March 2014",
      "location": "",
      "position": "Softwareentwicklung / IT",
      "responsibilities": []
    },
    {
      "company": "rexx systems GmbH",
      "duration": "December 2006 - May 2011",
      "location": "",
      "position": "Lead Developer",
      "responsibilities": []
    },
    {
      "company": "TimberTec AG",
      "duration": "November 2002 - December 2006",
      "location": "",
      "position": "Software-Entwickler",
      "responsibilities": []
    }
  ],
  "honorsAwards": [],
  "publications": [],
  "personalDetails": {
    "name": "Fabian Siebler",
    "location": "Hamburg und Umgebung",
    "expertise": [
      "IT Strategy",
      "Project Management",
      "Digital Transformation",
      "Software Development",
      "E-Commerce Systems",
      "Business Consulting",
      "Leadership & Team Management",
      "IT Operations"
    ],
    "qualifications": [
      "Founder & Managing Director Experience",
      "CIO / IT Director Leadership",
      "Technical Project Management Background"
    ],
    "currentPosition": "Geschäftsführer / CIO",
    "previousCompanies": [
      "DaVita Deutschland AG",
      "Jungheinrich PROFISHOP AG & Co. KG",
      "thinkfactories GmbH",
      "Gruner + Jahr",
      "rexx systems GmbH",
      "TimberTec AG"
    ]
  }
}



    # Use top-K relevant chunks but much more generous
candidate_cv_context = get_relevant_cv_snippets(
        job_description, candidate_cv_full, top_k=TOP_K_CV_CHUNKS
    )

input_data = {
        "prompt_instructions": prompt_instructions,
        "job_description": job_description,
        "scoring_framework": scoring_framework,
        "candidate_cv": candidate_cv_full,
    }

try:
        structured_output = call_llm_structured(input_data)

        cv_chunks = split_into_chunks(candidate_cv_full)
        jd_chunks = split_into_chunks(job_description)

        hallucination_rate = check_hallucinations(
            structured_output,
            candidate_cv_full,
            job_description,
            cv_chunks,
            jd_chunks,
        )

        # Run AT MOST ONE correction pass if hallucinations are high
        while hallucination_rate > 10.0:
            logger.info(f"High hallucination detected ({hallucination_rate:.2f}%), running correction pass...")
            structured_output = fix_hallucinations_manual(
                structured_output, candidate_cv_full, job_description
            )
            hallucination_rate = check_hallucinations(
                structured_output,
                candidate_cv_full,
                job_description,
                cv_chunks,
                jd_chunks,
            )

        # LLM raw score
        fixed_total = recompute_total_score(structured_output)
        structured_output.overall_match_score_100 = fixed_total
        original_score = structured_output.overall_match_score_100
        # print(f"fixed total score: {fixed_total}, original LLM score: {original_score}")
        # Adjust for hallucinations (soft, not brutal)
        adjusted_llm_score = original_score * (1 - hallucination_rate / 100.0)

        # Fallback similarity-based score
        similarity_score = 0

        # FINAL SCORE LOGIC:
        # 1) If LLM score is 0 or negative → use similarity
        # 2) Else, take a weighted blend when LLM score is very low compared to 
        logger.info(f"[DEBUG] Original: {original_score:.2f}, Adjusted: {adjusted_llm_score:.2f}, Similarity: {similarity_score:.2f}")

        if original_score <= 0:
            final_score = similarity_score
            score_source = "similarity_fallback"
        else:
            # If LLM score is way below semantic similarity, blend
            if adjusted_llm_score < 0.5 * similarity_score:
                final_score = 0.7 * adjusted_llm_score + 0.3 * similarity_score
                score_source = "blended_llm_similarity"
            else:
                final_score = adjusted_llm_score
                score_source = "llm_adjusted"

        num_skill_matches = len(structured_output.skill_matches)

        results.append({
            # "cv_file": cv_path,
            "job_fit_category": structured_output.job_fit_category,
            "original_score": original_score,
            "hallucination_rate": hallucination_rate,
            "adjusted_llm_score": adjusted_llm_score,
            "similarity_score": similarity_score,
            "final_score": final_score,
            "score_source": score_source,
            "skills_matched": structured_output.skill_matches,
            "num_skill_matches": num_skill_matches,
            "structured_output": structured_output.model_dump(),
        })

except Exception as e:
        logger.error(f"Exception processing ", exc_info=True)



# 1. Find all unique dimension names across all candidates
dimension_names = set()
for r in results:
    structured = r['structured_output']
    for pillar in structured['pillars']:
        for dim in pillar['dimensions']:
            dimension_names.add(dim['dimension_name'])
dimension_names = sorted(list(dimension_names))

# 2. Build rows (one per candidate)
dimension_fields = []
wide_rows = []
for idx, r in enumerate(results):
    structured = r['structured_output']
    candidate_id = get_candidate_id(idx)
    candidate_name = candidate_cv_full['personalDetails']['name'].replace(" ", "_")
    row = {
        "candidate_id": candidate_id,
        "candidate_name": candidate_name,
        "overall_match_score_100": structured.get('overall_match_score_100'),
        "job_fit_category": structured.get('job_fit_category'),
    }
    for pillar in structured['pillars']:
        for dim in pillar['dimensions']:
            dim_name = dim['dimension_name']
            row[f"{dim_name}_weight_percent"] = dim['weight_percent']
            row[f"{dim_name}_score"] = dim['behavioral_anchor_score_1_5']
            row[f"{dim_name}_score_pt"] = dim['score_10_scale']
            row[f"{dim_name}_weighted"] = dim['weighted_score_out_of_100']
            row[f"{dim_name}_evidence_from_CV"] = dim['evidence_from_cv']
            row[f"{dim_name}_rationale"] = dim.get('comments', "")
    # Fill missing dimensions
    # dimension_fields = []
    for dim_name in dimension_names:
        dimension_fields.extend([
            f"{dim_name}_weight_percent",
            f"{dim_name}_score",
            f"{dim_name}_score_pt",
            f"{dim_name}_weighted",
            f"{dim_name}_rationale",])
    for dim_name in dimension_fields:
        if dim_name not in row:
            row[dim_name] = "NA"
    # Add justification at the end
    row["justification_summary"] = structured.get("justification_summary", "")
    df_single = pd.DataFrame([row])
    # Create candidate output subfolder
    candidate_folder_name = f"{candidate_id}_{candidate_name}"
    candidate_output_dir = os.path.join(OUTPUT_DIR, candidate_folder_name)
    os.makedirs(candidate_output_dir, exist_ok=True)

    # Save candidate CSV
    csv_path = os.path.join(candidate_output_dir, f"{candidate_id}_{candidate_name}.csv")
    df_single.to_csv(csv_path, index=False)
    logger.info(f"Saved candidate CSV: {csv_path}")

    # Save candidate DOCX
    docx_path = os.path.join(candidate_output_dir, f"{candidate_id}_{candidate_name}.docx")
    export_candidate_report(structured, candidate_id, candidate_name, docx_path)

    wide_rows.append(row)

# 3. Column order (match your reference)
columns = ["candidate_id", "candidate_name", "overall_match_score_100", "job_fit_category"] + dimension_fields + ["justification_summary"]

# 4. Export
file_path = get_next_batch_filename(OUTPUT_DIR, base_name="cv_results_batch", extension=".csv")
df_wide = pd.DataFrame(wide_rows)[columns]
df_wide = df_wide.sort_values(by="overall_match_score_100", ascending=False)
df_wide.to_csv(file_path, index=False)  
logger.info("Saved combined CSV: combined_cv.csv")

# ---------------------------------------------
# FINAL RANKING
# ---------------------------------------------
ranked = sorted(results, key=lambda x: x["final_score"], reverse=True)

logger.info("\n========== FINAL RANKED RESULTS ==========")
for i, r in enumerate(ranked, 1):
    # logger.info(f"Rank {i}: {r['cv_file']}")
    logger.info(f"  Fit Category       : {r['job_fit_category']}")
    logger.info(f"  Original LLM Score : {r['original_score']}")
    logger.info(f"  Hallucination %    : {r['hallucination_rate']:.2f}%")
    logger.info(f"  Adj. LLM Score     : {r['adjusted_llm_score']:.2f}")
    logger.info(f"  Similarity Score   : {r['similarity_score']:.2f}")
    logger.info(f"  Final Score        : {r['final_score']:.2f}  (source={r['score_source']})")
    logger.info(f"  Skill matches      : {r['num_skill_matches']}")

    # print(f"skill_matches details:{r['skills_matched']}")
    logger.info(f"  justification_summary: {r['structured_output']['justification_summary']}")
    for pillar in r['structured_output']['pillars']:
        for dim in pillar['dimensions']:
            print(f"  {pillar['pillar_name']} - {dim['dimension_name']}:")
            print(f"    Score: {dim['behavioral_anchor_score_1_5']}")
            print(f"    CV Evidence: {dim['evidence_from_cv']}")
            print(f"    JD Evidence: {dim['evidence_from_jd']}")
            print(f"    Comments: {dim.get('comments', '')}\n")

