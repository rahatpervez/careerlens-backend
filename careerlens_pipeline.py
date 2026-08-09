"""
CareerLens - Core ML Pipeline
Main model: parvez30/careerlens-fit-classifier (jjzha/jobbert-base-cased, fine-tuned)
Skill matching: ESCO taxonomy (13,896 skills) + curated tech supplement
Recommendation/chat: Google Gemini API (gemini-3.5-flash-lite)

This file is the single source of truth for the fit-prediction pipeline.
Import these functions directly in the FastAPI backend.
"""

import re
import torch
import torch.nn.functional as F
import spacy
from spacy.matcher import PhraseMatcher
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from google import genai

HF_REPO = "parvez30/careerlens-fit-classifier"
GEMINI_MODEL = "gemini-3.5-flash-lite"

TECH_SUPPLEMENT = [
    "django", "flask", "fastapi", "react", "angular", "vue", "next.js", "node.js", "express",
    "spring", "spring boot", ".net", "asp.net", "laravel", "ruby on rails",
    "aws", "amazon web services", "azure", "gcp", "google cloud platform",
    "docker", "kubernetes", "terraform", "jenkins", "github actions", "gitlab ci",
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
    "mongodb", "redis", "elasticsearch", "graphql", "rest api", "grpc",
    "react native", "flutter", "swift", "kotlin",
    "python", "java", "javascript", "typescript", "sql", "php", "c++", "c#",
    "ruby", "scala", "perl", "matlab", "bash", "html", "css", "golang", "rust",
    "redux", "webpack", "sagemaker",
    # QA / SQA specific additions
    "selenium", "postman", "jira", "appium", "cypress", "playwright",
    "junit", "testng", "manual testing", "automated testing", "automation testing",
    "test cases", "test case design", "test planning", "test plan",
    "regression testing", "functional testing", "api testing", "unit testing",
    "integration testing", "black box testing", "white box testing",
    "cross-browser testing", "smoke testing", "sanity testing",
    "load testing", "performance testing", "bug tracking", "bug reporting",
    "quality assurance", "software testing", "sdlc", "stlc",
    "agile", "scrum", "kanban", "git", "github", "bitbucket",
    "mysql", "postgresql", "wordpress", "figma", "trello",
]


def load_model(device=None):
    """Load the main fit-classification model and tokenizer from Hugging Face Hub."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(HF_REPO).to(device)
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
    return model, tokenizer, device


def build_skill_matcher(esco_csv_path):
    """Build the ESCO + tech-supplement PhraseMatcher for skill extraction."""
    import pandas as pd
    esco_df = pd.read_csv(esco_csv_path)
    nlp = spacy.load("en_core_web_sm")
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    skill_terms = set(TECH_SUPPLEMENT)
    for _, row in esco_df.iterrows():
        pref = str(row.get("PREFERREDLABEL", "")).strip().lower()
        if pref and pref != "nan" and len(pref.split()) >= 2:
            skill_terms.add(pref)
        alt = row.get("ALTLABELS", "")
        if isinstance(alt, str) and alt.strip():
            for label in alt.split("\n"):
                label = label.strip().lower()
                if label and len(label.split()) >= 2:
                    skill_terms.add(label)

    skill_terms_list = list(skill_terms)
    for i in range(0, len(skill_terms_list), 2000):
        batch = skill_terms_list[i:i + 2000]
        matcher.add(f"SKILLS_{i}", [nlp.make_doc(t) for t in batch])

    return nlp, matcher


def extract_text_from_file(filepath):
    """Extract raw text from a resume file (.pdf, .docx, or .txt)."""
    import pdfplumber
    import docx

    ext = filepath.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        text_parts = []
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        raw_text = "\n".join(text_parts)
    elif ext == "docx":
        d = docx.Document(filepath)
        raw_text = "\n".join(p.text for p in d.paragraphs)
    elif ext == "txt":
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()
    else:
        raise ValueError(f"Unsupported file type: .{ext}")

    if not raw_text.strip():
        raise ValueError("No extractable text found — this may be a scanned/image-based file.")

    cleaned = re.sub(r"\n{3,}", "\n\n", raw_text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def predict_fit(resume_text, jd_text, model, tokenizer, device):
    """Raw model prediction (before the skill-overlap safety net)."""
    inputs = tokenizer(resume_text, jd_text, truncation=True, max_length=512,
                        padding=True, return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(**inputs).logits, dim=-1)[0]
    pred_id = int(probs.argmax())
    return {
        "predicted_label": model.config.id2label[pred_id],
        "confidence": round(float(probs[pred_id]) * 100, 1),
        "all_scores": {model.config.id2label[i]: round(float(probs[i]) * 100, 1) for i in range(len(probs))},
    }


def extract_skills(text, nlp, matcher):
    doc = nlp(text)
    return {doc[start:end].text.lower() for _, start, end in matcher(doc)}


def get_skill_gap(resume_text, jd_text, nlp, matcher):
    r = extract_skills(resume_text, nlp, matcher)
    j = extract_skills(jd_text, nlp, matcher)
    return {
        "matched_skills": sorted(r & j),
        "missing_skills": sorted(j - r),
        "extra_skills": sorted(r - j),
    }


def predict_fit_safe(resume_text, jd_text, model, tokenizer, device, nlp, matcher):
    """
    Main entry point for fit prediction. Combines the raw model prediction with an
    ESCO skill-overlap sanity check, since adversarial testing showed the raw model
    has weak JD-sensitivity (it can score an unrelated resume as "Good Fit").
    Downgrades: 0% overlap -> No Fit, <30% overlap -> Potential Fit.
    Use this function (not predict_fit alone) for both job-seeker fit scoring and
    recruiter-side candidate ranking.
    """
    model_result = predict_fit(resume_text, jd_text, model, tokenizer, device)
    skill_result = get_skill_gap(resume_text, jd_text, nlp, matcher)

    matched_count = len(skill_result["matched_skills"])
    missing_count = len(skill_result["missing_skills"])
    total_required = matched_count + missing_count
    overlap_ratio = matched_count / total_required if total_required > 0 else None

    final_label = model_result["predicted_label"]
    warning = None

    if overlap_ratio is not None and final_label == "Good Fit":
        if overlap_ratio == 0:
            final_label = "No Fit"
            warning = "Downgraded to No Fit: zero required skills matched"
        elif overlap_ratio < 0.3:
            final_label = "Potential Fit"
            warning = f"Downgraded to Potential Fit: {matched_count}/{total_required} skills matched ({overlap_ratio:.0%})"
    elif total_required == 0:
        warning = "No skills detected in JD — model-only result, verify manually"

    return {
        "final_label": final_label,
        "model_predicted_label": model_result["predicted_label"],
        "confidence": model_result["confidence"],
        "all_scores": model_result["all_scores"],
        "skill_overlap_ratio": overlap_ratio,
        "matched_skills": skill_result["matched_skills"],
        "missing_skills": skill_result["missing_skills"],
        "warning": warning,
    }


def generate_recommendation(candidate_name, target_role, fit_result, skill_result, gemini_client):
    """Generate a 3-paragraph AI recommendation using Gemini (strength / gap-action / interview tip)."""
    prompt = f"""You are a career coach helping a job seeker improve their application.

Candidate: {candidate_name}
Target role: {target_role}
Fit assessment: {fit_result['final_label']} ({fit_result['confidence']}% model confidence)
Matched skills: {', '.join(skill_result['matched_skills']) or 'None found'}
Missing skills: {', '.join(skill_result['missing_skills']) or 'None'}

Write a short, encouraging, and realistic recommendation for this candidate, structured as
exactly 3 short paragraphs separated by a blank line (no headings, no markdown, no bullet
symbols — just plain paragraphs):

Paragraph 1: One honest strength based on the matched skills.
Paragraph 2: The most important 1-2 missing skills to address, and a concrete way to do it.
Paragraph 3: One practical interview-prep tip specific to this role.

Keep the tone supportive but realistic — do not promise or guarantee job offers or outcomes.
Do not use markdown formatting, just plain text."""

    response = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text.strip()


def generate_report(candidate_name, target_role, resume_text, jd_text,
                     model, tokenizer, device, nlp, matcher, gemini_client):
    """Full pipeline: fit score (with safety net) -> skill gap -> AI recommendation."""
    fit_result = predict_fit_safe(resume_text, jd_text, model, tokenizer, device, nlp, matcher)
    skill_result = {
        "matched_skills": fit_result["matched_skills"],
        "missing_skills": fit_result["missing_skills"],
    }
    recommendation = generate_recommendation(candidate_name, target_role, fit_result, skill_result, gemini_client)

    return {
        "candidate_name": candidate_name,
        "target_role": target_role,
        "fit_score": fit_result,
        "skill_analysis": skill_result,
        "ai_recommendation": recommendation,
    }