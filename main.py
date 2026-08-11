import os
import tempfile
import traceback
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from google import genai

import careerlens_pipeline as clp

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading model...")
model, tokenizer, device = clp.load_model()
print("Model loaded!")

print("Building skill matcher (ESCO + tech supplement)...")
nlp, matcher = clp.build_skill_matcher("data/esco_skills.csv")
print("Skill matcher ready!")

gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

LABEL_PRIORITY = {"Good Fit": 0, "Potential Fit": 1, "No Fit": 2}


@app.get("/")
def root():
    return {"status": "alive"}


def _extract_text_safe(upload_file, contents):
    suffix = "." + upload_file.filename.rsplit(".", 1)[-1] if "." in upload_file.filename else ""
    if suffix.lower() not in (".pdf", ".docx", ".txt"):
        raise ValueError(f"Unsupported file type '{suffix}'. Please upload a PDF, DOCX, or TXT file.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    try:
        text = clp.extract_text_from_file(tmp_path)
    finally:
        os.remove(tmp_path)
    return text


@app.post("/predict")
async def predict(
    resume_file: UploadFile = File(...),
    jd_text: str = Form(...),
    target_role: str = Form(...),
):
    try:
        if not jd_text.strip():
            return {"error": "Job description cannot be empty."}

        contents = await resume_file.read()
        resume_text = _extract_text_safe(resume_file, contents)

        first_line = next(
            (line.strip() for line in resume_text.split("\n") if line.strip()),
            "Candidate",
        )
        candidate_name = first_line[:60]

        report = clp.generate_report(
            candidate_name=candidate_name,
            target_role=target_role,
            resume_text=resume_text,
            jd_text=jd_text,
            model=model,
            tokenizer=tokenizer,
            device=device,
            nlp=nlp,
            matcher=matcher,
            gemini_client=gemini_client,
        )
        report["fit_score"].pop("all_scores", None)
        return report
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        traceback.print_exc()
        return {"error": "Something went wrong while processing your resume.", "detail": str(e)}


@app.post("/rank-candidates")
async def rank_candidates(
    resume_files: list[UploadFile] = File(...),
    jd_text: str = Form(...),
    target_role: str = Form(...),
):
    if not jd_text.strip():
        return {"error": "Job description cannot be empty."}
    if not resume_files:
        return {"error": "Upload at least one resume."}

    results = []
    for resume_file in resume_files:
        try:
            contents = await resume_file.read()
            resume_text = _extract_text_safe(resume_file, contents)

            first_line = next(
                (line.strip() for line in resume_text.split("\n") if line.strip()),
                "Candidate",
            )
            candidate_name = first_line[:60]

            fit_result = clp.predict_fit_safe(
                resume_text, jd_text, model, tokenizer, device, nlp, matcher
            )
            fit_result.pop("all_scores", None)

            results.append({
                "candidate_name": candidate_name,
                "filename": resume_file.filename,
                "fit_score": fit_result,
            })
        except ValueError as e:
            results.append({"filename": resume_file.filename, "error": str(e)})
        except Exception as e:
            traceback.print_exc()
            results.append({"filename": resume_file.filename, "error": "Failed to process this resume."})

    ranked = sorted(
        (r for r in results if "fit_score" in r),
        key=lambda r: (
            LABEL_PRIORITY.get(r["fit_score"]["final_label"], 3),
            -r["fit_score"]["confidence"],
        ),
    )
    failed = [r for r in results if "error" in r]

    return {
        "target_role": target_role,
        "ranked_candidates": ranked,
        "failed": failed,
    }