import os
import tempfile
import traceback
from fastapi import FastAPI, UploadFile, File, Form
from dotenv import load_dotenv
from google import genai

import careerlens_pipeline as clp

load_dotenv()

app = FastAPI()

print("Loading model...")
model, tokenizer, device = clp.load_model()
print("Model loaded!")

print("Building skill matcher (ESCO + tech supplement)...")
nlp, matcher = clp.build_skill_matcher("data/esco_skills.csv")
print("Skill matcher ready!")

gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


@app.get("/")
def root():
    return {"status": "alive"}


@app.post("/predict")
async def predict(
    resume_file: UploadFile = File(...),
    jd_text: str = Form(...),
    target_role: str = Form(...),
):
    try:
        suffix = "." + resume_file.filename.rsplit(".", 1)[-1]
        contents = await resume_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        resume_text = clp.extract_text_from_file(tmp_path)
        os.remove(tmp_path)

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
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}