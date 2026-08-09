import time
from dotenv import load_dotenv
load_dotenv()
import careerlens_pipeline as clp

print("Loading model...")
t0 = time.time()
model, tokenizer, device = clp.load_model()
print(f"Model load: {time.time()-t0:.1f}s")

print("Building matcher...")
t0 = time.time()
nlp, matcher = clp.build_skill_matcher("data/esco_skills.csv")
print(f"Matcher build: {time.time()-t0:.1f}s")

resume_text = "Experienced QA engineer skilled in manual and automated testing, Selenium, Postman, JIRA, SQL, API testing."
jd_text = "Looking for an SQA engineer with experience in Selenium, Postman, JIRA, API testing, and SQL."

print("Running predict_fit_safe...")
t0 = time.time()
result = clp.predict_fit_safe(resume_text, jd_text, model, tokenizer, device, nlp, matcher)
print(f"Predict: {time.time()-t0:.1f}s")
print(result)