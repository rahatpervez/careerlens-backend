import requests

url = "http://127.0.0.1:8000/rank-candidates"

files = [
    ("resume_files", ("sqa_cv.pdf", open("Md.Rahat Pervez.pdf", "rb"), "application/pdf")),
    ("resume_files", ("ml_cv.pdf", open("Md.Rahat Pervez_ML.pdf", "rb"), "application/pdf")),
]
data = {
    "jd_text": "We are looking for a Software Quality Assurance (SQA) Engineer to join our team. The ideal candidate will have hands-on experience in manual and automated testing, writing test cases, and identifying bugs before release. Experience with tools like Selenium, Postman, JIRA is a plus. Familiarity with API testing and SQL is desired.",
    "target_role": "SQA Engineer",
}

response = requests.post(url, files=files, data=data)
print(response.status_code)

import json
result = response.json()
print(json.dumps(result, indent=2))