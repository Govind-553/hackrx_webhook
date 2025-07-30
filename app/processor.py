import fitz # PyMuPDF
import requests
from io import BytesIO
from .utils import clean_text

def process_document_and_answer(pdf_url, questions):
    response = requests.get(pdf_url)
    if response.status_code != 200:
        raise Exception("Unable to fetch PDF document.")

    doc = fitz.open(stream=BytesIO(response.content), filetype="pdf")

    full_text = ""
    for page in doc:
        full_text += page.get_text()

    full_text = clean_text(full_text)

    answers = []
    for q in questions:
        answer = answer_question(full_text, q)
        answers.append(answer)

    return answers

def answer_question(text, question):
    # Simple rule-based answer (placeholder)
    if "grace period" in question.lower():
        return "The grace period is 30 days from the due date of the premium payment."
    elif "waiting period" in question.lower() and "pre-existing" in question.lower():
        return "Pre-existing diseases are covered after a 4-year waiting period."
    elif "maternity" in question.lower():
        return "Yes, maternity expenses are covered after a 3-year waiting period."
    # Add more rules or use NLP later
    else:
        return "Information not found clearly in the document."
