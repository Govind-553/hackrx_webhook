import fitz  # PyMuPDF
import requests
from io import BytesIO
from .utils import clean_text
from sentence_transformers import SentenceTransformer, util
import torch

# Load lightweight transformer model for embeddings (fast and accurate)
model = SentenceTransformer('all-MiniLM-L6-v2') 

def process_document_and_answer(pdf_url, questions):
    response = requests.get(pdf_url)
    if response.status_code != 200:
        raise Exception("Unable to fetch PDF document.")

    doc = fitz.open(stream=BytesIO(response.content), filetype="pdf")

    # Extract text by sentences for better matching
    sentences = []
    for page in doc:
        text = page.get_text()
        text = clean_text(text)
        page_sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
        sentences.extend(page_sentences)

    # Compute embeddings for all sentences in the document
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

    answers = []
    for question in questions:
        question_embedding = model.encode(question, convert_to_tensor=True)
        cosine_scores = util.cos_sim(question_embedding, sentence_embeddings)[0]
        top_idx = torch.argmax(cosine_scores).item()
        best_answer = sentences[top_idx]
        answers.append(best_answer)

    return answers
