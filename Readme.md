# 📄 HackRx 5.0 – PDF Question Answering Webhook API

This project is a webhook-based PDF Question Answering (QA) system built using Python and Flask for HackRx 5.0. It accepts a public PDF document link along with a list of questions and returns relevant answers by extracting and analyzing the document's content.

---

## 🚀 API Endpoint

### `POST /hackrx/run`

Accepts a public PDF URL and a list of natural language questions in JSON format.

### ✅ Request Format:

```json
{
  "documents": "<valid_pdf_url>",
  "questions": [
    "What is the eligibility criteria?",
    "Who is the target audience?"
  ]
}
```

### 🔁 Response Format:

```json
{
  "answers": [
    "The eligibility criteria is mentioned on page 2...",
    "The target audience includes students and professionals..."
  ]
}
```

---

## ⚙️ How It Works

1. Accepts a valid PDF file URL via POST request.
2. Downloads and extracts the content using PyMuPDF (`fitz`).
3. Cleans and chunks the content intelligently using regex and heuristics.
4. Matches each question with the most relevant content block.
5. Returns accurate responses using text similarity-based ranking.

---

## 🧪 Sample Test

You can test this API using tools like [Postman](https://www.postman.com/) or `curl`:

```bash
curl -X POST http://127.0.0.1:5000/hackrx/run \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "<valid_pdf_url>",
    "questions": ["What is the objective?", "What is the last date to apply?"]
  }'
```

---

## 🛠 Tech Stack

- **Language**: Python
- **Framework**: Flask
- **Libraries**: 
  - `PyMuPDF` for PDF text extraction
  - `re` for text preprocessing and splitting
  - `difflib` for matching questions with relevant content

---

## 🔧 Run Locally

```bash
git clone <your_repo_link>
cd your_repo

python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
python app.py
```

Make sure the Flask server is running at `http://127.0.0.1:5000`.

---

## 🙏 Note for Judges

- You can test the webhook using any publicly accessible PDF document.
- Ensure your PDF is not password-protected or malformed.
- Only the `/hackrx/run` endpoint is active and necessary.

