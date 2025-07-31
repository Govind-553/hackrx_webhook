import fitz
import aiohttp
from io import BytesIO
from .utils import clean_text, chunk_text
import logging
import hashlib
import asyncio
import concurrent.futures
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
import torch
from transformers import pipeline, AutoTokenizer

# --- Model & System Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
logger = logging.getLogger(__name__)

try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    
    # We need the tokenizer to correctly truncate the context
    model_name = "facebook/bart-large-cnn"
    answer_tokenizer = AutoTokenizer.from_pretrained(model_name)
    answer_generator = pipeline("summarization", model=model_name, tokenizer=answer_tokenizer, device=0 if DEVICE == 'cuda' else -1)
    
    logger.info(f"Models loaded successfully on {DEVICE.upper()}.")
except Exception as e:
    logger.error(f"Fatal error loading models: {e}")
    raise

document_cache = {}
executor = concurrent.futures.ThreadPoolExecutor()

def get_document_hash(pdf_url: str) -> str:
    return hashlib.md5(pdf_url.encode()).hexdigest()

async def run_in_executor(func, *args):
    return await asyncio.get_event_loop().run_in_executor(executor, func, *args)

def build_faiss_index(text_chunks: list[str]):
    """Builds a FAISS index from text chunks for fast similarity search."""
    if not text_chunks:
        return None
    
    embeddings = embedding_model.encode(text_chunks, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(embeddings)
    
    index = faiss.IndexIDMap(faiss.IndexFlatIP(embeddings.shape[1]))
    index.add_with_ids(embeddings, np.arange(len(text_chunks)))
    
    return index

def search_and_generate(index, text_chunks, questions):
    """Searches the index and generates answers, ensuring context fits the model's limit."""
    if not index:
        return ["Document content could not be processed for search."] * len(questions)

    question_embeddings = embedding_model.encode(questions, convert_to_numpy=True)
    faiss.normalize_L2(question_embeddings)
    
    distances, indices = index.search(question_embeddings, k=3) # Retrieve top 3 chunks
    answers = []
    
    # Define the max token length for the context based on the model's limit
    max_context_tokens = answer_generator.model.config.max_position_embeddings - 200

    for i, question in enumerate(questions):
        if distances[i][0] < 0.3: # Confidence check
            answers.append("I could not find a relevant answer in the provided document.")
            continue

        context = " ".join([text_chunks[idx] for idx in indices[i] if idx != -1])

        # --- THE CRITICAL FIX ---
        # Truncate the context to ensure it fits within the model's token limit.
        tokenized_context = answer_tokenizer.encode(context, truncation=True, max_length=max_context_tokens)
        truncated_context = answer_tokenizer.decode(tokenized_context, skip_special_tokens=True)
        # --- END OF FIX ---

        prompt = f'Based on the context: "{truncated_context}", answer the question: "{question}"'
        
        try:
            generated = answer_generator(prompt, max_length=120, min_length=10, do_sample=False)
            answers.append(generated[0]['summary_text'].strip())
        except Exception as e:
            logger.error(f"Error generating answer for question '{question}': {e}")
            answers.append("An error occurred during answer generation.")
            
    return answers

async def process_document_and_answer(pdf_url: str, questions: list[str]) -> list[str]:
    """Main workflow: downloads, processes, and answers questions from a PDF."""
    doc_hash = get_document_hash(pdf_url)

    if doc_hash in document_cache:
        logger.info(f"Using cached FAISS index for {pdf_url}")
        index, text_chunks = document_cache[doc_hash]
    else:
        logger.info(f"Fetching and processing new document: {pdf_url}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(pdf_url, timeout=45) as response:
                    response.raise_for_status()
                    doc_bytes = await response.read()
            
            with fitz.open(stream=doc_bytes, filetype="pdf") as doc:
                raw_text = " ".join(page.get_text() for page in doc)
            
            if not raw_text.strip():
                raise ValueError("The document is empty or contains no readable text.")

            cleaned_text = await run_in_executor(clean_text, raw_text)
            text_chunks = await run_in_executor(chunk_text, cleaned_text, 400, 50)
            
            index = await run_in_executor(build_faiss_index, text_chunks)
            if not index:
                raise ValueError("Failed to build a search index from the document content.")

            document_cache[doc_hash] = (index, text_chunks)
            if len(document_cache) > 10:
                del document_cache[next(iter(document_cache))]
        except Exception as e:
            logger.error(f"Failed to process document from {pdf_url}: {e}")
            raise RuntimeError(f"Document processing error: {e}") from e

    return await run_in_executor(search_and_generate, index, text_chunks, questions)