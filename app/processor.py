import fitz  # PyMuPDF
import requests
from io import BytesIO
from .utils import clean_text, chunk_text, calculate_relevance_score
import logging
from functools import lru_cache
import hashlib

from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from transformers import pipeline
import re

# Initialize models with optimized settings
model = SentenceTransformer('all-MiniLM-L6-v2')
model.max_seq_length = 512  # Optimize for speed

# Initialize a more powerful model for complex questions
backup_model = SentenceTransformer('paraphrase-mpnet-base-v2')

# Question-answering pipeline for extractive QA
qa_pipeline = pipeline("question-answering", 
                      model="distilbert-base-cased-distilled-squad",
                      tokenizer="distilbert-base-cased-distilled-squad")

# Cache for document processing
document_cache = {}
embedding_cache = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@lru_cache(maxsize=100)
def get_document_hash(pdf_url):
    """Generate hash for PDF URL to enable caching"""
    return hashlib.md5(pdf_url.encode()).hexdigest()

def extract_text_optimized(doc):
    """Optimized text extraction with better structure preservation"""
    full_text = ""
    paragraphs = []
    
    for page_num, page in enumerate(doc):
        # Extract text with layout preservation
        text = page.get_text("text")
        
        # Extract tables separately if they exist
        try:
            tables = page.find_tables()
            for table in tables:
                table_text = table.extract()
                if table_text:
                    formatted_table = "\n".join([" | ".join(row) for row in table_text])
                    text += f"\n\nTable on page {page_num + 1}:\n{formatted_table}\n"
        except:
            pass  # Skip if table extraction fails
        
        full_text += f"\n--- Page {page_num + 1} ---\n{text}"
        
        # Split into paragraphs for better semantic chunking
        page_paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 30]
        paragraphs.extend(page_paragraphs)
    
    return full_text, paragraphs

def preprocess_questions(questions):
    """Analyze and categorize questions for optimal processing"""
    processed_questions = []
    
    for question in questions:
        q_lower = question.lower().strip()
        
        # Categorize question types
        question_type = "general"
        if any(word in q_lower for word in ["what", "define", "explain", "describe"]):
            question_type = "definition"
        elif any(word in q_lower for word in ["how many", "count", "number"]):
            question_type = "numerical"
        elif any(word in q_lower for word in ["when", "date", "time"]):
            question_type = "temporal"
        elif any(word in q_lower for word in ["where", "location"]):
            question_type = "location"
        elif any(word in q_lower for word in ["why", "reason", "because"]):
            question_type = "causal"
        elif any(word in q_lower for word in ["how", "process", "steps"]):
            question_type = "procedural"
        
        processed_questions.append({
            "original": question,
            "processed": q_lower,
            "type": question_type
        })
    
    return processed_questions

def multi_level_search(question_data, text_chunks, embeddings):
    """Multi-level search combining different techniques"""
    question = question_data["original"]
    q_type = question_data["type"]
    
    # Level 1: Semantic similarity search
    question_embedding = model.encode(question, convert_to_tensor=True)
    cosine_scores = util.cos_sim(question_embedding, embeddings)[0]
    
    # Get top 5 candidates instead of just 1
    top_indices = torch.topk(cosine_scores, k=min(5, len(text_chunks))).indices.tolist()
    
    # Level 2: Keyword matching boost
    question_keywords = set(re.findall(r'\b\w+\b', question.lower()))
    
    boosted_scores = []
    for idx in top_indices:
        chunk = text_chunks[idx]
        chunk_keywords = set(re.findall(r'\b\w+\b', chunk.lower()))
        
        # Keyword overlap boost
        keyword_overlap = len(question_keywords.intersection(chunk_keywords))
        keyword_boost = keyword_overlap / max(len(question_keywords), 1)
        
        # Question type specific boost
        type_boost = 0
        if q_type == "numerical" and re.search(r'\d+', chunk):
            type_boost = 0.1
        elif q_type == "temporal" and re.search(r'\b(19|20)\d{2}\b|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', chunk.lower()):
            type_boost = 0.1
        
        final_score = cosine_scores[idx].item() + (keyword_boost * 0.2) + type_boost
        boosted_scores.append((idx, final_score, chunk))
    
    # Sort by boosted scores
    boosted_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Level 3: Extractive QA for top candidate
    best_chunk = boosted_scores[0][2]
    
    try:
        # Use extractive QA for more precise answers
        qa_result = qa_pipeline(question=question, context=best_chunk)
        
        # If confidence is high, use the extracted answer
        if qa_result['score'] > 0.3:
            return qa_result['answer']
        else:
            # Fall back to best chunk
            return best_chunk
    except:
        # If QA pipeline fails, return best chunk
        return best_chunk

def process_document_and_answer(pdf_url, questions):
    """Enhanced document processing with multi-level optimization"""
    try:
        # Check cache first
        doc_hash = get_document_hash(pdf_url)
        
        if doc_hash in document_cache:
            logger.info("Using cached document")
            text_chunks, embeddings = document_cache[doc_hash]
        else:
            logger.info("Processing new document")
            # Fetch document with timeout
            response = requests.get(pdf_url, timeout=30)
            if response.status_code != 200:
                raise Exception(f"Unable to fetch PDF document. Status code: {response.status_code}")

            # Open and process PDF
            doc = fitz.open(stream=BytesIO(response.content), filetype="pdf")
            
            # Extract text with optimization
            full_text, paragraphs = extract_text_optimized(doc)
            doc.close()
            
            if not full_text.strip():
                raise Exception("The document has no readable text content.")
            
            # Create optimized text chunks
            text_chunks = []
            
            # Use paragraphs as primary chunks
            for para in paragraphs:
                cleaned_para = clean_text(para)
                if len(cleaned_para) > 50:  # Minimum chunk size
                    text_chunks.append(cleaned_para)
            
            # Add sentence-level chunks for better granularity
            sentences = chunk_text(full_text, chunk_size=200, overlap=50)
            text_chunks.extend(sentences)
            
            # Remove duplicates and very short chunks
            text_chunks = list(set([chunk for chunk in text_chunks if len(chunk.strip()) > 30]))
            
            if not text_chunks:
                raise Exception("No valid text chunks could be extracted from the document.")
            
            # Compute embeddings with batch processing
            logger.info(f"Computing embeddings for {len(text_chunks)} chunks")
            embeddings = model.encode(text_chunks, convert_to_tensor=True, batch_size=32, show_progress_bar=True)
            
            # Cache the results
            document_cache[doc_hash] = (text_chunks, embeddings)
            
            # Limit cache size
            if len(document_cache) > 10:
                oldest_key = next(iter(document_cache))
                del document_cache[oldest_key]
        
        # Process questions with optimization
        processed_questions = preprocess_questions(questions)
        
        answers = []
        for question_data in processed_questions:
            logger.info(f"Processing question: {question_data['original']}")
            
            # Multi-level search for best answer
            answer = multi_level_search(question_data, text_chunks, embeddings)
            
            # Post-process answer
            if len(answer) > 500:
                # Truncate very long answers intelligently
                sentences = answer.split('.')
                truncated = '.'.join(sentences[:3])
                if len(truncated) < 200 and len(sentences) > 3:
                    truncated = '.'.join(sentences[:5])
                answer = truncated + '.' if not truncated.endswith('.') else truncated
            
            answers.append(answer.strip())
        
        logger.info(f"Successfully processed {len(questions)} questions")
        return answers
        
    except requests.RequestException as e:
        logger.error(f"Network error: {str(e)}")
        raise Exception(f"Network error while fetching document: {str(e)}")
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise Exception(f"Document processing error: {str(e)}")

def health_check():
    """Health check function for monitoring"""
    try:
        # Test model loading
        test_embedding = model.encode("test", convert_to_tensor=True)
        return True
    except:
        return False