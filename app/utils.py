import re
import string
from typing import List, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import logging

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except:
        pass  # Fall back to basic processing if NLTK fails

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Enhanced text cleaning with multiple levels"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page headers/footers patterns
    text = re.sub(r'Page \d+.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Clean up common PDF artifacts
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\/\&\%\$\#\@\+\=\<\>\|\\\~\`]', ' ', text)
    
    # Fix broken words (common in PDFs)
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[\.]{3,}', '...', text)
    text = re.sub(r'[-]{3,}', '---', text)
    
    # Clean up spacing around punctuation
    text = re.sub(r'\s+([,.;:!?])', r'\1', text)
    text = re.sub(r'([,.;:!?])\s*([,.;:!?])', r'\1 \2', text)
    
    return text.strip()

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """Smart text chunking with semantic boundaries"""
    if not text or chunk_size <= 0:
        return []
    
    chunks = []
    
    try:
        # Try to use NLTK for better sentence tokenization
        sentences = sent_tokenize(text)
    except:
        # Fallback to regex-based sentence splitting
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
    
    current_chunk = ""
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_length = len(sentence)
        
        # If adding this sentence would exceed chunk size
        if current_length + sentence_length > chunk_size and current_chunk:
            # Add the current chunk
            chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap
            if overlap > 0 and len(current_chunk) > overlap:
                # Take last few words for overlap
                words = current_chunk.split()
                overlap_words = words[-min(overlap//5, len(words)//2):]
                current_chunk = ' '.join(overlap_words) + ' ' + sentence
                current_length = len(current_chunk)
            else:
                current_chunk = sentence
                current_length = sentence_length
        else:
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += ' ' + sentence
            else:
                current_chunk = sentence
            current_length += sentence_length
    
    # Add the last chunk if it exists
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Filter out very short chunks
    chunks = [chunk for chunk in chunks if len(chunk.strip()) > 20]
    
    return chunks

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract important keywords from text"""
    if not text:
        return []
    
    try:
        # Try to use NLTK stopwords
        stop_words = set(stopwords.words('english'))
    except:
        # Fallback stopwords
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
    
    # Clean and tokenize
    text = clean_text(text.lower())
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    
    # Filter out stopwords and short words
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Count frequency
    word_freq = {}
    for word in keywords:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_keywords[:max_keywords]]

def calculate_relevance_score(question: str, text: str) -> float:
    """Calculate relevance score between question and text"""
    if not question or not text:
        return 0.0
    
    question_keywords = set(extract_keywords(question.lower()))
    text_keywords = set(extract_keywords(text.lower()))
    
    if not question_keywords:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = len(question_keywords.intersection(text_keywords))
    union = len(question_keywords.union(text_keywords))
    
    jaccard_score = intersection / union if union > 0 else 0.0
    
    # Boost score for exact phrase matches
    question_lower = question.lower()
    text_lower = text.lower()
    
    phrase_boost = 0.0
    question_phrases = re.findall(r'\b\w+\s+\w+\b', question_lower)
    for phrase in question_phrases:
        if phrase in text_lower:
            phrase_boost += 0.1
    
    # Boost for question type words
    question_words = {'what', 'when', 'where', 'why', 'how', 'who', 'which'}
    type_boost = 0.05 if any(word in question_lower for word in question_words) else 0.0
    
    final_score = jaccard_score + min(phrase_boost, 0.3) + type_boost
    return min(final_score, 1.0)

def preprocess_question(question: str) -> str:
    """Preprocess question for better matching"""
    if not question:
        return ""
    
    # Clean the question
    question = clean_text(question)
    
    # Expand common abbreviations
    abbreviations = {
        "what's": "what is",
        "where's": "where is",
        "when's": "when is",
        "why's": "why is",
        "how's": "how is",
        "who's": "who is",
        "can't": "cannot",
        "won't": "will not",
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not"
    }
    
    question_lower = question.lower()
    for abbr, expansion in abbreviations.items():
        question_lower = question_lower.replace(abbr, expansion)
    
    return question_lower.strip()

def extract_entities(text: str) -> List[Tuple[str, str]]:
    """Extract named entities from text (basic implementation)"""
    entities = []
    
    # Dates
    date_patterns = [
        r'\b(19|20)\d{2}\b',  # Years
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+(19|20)\d{2}\b',  # Month Day, Year
        r'\b\d{1,2}[/-]\d{1,2}[/-](19|20)?\d{2}\b'  # MM/DD/YYYY or MM-DD-YY
    ]
    
    for pattern in date_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            entities.append((match.group(), 'DATE'))
    
    # Numbers/Quantities
    number_pattern = r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent|million|billion|thousand)?\b'
    matches = re.finditer(number_pattern, text, re.IGNORECASE)
    for match in matches:
        entities.append((match.group(), 'NUMBER'))
    
    # Proper nouns (basic capitalization heuristic)
    proper_noun_pattern = r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b'
    matches = re.finditer(proper_noun_pattern, text)
    for match in matches:
        word = match.group()
        if len(word) > 2 and word not in {'The', 'This', 'That', 'These', 'Those', 'When', 'Where', 'What', 'Why', 'How', 'Who'}:
            entities.append((word, 'PROPER_NOUN'))
    
    return entities

def is_question_answerable(question: str, text: str) -> bool:
    """Determine if a question can likely be answered from the text"""
    if not question or not text:
        return False
    
    question_lower = question.lower().strip()
    text_lower = text.lower()
    
    # Extract question keywords
    question_keywords = extract_keywords(question_lower)
    if not question_keywords:
        return False
    
    # Check if at least some keywords appear in text
    keyword_matches = sum(1 for keyword in question_keywords if keyword in text_lower)
    keyword_ratio = keyword_matches / len(question_keywords)
    
    # Basic threshold - at least 30% of keywords should match
    return keyword_ratio >= 0.3

def validate_input(data: dict) -> Tuple[bool, str]:
    """Validate input data structure"""
    if not isinstance(data, dict):
        return False, "Input must be a JSON object"
    
    if 'documents' not in data:
        return False, "Missing 'documents' field"
    
    if 'questions' not in data:
        return False, "Missing 'questions' field"
    
    documents = data.get('documents')
    questions = data.get('questions')
    
    if not documents:
        return False, "Documents field cannot be empty"
    
    if not questions:
        return False, "Questions field cannot be empty"
    
    if not isinstance(questions, list):
        return False, "Questions must be a list"
    
    if len(questions) == 0:
        return False, "Questions list cannot be empty"
    
    if len(questions) > 50:  # Reasonable limit
        return False, "Too many questions (maximum 50 allowed)"
    
    # Validate URL format
    url_pattern = r'^https?://'
    if not re.match(url_pattern, documents):
        return False, "Document URL must start with http:// or https://"
    
    # Validate questions
    for i, question in enumerate(questions):
        if not isinstance(question, str):
            return False, f"Question {i+1} must be a string"
        
        if not question.strip():
            return False, f"Question {i+1} cannot be empty"
        
        if len(question.strip()) < 3:
            return False, f"Question {i+1} is too short (minimum 3 characters)"
        
        if len(question.strip()) > 500:
            return False, f"Question {i+1} is too long (maximum 500 characters)"
    
    return True, "Valid input"