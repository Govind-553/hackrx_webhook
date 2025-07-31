import re
from typing import List
import nltk
import logging

# --- Pre-compile Regex for Performance ---
RE_WHITESPACE = re.compile(r'\s+')
RE_UNWANTED_CHARS = re.compile(r'[^a-zA-Z0-9\s.,;?!-:]')
RE_BROKEN_WORDS = re.compile(r'(\w+)-\s*\n\s*(\w+)') # Handle words broken across lines

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def clean_text(text: str) -> str:
    """Cleans text by removing artifacts and normalizing whitespace."""
    if not text:
        return ""
    text = RE_BROKEN_WORDS.sub(r'\1\2', text)
    text = RE_UNWANTED_CHARS.sub('', text)
    text = RE_WHITESPACE.sub(' ', text).strip()
    return text

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Chunks text into smaller, overlapping segments."""
    if not text:
        return []
    
    words = text.split()
    if not words:
        return []

    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return [c for c in chunks if len(c.strip()) > 30]