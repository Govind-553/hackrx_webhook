import re

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove excess whitespace
    return text.strip()
