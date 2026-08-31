"""Text Preprocessing and Cleaning Module for Resume & Job Descriptions.

Provides robust regex-based cleaning, normalization, and token preservation
specifically tuned for technical resumes (preserving C++, C#, .NET, Node.js, etc.).
"""

import re
import unicodedata


def clean_text(text: str) -> str:
    """Cleans raw text extracted from resumes or job descriptions.
    
    Operations:
    1. Normalizes unicode characters.
    2. Removes URLs, email addresses, and phone numbers.
    3. Preserves technical tokens (+, #, ., -) while removing symbols & punctuation.
    4. Normalizes whitespace and lowercases the text.
    
    Args:
        text: Raw input text.
        
    Returns:
        Cleaned, normalized string.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
        
    if not text.strip():
        return ""

    # Normalize unicode (e.g., smart quotes, bullet points, accents)
    text = unicodedata.normalize("NFKD", text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs (http, https, www)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " ", text)

    # Remove phone numbers (10+ digits or formatted patterns like +1-234-567-8900)
    text = re.sub(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4,6}", " ", text)
    text = re.sub(r"\d{10,}", " ", text)

    # Preserve technical terms like c++, c#, .net, node.js, rest-api
    # Replace other special characters/punctuations with spaces
    text = re.sub(r"[^a-zA-Z0-9+#.\-\s]", " ", text)

    # Handle standalone dots or dashes that are not part of terms like .net
    text = re.sub(r"(?<![a-zA-Z0-9])[.\-](?![a-zA-Z0-9])", " ", text)

    # Collapse multiple whitespaces into a single space
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_keywords_list(text: str) -> list[str]:
    """Tokenizes cleaned text into unique alphanumeric and tech keywords."""
    cleaned = clean_text(text)
    tokens = re.findall(r"\b[a-zA-Z0-9+#.-]+\b", cleaned)
    return sorted(list(set(tokens)))
