"""Resume Parsing and Automated Information Extraction Module.

Extracts:
1. Technical skills (with boundary-safe matching and synonym aliasing).
2. Estimated years of experience (from explicit statements and employment date ranges).
3. Highest education level (PhD, Master's, Bachelor's, etc.).
4. Estimated projects count.
5. Known industry certifications.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional
import pdfplumber

from src.preprocessing import clean_text


# Comprehensive Skill Taxonomy & Canonical Aliases
SKILL_TAXONOMY: Dict[str, List[str]] = {
    # Programming & Scripting
    "python": [r"\bpython\b", r"\bpython3\b"],
    "java": [r"\bjava\b(?!script)"],
    "c++": [r"\bc\+\+\b", r"\bcpp\b"],
    "c#": [r"\bc#\b", r"\bcsharp\b", r"\b\.net\b"],
    "javascript": [r"\bjavascript\b", r"\bjs\b", r"\bes6\b"],
    "typescript": [r"\btypescript\b", r"\bts\b"],
    "go": [r"\bgolang\b", r"\bgo\s+language\b"],
    "rust": [r"\brust\b"],
    "r": [r"\br\s+programming\b", r"\br\s+language\b", r"\b(using|in)\s+r\b"],
    "php": [r"\bphp\b"],
    "ruby": [r"\bruby\b", r"\bruby\s+on\s+rails\b"],
    "scala": [r"\bscala\b"],
    "sql": [r"\bsql\b", r"\bpostgresql\b", r"\bmysql\b", r"\bsqlite\b", r"\boracle\s+sql\b"],
    
    # Data Science & Machine Learning
    "machine learning": [r"\bmachine\s+learning\b", r"\bml\b"],
    "deep learning": [r"\bdeep\s+learning\b", r"\bdl\b"],
    "natural language processing": [r"\bnlp\b", r"\bnatural\s+language\s+processing\b"],
    "computer vision": [r"\bcomputer\s+vision\b", r"\bopencv\b"],
    "tensorflow": [r"\btensorflow\b", r"\btf\b"],
    "pytorch": [r"\bpytorch\b", r"\btorch\b"],
    "keras": [r"\bkeras\b"],
    "scikit-learn": [r"\bscikit-learn\b", r"\bsklearn\b"],
    "pandas": [r"\bpandas\b"],
    "numpy": [r"\bnumpy\b"],
    "scipy": [r"\bscipy\b"],
    "xgboost": [r"\bxgboost\b", r"\blightgbm\b"],
    "huggingface": [r"\bhuggingface\b", r"\btransformers\b"],
    "llm": [r"\bllms?\b", r"\blarge\s+language\s+models?\b", r"\bgenerative\s+ai\b", r"\bgenai\b"],
    
    # Data Engineering & Analytics
    "spark": [r"\bapache\s+spark\b", r"\bpyspark\b", r"\bspark\b"],
    "hadoop": [r"\bhadoop\b", r"\bhdfs\b"],
    "kafka": [r"\bkafka\b", r"\bapache\s+kafka\b"],
    "tableau": [r"\btableau\b"],
    "power bi": [r"\bpower\s*bi\b"],
    "excel": [r"\bexcel\b", r"\bms\s+excel\b", r"\badvanced\s+excel\b"],
    "mongodb": [r"\bmongodb\b", r"\bmongo\b", r"\bnosql\b"],
    "snowflake": [r"\bsnowflake\b"],
    
    # Web & Frameworks
    "react": [r"\breact\b", r"\breactjs\b", r"\breact\.js\b"],
    "angular": [r"\bangular\b", r"\bangularjs\b"],
    "vue": [r"\bvue\b", r"\bvuejs\b"],
    "node.js": [r"\bnode\.?js\b", r"\bnodejs\b", r"\bnode\b"],
    "flask": [r"\bflask\b"],
    "django": [r"\bdjango\b"],
    "fastapi": [r"\bfastapi\b"],
    "spring boot": [r"\bspring\s+boot\b", r"\bspring\s+framework\b"],
    "html": [r"\bhtml\b", r"\bhtml5\b"],
    "css": [r"\bcss\b", r"\bcss3\b", r"\btailwind\b", r"\bbootstrap\b"],
    "graphql": [r"\bgraphql\b"],
    "rest api": [r"\brest\s*apis?\b", r"\brestful\b"],
    
    # Cloud, DevOps & Infrastructure
    "docker": [r"\bdocker\b", r"\bcontainerization\b"],
    "kubernetes": [r"\bkubernetes\b", r"\bk8s\b"],
    "aws": [r"\baws\b", r"\bamazon\s+web\s+services\b"],
    "azure": [r"\bazure\b", r"\bmicrosoft\s+azure\b"],
    "gcp": [r"\bgcp\b", r"\bgoogle\s+cloud\b"],
    "ci/cd": [r"\bci[/-]cd\b", r"\bjenkins\b", r"\bgithub\s+actions\b", r"\bgitlab\s+ci\b"],
    "git": [r"\bgit\b", r"\bgithub\b", r"\bgitlab\b"],
    "linux": [r"\blinux\b", r"\bunix\b", r"\bubuntu\b", r"\bbash\b", r"\bshell\s+scripting\b"],
    
    # Security & Networking
    "cybersecurity": [r"\bcybersecurity\b", r"\binformation\s+security\b", r"\bnetwork\s+security\b"],
    "penetration testing": [r"\bpenetration\s+testing\b", r"\bpen\s+test\b", r"\bethical\s+hacking\b"],
    "networking": [r"\bnetworking\b", r"\btcp[/-]ip\b", r"\bdns\b", r"\bwireshark\b"]
}


def extract_text_from_pdf(uploaded_file: Any) -> str:
    """Extracts raw text from a PDF file stream using pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        raise ValueError(f"Failed to read PDF: {str(e)}")

    return text


def extract_skills(text: str) -> List[str]:
    """Extracts technical skills from text using boundary-safe regex patterns."""
    if not text:
        return []
    
    lowered = text.lower()
    found_skills = set()
    
    for canonical_name, patterns in SKILL_TAXONOMY.items():
        for pattern in patterns:
            if re.search(pattern, lowered, re.IGNORECASE):
                found_skills.add(canonical_name)
                break
                
    return sorted(list(found_skills))


def estimate_experience_years(text: str) -> int:
    """Estimates candidate total years of experience from resume text.
    
    Heuristics:
    1. Looks for explicit statements like 'X+ years of experience' or 'X yrs experience'.
    2. Looks for employment date ranges (e.g. 2018-2022, 2019 to Present).
    """
    if not text:
        return 1

    current_year = datetime.now().year

    # Heuristic 1: Explicit pattern "X years of experience"
    exp_matches = re.findall(
        r"(\d{1,2})\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)",
        text,
        re.IGNORECASE
    )
    if exp_matches:
        years = [int(y) for y in exp_matches if int(y) <= 40]
        if years:
            return max(years)

    # Heuristic 2: Find date ranges (e.g., 2017 - 2021, 2019 - Present)
    year_ranges = re.findall(
        r"\b(19\d{2}|20\d{2})\s*(?:-|–|to|until)\s*(19\d{2}|20\d{2}|present|current)\b",
        text,
        re.IGNORECASE
    )

    total_years = 0
    calculated_spans = []

    for start_str, end_str in year_ranges:
        start_yr = int(start_str)
        if end_str.lower() in ["present", "current"]:
            end_yr = current_year
        else:
            end_yr = int(end_str)

        if 1980 <= start_yr <= current_year and start_yr <= end_yr <= current_year + 1:
            span = end_yr - start_yr
            if span > 0:
                calculated_spans.append(span)

    if calculated_spans:
        # Sum spans but cap at reasonable upper limit
        total_years = min(sum(calculated_spans), 35)
        if total_years > 0:
            return total_years

    # Default fallback for student/entry level resumes
    return 1


def extract_education(text: str) -> str:
    """Detects highest or primary education degree from resume text."""
    if not text:
        return "B.Tech"

    lowered = text.lower()

    if re.search(r"\b(ph\.?d|doctorate|doctor of philosophy)\b", lowered):
        return "PhD"
    elif re.search(r"\b(m\.?tech|m\.?s|master of technology|master of science|mca)\b", lowered):
        return "M.Tech"
    elif re.search(r"\b(mba|master of business administration)\b", lowered):
        return "MBA"
    elif re.search(r"\b(b\.?tech|b\.?e|bachelor of technology|bachelor of engineering)\b", lowered):
        return "B.Tech"
    elif re.search(r"\b(b\.?sc|bca|bachelor of science|bachelor of computer applications|b\.?com)\b", lowered):
        return "B.Sc"

    return "B.Tech"


def estimate_project_count(text: str) -> int:
    """Estimates number of projects mentioned in the resume."""
    if not text:
        return 1

    # Count project headings or bullet points under project section
    project_matches = re.findall(
        r"(?:project\s*\d+|project\s*name|developed\s+|built\s+|created\s+an?\s+|key\s+project)",
        text,
        re.IGNORECASE
    )
    
    count = len(project_matches)
    if count == 0:
        # Check if there is a 'Projects' section heading
        if re.search(r"\bprojects?\b", text, re.IGNORECASE):
            return 3
        return 1

    return min(max(count, 1), 10)


def extract_certifications(text: str) -> str:
    """Detects certifications from resume text matching model categories."""
    if not text:
        return "None"

    lowered = text.lower()

    if "aws" in lowered and ("certified" in lowered or "certification" in lowered or "solutions architect" in lowered):
        return "AWS Certified"
    elif "google" in lowered and ("ml" in lowered or "machine learning" in lowered or "tensorflow" in lowered or "cloud" in lowered):
        return "Google ML"
    elif "deep learning specialization" in lowered or "coursera deep learning" in lowered:
        return "Deep Learning Specialization"

    return "None"


def parse_resume_full(raw_text: str) -> Dict[str, Any]:
    """Parses a resume text and returns a comprehensive structured profile."""
    skills = extract_skills(raw_text)
    exp = estimate_experience_years(raw_text)
    edu = extract_education(raw_text)
    cert = extract_certifications(raw_text)
    projects = estimate_project_count(raw_text)

    return {
        "skills": skills,
        "skills_str": ", ".join(skills),
        "experience_years": exp,
        "education": edu,
        "certification": cert,
        "projects_count": projects,
        "clean_text": clean_text(raw_text)
    }
