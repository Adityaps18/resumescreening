"""Resume & Job Description Parsing and Automated Information Extraction Module.

Extracts:
1. Technical skills (with boundary-safe matching and synonym aliasing across 160+ skills).
2. Estimated years of experience (section-aware, preventing education/college date leakage).
3. Highest education level (PhD, Master's, Bachelor's, etc.).
4. Estimated projects count and industry certifications.
5. Target requirements from Job Descriptions (Fresher/Entry-level vs experienced, minimum years, degree, skills).
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import pdfplumber

from src.preprocessing import clean_text


# Comprehensive Skill Taxonomy & Canonical Aliases (160+ industry technical skills)
SKILL_TAXONOMY: Dict[str, List[str]] = {
    # Foundational CS & Core Concepts
    "data science": [r"\bdata\s+science\b", r"\bdata\s+scientist\b"],
    "data structures & algorithms": [r"\bdata\s+structures?\b", r"\balgorithms?\b", r"\bdsa\b"],
    "oop": [r"\boop\b", r"\boops\b", r"\bobject[\s-]oriented\b", r"\bobject[\s-]oriented\s+programming\b"],
    "databases": [r"\bdatabases?\b", r"\brdbms\b", r"\bdatabase\s+management\b", r"\bsql\s+database\b"],
    "software development": [r"\bsoftware\s+development\b", r"\bsoftware\s+engineering\b", r"\bsoftware\s+design\b", r"\bsoftware\s+applications?\b", r"\bsoftware\s+development\s+principles\b"],
    "programming": [r"\bprogramming\b", r"\bcoding\b", r"\bproblem\s+solving\b", r"\bclean\s+code\b"],
    "testing & debugging": [r"\btesting\b", r"\bdebugging\b", r"\bsoftware\s+testing\b", r"\bunit\s+testing\b", r"\bcode\s+quality\b"],
    "system design": [r"\bsystem\s+design\b", r"\bdesign\s+patterns\b", r"\bdistributed\s+systems?\b"],
    "computer science": [r"\bcomputer\s+science\b", r"\bcs\s+fundamentals\b"],

    # Programming & Scripting Languages
    "python": [r"\bpython\b", r"\bpython3\b", r"\bpy\b"],
    "java": [r"\bjava\b(?!script)"],
    "c++": [r"\bc\+\+\b", r"\bcpp\b"],
    "c#": [r"\bc#\b", r"\bcsharp\b", r"\b\.net\b", r"\bdotnet\b"],
    "c": [r"\bc\s+programming\b", r"\bansi\s+c\b", r"\bembedded\s+c\b"],
    "javascript": [r"\bjavascript\b", r"\bjs\b", r"\bes6\b", r"\becmascript\b"],
    "typescript": [r"\btypescript\b", r"\bts\b"],
    "go": [r"\bgolang\b", r"\bgo\s+language\b", r"\bgo\s+programming\b"],
    "rust": [r"\brust\b", r"\brustlang\b"],
    "r": [r"\br\s+programming\b", r"\br\s+language\b", r"\b(using|in)\s+r\b"],
    "php": [r"\bphp\b", r"\bphp8\b"],
    "ruby": [r"\bruby\b", r"\bruby\s+on\s+rails\b", r"\brails\b"],
    "scala": [r"\bscala\b"],
    "swift": [r"\bswift\b", r"\bswiftui\b", r"\bios\s+development\b"],
    "kotlin": [r"\bkotlin\b", r"\bandroid\s+development\b"],
    "dart": [r"\bdart\b", r"\bflutter\b"],
    "sql": [r"\bsql\b", r"\bt-sql\b", r"\bpl/sql\b", r"\bansi\s+sql\b"],
    "bash": [r"\bbash\b", r"\bshell\s+scripting\b", r"\bpowershell\b", r"\bsh\b"],
    
    # Web & Frontend Development
    "html": [r"\bhtml\b", r"\bhtml5\b"],
    "css": [r"\bcss\b", r"\bcss3\b", r"\bsass\b", r"\bscss\b", r"\bless\b"],
    "tailwind css": [r"\btailwind\b", r"\btailwindcss\b"],
    "bootstrap": [r"\bbootstrap\b", r"\bbootstrap5\b"],
    "react": [r"\breact\b", r"\breactjs\b", r"\breact\.js\b"],
    "next.js": [r"\bnext\.?js\b", r"\bnextjs\b"],
    "vue": [r"\bvue\b", r"\bvuejs\b", r"\bvue\.js\b"],
    "nuxt.js": [r"\bnuxt\.?js\b", r"\bnuxtjs\b"],
    "angular": [r"\bangular\b", r"\bangularjs\b", r"\bangular\.js\b"],
    "svelte": [r"\bsvelte\b", r"\bsveltekit\b"],
    "redux": [r"\bredux\b", r"\bredux\s+toolkit\b", r"\bzustand\b", r"\bmobx\b"],
    "jquery": [r"\bjquery\b"],

    # Backend & Frameworks
    "node.js": [r"\bnode\.?js\b", r"\bnodejs\b", r"\bnode\b"],
    "express": [r"\bexpress\.?js\b", r"\bexpressjs\b", r"\bexpress\b"],
    "django": [r"\bdjango\b", r"\bdjango\s+rest\s+framework\b", r"\bdrf\b"],
    "flask": [r"\bflask\b"],
    "fastapi": [r"\bfastapi\b"],
    "spring boot": [r"\bspring\s+boot\b", r"\bspring\s+framework\b", r"\bspring\b"],
    "asp.net": [r"\basp\.net\b", r"\b\.net\s+core\b", r"\bdotnet\s+core\b"],
    "laravel": [r"\blaravel\b"],
    "graphql": [r"\bgraphql\b", r"\bapollo\b"],
    "rest api": [r"\brest\s*apis?\b", r"\brestful\b", r"\brest\s+web\s+services\b"],
    "microservices": [r"\bmicroservices?\b", r"\bmicroservice\s+architecture\b", r"\bsoa\b"],
    "websockets": [r"\bwebsockets?\b", r"\bsocket\.io\b"],

    # Data Science & Machine Learning
    "machine learning": [r"\bmachine\s+learning\b", r"\bml\b"],
    "deep learning": [r"\bdeep\s+learning\b", r"\bdl\b", r"\bneural\s+networks?\b", r"\bcnn\b", r"\brnn\b", r"\blstm\b"],
    "natural language processing": [r"\bnlp\b", r"\bnatural\s+language\s+processing\b", r"\btext\s+mining\b"],
    "computer vision": [r"\bcomputer\s+vision\b", r"\bopencv\b", r"\byolo\b", r"\bimage\s+processing\b"],
    "generative ai": [r"\bgenerative\s+ai\b", r"\bgenai\b", r"\bllms?\b", r"\blarge\s+language\s+models?\b", r"\bprompt\s+engineering\b"],
    "rag": [r"\brag\b", r"\bretrieval\s+augmented\s+generation\b"],
    "langchain": [r"\blangchain\b", r"\bllamaindex\b"],
    "transformers": [r"\btransformers?\b", r"\bhugging\s*face\b", r"\bbert\b", r"\bgpt\b"],
    "tensorflow": [r"\btensorflow\b", r"\btf\b"],
    "pytorch": [r"\bpytorch\b", r"\btorch\b"],
    "keras": [r"\bkeras\b"],
    "scikit-learn": [r"\bscikit-learn\b", r"\bsklearn\b"],
    "pandas": [r"\bpandas\b"],
    "numpy": [r"\bnumpy\b"],
    "scipy": [r"\bscipy\b"],
    "xgboost": [r"\bxgboost\b", r"\blightgbm\b", r"\bcatboost\b"],
    "statistics": [r"\bstatistics\b", r"\bstatistical\s+analysis\b", r"\bhypothesis\s+testing\b", r"\ba/b\s+testing\b"],
    "data analysis": [r"\bdata\s+analysis\b", r"\bdata\s+analytics\b", r"\beda\b", r"\bexploratory\s+data\s+analysis\b"],
    "data visualization": [r"\bdata\s+visualization\b", r"\bmatplotlib\b", r"\bseaborn\b", r"\bplotly\b"],
    "tableau": [r"\btableau\b"],
    "power bi": [r"\bpower\s*bi\b", r"\bpowerbi\b", r"\bdax\b"],
    "excel": [r"\bexcel\b", r"\bms\s+excel\b", r"\badvanced\s+excel\b", r"\bvba\b"],

    # Databases & Big Data Engineering
    "postgresql": [r"\bpostgresql\b", r"\bpostgres\b"],
    "mysql": [r"\bmysql\b"],
    "mongodb": [r"\bmongodb\b", r"\bmongo\b", r"\bnosql\b"],
    "redis": [r"\bredis\b"],
    "cassandra": [r"\bcassandra\b"],
    "dynamodb": [r"\bdynamodb\b"],
    "oracle": [r"\boracle\s+database\b", r"\boracle\s+db\b", r"\boracle\s+sql\b"],
    "sqlite": [r"\bsqlite\b", r"\bsqlite3\b"],
    "sql server": [r"\bsql\s+server\b", r"\bmssql\b"],
    "elasticsearch": [r"\belasticsearch\b", r"\belk\s+stack\b", r"\bkibana\b"],
    "neo4j": [r"\bneo4j\b", r"\bgraph\s+database\b"],
    "spark": [r"\bapache\s+spark\b", r"\bpyspark\b", r"\bspark\b"],
    "hadoop": [r"\bhadoop\b", r"\bhdfs\b", r"\bmapreduce\b", r"\bhive\b"],
    "kafka": [r"\bkafka\b", r"\bapache\s+kafka\b"],
    "airflow": [r"\bairflow\b", r"\bapache\s+airflow\b"],
    "snowflake": [r"\bsnowflake\b"],
    "databricks": [r"\bdatabricks\b"],
    "bigquery": [r"\bbigquery\b", r"\bgoogle\s+bigquery\b"],
    "redshift": [r"\bredshift\b", r"\bamazon\s+redshift\b"],
    "etl": [r"\betl\b", r"\bdata\s+pipeline\b", r"\bdata\s+warehousing\b", r"\bdbt\b"],

    # Cloud, DevOps & Infrastructure
    "aws": [r"\baws\b", r"\bamazon\s+web\s+services\b", r"\bec2\b", r"\bs3\b", r"\blambda\b", r"\bcloudformation\b"],
    "azure": [r"\bazure\b", r"\bmicrosoft\s+azure\b"],
    "gcp": [r"\bgcp\b", r"\bgoogle\s+cloud\b", r"\bgoogle\s+cloud\s+platform\b"],
    "docker": [r"\bdocker\b", r"\bcontainerization\b", r"\bcontainers\b"],
    "kubernetes": [r"\bkubernetes\b", r"\bk8s\b"],
    "terraform": [r"\bterraform\b", r"\binfrastructure\s+as\s+code\b", r"\biac\b"],
    "ansible": [r"\bansible\b"],
    "jenkins": [r"\bjenkins\b"],
    "ci/cd": [r"\bci[/-]cd\b", r"\bcontinuous\s+integration\b", r"\bcontinuous\s+deployment\b", r"\bgithub\s+actions\b", r"\bgitlab\s+ci\b"],
    "git": [r"\bgit\b", r"\bgithub\b", r"\bgitlab\b", r"\bbitbucket\b", r"\bversion\s+control\b"],
    "linux": [r"\blinux\b", r"\bunix\b", r"\bubuntu\b", r"\bcentos\b", r"\bdebian\b", r"\bredhat\b"],
    "nginx": [r"\bnginx\b", r"\bapache\s+server\b"],
    "monitoring": [r"\bprometheus\b", r"\bgrafana\b", r"\bdatadog\b", r"\bsplunk\b"],

    # Security & Networking
    "cybersecurity": [r"\bcybersecurity\b", r"\binformation\s+security\b", r"\binfosec\b", r"\bnetwork\s+security\b"],
    "penetration testing": [r"\bpenetration\s+testing\b", r"\bpen\s+test\b", r"\bethical\s+hacking\b", r"\bkali\s+linux\b", r"\bmetasploit\b", r"\bburp\s+suite\b"],
    "cryptography": [r"\bcryptography\b", r"\bencryption\b", r"\bpki\b", r"\bssl/tls\b"],
    "siem": [r"\bsiem\b", r"\bsoc\b", r"\bvulnerability\s+assessment\b", r"\bowasp\b"],
    "networking": [r"\bnetworking\b", r"\btcp[/-]ip\b", r"\bdns\b", r"\bdhcp\b", r"\bvpn\b", r"\bfirewalls?\b", r"\bwireshark\b"],
    
    # Testing & QA
    "selenium": [r"\bselenium\b", r"\bcypress\b", r"\bplaywright\b"],
    "postman": [r"\bpostman\b", r"\bapi\s+testing\b", r"\bswagger\b"],
    "agile": [r"\bagile\b", r"\bscrum\b", r"\bkanban\b", r"\bsprint\b", r"\bjira\b", r"\bconfluence\b"]
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
    """Estimates candidate total years of professional experience from resume text.
    
    Prevents false positives from college degree durations (e.g. 2020-2024 = 4 years B.Tech)
    and correctly marks freshers and students as 0 years of experience.
    """
    if not text or not text.strip():
        return 0

    current_year = datetime.now().year
    lowered = text.lower()

    # 1. Explicit Fresher / Student Indicators
    fresher_indicators = [
        r"\bfresher\b",
        r"\bentry[\s-]level\b",
        r"\b0\s*(?:years?|yrs?)(?:\s+of)?\s+experience\b",
        r"\bno\s+(?:prior\s+)?experience\b",
        r"\bundergraduate\s+student\b",
        r"\bgraduate\s+student\b",
        r"\bseeking\s+(?:an?\s+)?(?:entry[\s-]level|internship)\b",
        r"\bbatch\s+of\s+20\d{2}\b",
        r"\bgraduating\s+in\s+20\d{2}\b"
    ]
    for ind in fresher_indicators:
        if re.search(ind, lowered):
            # Check if there is an explicit non-zero experience statement
            explicit = re.findall(
                r"(\d{1,2})\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:work|professional|industry|relevant)\s+experience\b",
                text,
                re.IGNORECASE
            )
            non_zero = [int(y) for y in explicit if 1 <= int(y) <= 40]
            if not non_zero:
                return 0

    # 2. Explicit Statement "X years of experience"
    exp_matches = re.findall(
        r"(\d{1,2})\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:work|professional|industry|relevant)\s+experience\b",
        text,
        re.IGNORECASE
    )
    if not exp_matches:
        exp_matches = re.findall(
            r"(\d{1,2})\+?\s*(?:years?|yrs?)\s+(?:of\s+)?experience\b",
            text,
            re.IGNORECASE
        )
    if exp_matches:
        years = [int(y) for y in exp_matches if int(y) <= 40]
        if years:
            return max(years)

    # 3. Section-Aware Date Range Extraction (ONLY from Experience / Employment sections)
    # Split text by common section headers
    sections = re.split(r"\n\s*(?=[A-Z\s]{3,30}(?:\:|\n))", text)
    exp_text_blocks = []
    has_work_exp_section = False

    for sec in sections:
        first_line = sec.strip().split("\n")[0].upper()
        # Skip Education, Academics, Projects, Skills sections
        if any(skip_kw in first_line for skip_kw in ["EDUCATION", "ACADEMIC", "QUALIFICATION", "DEGREE", "SCHOOL", "COLLEGE", "SKILLS", "CERTIFICATION"]):
            continue
        # Check if this is an experience section
        if any(exp_kw in first_line for exp_kw in ["EXPERIENCE", "EMPLOYMENT", "WORK HISTORY", "WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE", "CAREER", "INTERNSHIP"]):
            has_work_exp_section = True
            exp_text_blocks.append(sec)

    # If there is no dedicated work experience section in the resume, it's a fresher / student resume
    if not has_work_exp_section or not exp_text_blocks:
        return 0

    target_text = "\n".join(exp_text_blocks)

    year_ranges = re.findall(
        r"\b(19\d{2}|20\d{2})\s*(?:-|–|to|until)\s*(19\d{2}|20\d{2}|present|current)\b",
        target_text,
        re.IGNORECASE
    )

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
        total_years = min(sum(calculated_spans), 35)
        return total_years

    return 0


def extract_required_experience_from_jd(jd_text: str) -> Optional[int]:
    """Extracts required minimum years of experience stated in a Job Description.
    
    Correctly recognizes 'Fresher', 'Entry-Level', and '0 years' as 0 years required.
    """
    if not jd_text:
        return None

    lowered = jd_text.lower()

    # 1. Fresher / Entry Level check
    fresher_pats = [
        r"\bfresher\b",
        r"\bentry[\s-]level\b",
        r"\bintern(?:ship)?\b",
        r"\b0\s*(?:-\s*1)?\s*(?:years?|yrs?)\b",
        r"\bno\s+(?:prior\s+)?experience\b",
        r"\bcollege\s+graduates?\b",
        r"\bfreshers?\s+(?:can\s+apply|welcome)\b"
    ]
    for pat in fresher_pats:
        if re.search(pat, lowered):
            return 0

    # 2. Explicit Numeric Years
    patterns = [
        r"(?:minimum|at least|min\.?)\s*(\d{1,2})\+?\s*(?:years?|yrs?)",
        r"(\d{1,2})\+?\s*(?:-\s*\d{1,2})?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp|relevant|industry|work)",
        r"(\d{1,2})\+?\s*(?:years?|yrs?)\s+required"
    ]

    found = []
    for pat in patterns:
        matches = re.findall(pat, jd_text, re.IGNORECASE)
        for m in matches:
            if isinstance(m, tuple):
                m = m[0]
            val = int(m)
            if 1 <= val <= 25:
                found.append(val)

    if found:
        return min(found)  # Return minimum required years stated
    return None


def extract_education(text: str) -> str:
    """Detects highest or primary education degree from resume text."""
    if not text:
        return "B.Tech"

    lowered = text.lower()

    if re.search(r"\b(ph\.?d|doctorate|doctor of philosophy)\b", lowered):
        return "PhD"
    elif re.search(r"\b(m\.?tech|m\.?s|master of technology|master of science|mca|masters?)\b", lowered):
        return "M.Tech"
    elif re.search(r"\b(mba|master of business administration)\b", lowered):
        return "MBA"
    elif re.search(r"\b(b\.?tech|b\.?e|bachelor of technology|bachelor of engineering|bachelors?)\b", lowered):
        return "B.Tech"
    elif re.search(r"\b(b\.?sc|bca|bachelor of science|bachelor of computer applications|b\.?com)\b", lowered):
        return "B.Sc"

    return "B.Tech"



def extract_required_education_from_jd(jd_text: str) -> Optional[str]:
    """Extracts degree requirements mentioned in a Job Description."""
    if not jd_text:
        return None

    lowered = jd_text.lower()
    if re.search(r"\b(ph\.?d|doctorate)\b", lowered):
        return "PhD"
    elif re.search(r"\b(master'?s|m\.?tech|m\.?s|mca|mba)\b", lowered):
        return "Master's"
    elif re.search(r"\b(bachelor'?s|b\.?tech|b\.?e|b\.?sc|bca|undergraduate|degree in computer science)\b", lowered):
        return "Bachelor's"

    return None


def estimate_project_count(text: str) -> int:
    """Estimates number of projects mentioned in the resume."""
    if not text:
        return 1

    project_matches = re.findall(
        r"(?:project\s*\d+|project\s*name|developed\s+|built\s+|created\s+an?\s+|key\s+project|implemented\s+)",
        text,
        re.IGNORECASE
    )
    
    count = len(project_matches)
    if count == 0:
        if re.search(r"\bprojects?\b", text, re.IGNORECASE):
            return 3
        return 1

    return min(max(count, 1), 10)


def extract_certifications(text: str) -> str:
    """Detects certifications from resume text."""
    if not text:
        return "None"

    lowered = text.lower()
    certs = []

    if "aws" in lowered and ("certified" in lowered or "certification" in lowered or "solutions architect" in lowered):
        certs.append("AWS Certified")
    if "google" in lowered and ("ml" in lowered or "machine learning" in lowered or "cloud" in lowered or "tensorflow" in lowered):
        certs.append("Google Cloud / ML Certified")
    if "azure" in lowered and ("certified" in lowered or "certification" in lowered or "az-" in lowered):
        certs.append("Azure Certified")
    if "deep learning specialization" in lowered or "coursera deep learning" in lowered:
        certs.append("Deep Learning Specialization")
    if "kubernetes" in lowered and ("certified" in lowered or "cka" in lowered or "ckad" in lowered):
        certs.append("Certified Kubernetes (CKA/CKAD)")
    if "cissp" in lowered or "ceh" in lowered or "certified ethical hacker" in lowered:
        certs.append("Security Certified (CISSP/CEH)")

    return ", ".join(certs) if certs else "None"


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

