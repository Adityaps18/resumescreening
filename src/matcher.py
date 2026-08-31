"""Job Description Matching Engine with Skill & Semantic Similarity.

Combines:
1. Exact/Synonym Skill Coverage Analysis.
2. N-Gram TF-IDF Semantic Context Similarity.
3. Weighted Composite Scoring.
4. Role-Aware Holistic Hiring Recommendation Synthesis.
"""

from typing import Any, Dict, List, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.parser import extract_skills
from src.preprocessing import clean_text


def calculate_semantic_similarity(resume_text: str, job_description: str) -> float:
    """Calculates TF-IDF N-Gram Cosine Similarity between resume and job description.
    
    Returns:
        Similarity score as a percentage between 0.0 and 100.0.
    """
    clean_resume = clean_text(resume_text)
    clean_job = clean_text(job_description)

    if not clean_resume or not clean_job:
        return 0.0

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=2000
    )

    try:
        tfidf_matrix = vectorizer.fit_transform([clean_resume, clean_job])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(similarity) * 100.0, 2)
    except Exception:
        return 0.0


def match_resume_to_job(resume_text: str, job_description: str) -> Dict[str, Any]:
    """Evaluates candidate fit against a job description.
    
    Computes:
    - matched_skills: Skills required in job description present in resume.
    - missing_skills: Required skills absent from resume.
    - extra_skills: Additional skills candidate possesses.
    - skill_score: Percentage of required skills covered.
    - semantic_score: Contextual similarity between resume content and JD.
    - composite_score: Balanced overall match score.
    """
    resume_skills: Set[str] = set(extract_skills(resume_text))
    job_skills: Set[str] = set(extract_skills(job_description))

    matched = sorted(list(resume_skills.intersection(job_skills)))
    missing = sorted(list(job_skills - resume_skills))
    extra = sorted(list(resume_skills - job_skills))

    # Skill coverage calculation
    if not job_skills:
        skill_score = 100.0 if resume_skills else 0.0
    else:
        skill_score = (len(matched) / len(job_skills)) * 100.0

    skill_score = round(skill_score, 2)

    # Semantic similarity calculation
    semantic_score = calculate_semantic_similarity(resume_text, job_description)

    # Composite weighted score (60% skills, 40% contextual semantics)
    if job_skills:
        composite_score = round((0.60 * skill_score) + (0.40 * semantic_score), 2)
    else:
        composite_score = semantic_score

    # Human-readable summary
    if composite_score >= 70:
        summary_grade = "Strong Match"
    elif composite_score >= 40:
        summary_grade = "Moderate Match"
    else:
        summary_grade = "Low Match"

    return {
        "composite_score": composite_score,
        "skill_score": skill_score,
        "semantic_score": semantic_score,
        "matched_skills": matched,
        "missing_skills": missing,
        "extra_skills": extra,
        "grade": summary_grade,
        "total_required_skills": len(job_skills),
        "total_matched_skills": len(matched)
    }


def synthesize_hiring_decision(
    raw_model_decision: str,
    raw_confidence: float,
    match_result: Dict[str, Any],
    has_job_description: bool = True
) -> Dict[str, Any]:
    """Synthesizes ML model prediction with JD Match Score to produce a realistic recommendation.
    
    Prevents logical contradictions (e.g. recommending 'Hire' when candidate has an 11% job match).
    """
    if not has_job_description or match_result["total_required_skills"] == 0:
        return {
            "decision": raw_model_decision,
            "confidence": raw_confidence,
            "category": "Hire" if raw_model_decision == "Hire" else "Reject",
            "reason": "Evaluated based on overall candidate qualifications (No specific job requirements provided)."
        }

    comp_score = match_result["composite_score"]
    missing_count = len(match_result["missing_skills"])

    # Tier 1: Severe Mismatch (< 40% match)
    if comp_score < 40.0:
        return {
            "decision": "Reject (Role Mismatch)",
            "confidence": max(raw_confidence, 85.0),
            "category": "Reject",
            "reason": f"Candidate match score ({comp_score}%) is below the minimum threshold (40%). Missing {missing_count} required core skills for this role."
        }

    # Tier 2: Moderate Match (40% to 69%)
    elif comp_score < 70.0:
        if raw_model_decision == "Hire":
            return {
                "decision": "Review / Screen Further",
                "confidence": raw_confidence,
                "category": "Review",
                "reason": f"Candidate has a strong overall profile but moderate job overlap ({comp_score}%). A technical recruiter screening is recommended."
            }
        else:
            return {
                "decision": "Reject",
                "confidence": raw_confidence,
                "category": "Reject",
                "reason": f"Candidate profile and moderate job match ({comp_score}%) do not meet current benchmark requirements."
            }

    # Tier 3: Strong Match (>= 70%)
    else:
        if raw_model_decision == "Hire":
            return {
                "decision": "Hire (Strong Fit)",
                "confidence": raw_confidence,
                "category": "Hire",
                "reason": f"High candidate qualification alignment and strong job match score ({comp_score}%)."
            }
        else:
            return {
                "decision": "Review / Potential Fit",
                "confidence": 65.0,
                "category": "Review",
                "reason": f"Strong job match score ({comp_score}%), but profile metrics (experience or education) require recruiter review."
            }
