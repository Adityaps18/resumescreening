"""Model Explainability and Decision Breakdown Module.

Provides human-interpretable reasoning, feature attribution, tailored interview
screening questions, and candidate optimization recommendations.
"""

from typing import Any, Dict, List, Optional
import pandas as pd


def explain_prediction(
    pipeline: Any = None,
    candidate_df: Optional[pd.DataFrame] = None,
    match_result: Optional[Dict[str, Any]] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """Generates human-readable decision factors, interview questions, and suggestions.
    
    Args:
        pipeline: Optional classification pipeline.
        candidate_df: Optional DataFrame of candidate features.
        match_result: Dictionary from match_resume_to_job.
        
    Returns:
        Dictionary containing summary reasons, interview questions, and improvement tips.
    """
    reasons: List[str] = []
    interview_questions: List[str] = []
    improvement_tips: List[str] = []
    feature_impacts: List[Dict[str, Any]] = []

    if match_result:
        comp_score = match_result.get("composite_score", 0.0)
        skill_score = match_result.get("skill_score", 0.0)
        semantic_score = match_result.get("semantic_score", 0.0)
        missing = match_result.get("missing_skills", [])
        matched = match_result.get("matched_skills", [])
        extra = match_result.get("extra_skills", [])
        cand_exp = match_result.get("candidate_experience", 1)
        req_exp = match_result.get("required_experience")
        cand_edu = match_result.get("candidate_education", "Bachelor's")

        # 1. Job Role Fit Insights
        if comp_score >= 70.0:
            reasons.append(
                f"✅ **Strong Skill Alignment ({skill_score}%):** Candidate possesses {len(matched)} core competencies required in the JD ({', '.join(matched[:5])})."
            )
        elif comp_score >= 45.0:
            reasons.append(
                f"ℹ️ **Moderate Fit ({comp_score}%):** Candidate covers {len(matched)} skills ({', '.join(matched[:4])}), but misses {len(missing)} requirement(s) ({', '.join(missing[:4])})."
            )
        else:
            reasons.append(
                f"⚠️ **Skill Deficit ({skill_score}%):** Resume lacks critical mandatory skills ({', '.join(missing[:5]) or 'Key requirements'})."
            )

        # 2. Semantic Context
        if semantic_score >= 40.0:
            reasons.append(
                f"✅ **High Semantic Relevance ({semantic_score}%):** Resume vocabulary strongly corresponds to the job description terminology."
            )
        else:
            reasons.append(
                f"ℹ️ **Semantic Relevance ({semantic_score}%):** General domain terminology overlap."
            )

        # 3. Experience Analysis
        if req_exp == 0:
            if cand_exp == 0:
                reasons.append("✅ **Experience Criteria Met:** Candidate profile aligns perfectly with Fresher / Entry-Level position requirements.")
            else:
                reasons.append(f"✅ **Experience Criteria Met:** Candidate has {cand_exp} year(s) of experience for a Fresher / Entry-Level position.")
        elif req_exp is not None and req_exp > 0:
            if cand_exp >= req_exp:
                reasons.append(
                    f"✅ **Experience Criteria Met:** Candidate has {cand_exp} years of experience (JD required minimum {req_exp} years)."
                )
            else:
                reasons.append(
                    f"⚠️ **Experience Gap:** Candidate has {cand_exp} year(s) of experience vs {req_exp} years requested in the JD."
                )
        else:
            if cand_exp == 0:
                reasons.append("ℹ️ **Candidate Experience:** Fresher / Entry-Level profile (0 years industry experience).")
            else:
                reasons.append(f"ℹ️ **Candidate Experience:** ~{cand_exp} years estimated industry experience.")

        # 4. Education
        reasons.append(f"🎓 **Education Level:** {cand_edu} detected.")


        # 5. Generate Target Interview Screening Questions
        if missing:
            for skill in missing[:3]:
                interview_questions.append(
                    f"Can you explain your familiarity or hands-on experience with **{skill.title()}**, or an equivalent tool you have used in production?"
                )
        if matched:
            top_skill = matched[0]
            interview_questions.append(
                f"Walk us through a challenging project where you leveraged **{top_skill.title()}** to deliver measurable business impact."
            )
        interview_questions.append(
            "How do you approach learning and rapidly mastering new frameworks or architectures required for a new role?"
        )

        # 6. Candidate Resume Optimization Tips
        if missing:
            improvement_tips.append(
                f"Highlight practical project experience with missing core skills: **{', '.join(missing[:4])}**."
            )
        if cand_exp < 2:
            improvement_tips.append(
                "Include quantifiable project metrics, GitHub repository links, and deployment architecture details to demonstrate depth."
            )
        if len(matched) < 3:
            improvement_tips.append(
                "Align resume terminology and action verbs more closely with the specific requirements in the job posting."
            )

        # 7. Impact Table
        feature_impacts.append({
            "feature": "Skill Match Coverage",
            "impact": f"{skill_score}%",
            "type": "Positive" if skill_score >= 50 else "Negative"
        })
        feature_impacts.append({
            "feature": "Contextual Semantic Similarity",
            "impact": f"{semantic_score}%",
            "type": "Positive" if semantic_score >= 35 else "Neutral"
        })
        feature_impacts.append({
            "feature": "Experience Qualification",
            "impact": f"{cand_exp} yrs",
            "type": "Positive" if (req_exp is None or cand_exp >= req_exp) else "Gap"
        })

    return {
        "summary_reasons": reasons,
        "interview_questions": interview_questions,
        "improvement_tips": improvement_tips,
        "feature_contributions": feature_impacts
    }

