"""AI Resume Screening & Candidate Fit Assessment System.

A modern Streamlit application providing automated resume parsing,
NLP job category classification, high-precision job description matching,
skill gap analysis, interview question generation, and explainability.
"""

import os
import sys
import joblib
import pandas as pd
import streamlit as st

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import clean_text
from src.parser import extract_text_from_pdf, parse_resume_full
from src.matcher import match_resume_to_job, synthesize_hiring_decision
from src.explainability import explain_prediction


# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Screening & Job Fit System",
    page_icon="ðŸ“„",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI aesthetics
st.markdown("""
<style>
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        color: #f8fafc;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 700;
        margin-top: 4px;
        color: #ffffff;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #38bdf8;
        margin-top: 2px;
    }
    /* Skill Badges */
    .badge-matched {
        display: inline-block;
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 9999px;
        padding: 4px 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 3px 4px;
    }
    .badge-missing {
        display: inline-block;
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 9999px;
        padding: 4px 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 3px 4px;
    }
    .badge-extra {
        display: inline-block;
        background-color: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.4);
        border-radius: 9999px;
        padding: 4px 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 3px 4px;
    }
    /* Tag lists */
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Model Loader Helper (Cached)
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    """Loads and caches the NLP job category classification model."""
    def find_model(model_name):
        paths = [
            os.path.join(BASE_DIR, "models", model_name),
            os.path.join(BASE_DIR, model_name)
        ]
        for p in paths:
            if os.path.exists(p):
                return joblib.load(p)
        return None

    cat_model = find_model("category_model.pkl")
    return cat_model


category_model = load_models()
models_ready = category_model is not None


# ---------------------------------------------------------
# Sidebar: System Metadata & Quick Presets
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/resume.png", width=64)
    st.title("System Control")
    
    if models_ready:
        st.success("ðŸŸ¢ Category NLP Model Loaded")
    else:
        st.warning("ðŸŸ¡ Standard Rule & Keyword Engine Active (Category model not found)")

    st.markdown("---")
    st.subheader("ðŸ“– Sample Job Description Presets")
    sample_jds = {
        "Data Scientist / ML Engineer": (
            "We are seeking a Data Scientist with 3+ years of experience in Python, SQL, and Machine Learning. "
            "Hands-on expertise in Deep Learning, PyTorch, TensorFlow, Pandas, Scikit-Learn, and Docker. "
            "Experience with Tableau, Power BI, and AWS cloud deployments is preferred. Bachelor's or Master's degree in Computer Science or related field."
        ),
        "Full Stack Web Developer": (
            "Looking for a Full Stack Developer with 2+ years of experience proficient in JavaScript, TypeScript, React, Next.js, and Node.js. "
            "Strong skills in HTML5, CSS3, Tailwind CSS, REST API development, PostgreSQL, MongoDB, and Git. "
            "Familiarity with Docker and AWS is a plus. Bachelor's degree required."
        ),
        "Cloud / DevOps Engineer": (
            "Seeking a DevOps Engineer with 3+ years of experience in Linux, Docker, Kubernetes, Terraform, CI/CD pipelines, "
            "and AWS or Azure cloud infrastructure. Proficient in Python, Bash scripting, Networking, and Git."
        ),
        "Cybersecurity Specialist": (
            "Hiring a Cybersecurity Analyst with 2+ years of experience in Networking, Linux, Penetration Testing, "
            "SIEM, Python scripting, Ethical Hacking, and Information Security best practices."
        ),
        "Java Backend Developer": (
            "Seeking a Senior Java Developer with 4+ years of experience in Java, Spring Boot, Microservices, REST APIs, "
            "SQL, PostgreSQL, Docker, Redis, and Git. Experience with AWS and CI/CD pipelines is a plus."
        )
    }

    def load_selected_jd():
        """Load a preset once into session state while leaving it editable."""
        choice = st.session_state.jd_preset
        if choice in sample_jds:
            st.session_state.job_description = sample_jds[choice]

    preset_choice = st.selectbox(
        "Load Sample JD Template",
        ["None (Custom)", *sample_jds.keys()],
        key="jd_preset",
        on_change=load_selected_jd,
    )

    st.markdown("---")
    st.caption("AI Resume Screening System v2.5 â€¢ JD Matching Edition")


# ---------------------------------------------------------
# Main UI Header
# ---------------------------------------------------------
st.title("ðŸ“„ Intelligent AI Resume Screening & JD Fit Assessment")
st.markdown(
    "Automated resume analysis, candidate-to-job matching, skill gap identification, "
    "and explainable AI screening recommendations tailored directly to your Job Description."
)

# ---------------------------------------------------------
# Input Section: Resume Upload & Job Description
# ---------------------------------------------------------
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    st.subheader("1. Candidate Resume")
    input_mode = st.radio("Resume Input Mode", ["Upload PDF File", "Paste Resume Text"], horizontal=True)
    
    resume_text = ""
    uploaded_file_name = "Pasted Resume"

    if input_mode == "Upload PDF File":
        uploaded_file = st.file_uploader(
            "Upload Resume (PDF)",
            type=["pdf"],
            help="Upload a standard PDF resume with readable text."
        )
        if uploaded_file is not None:
            uploaded_file_name = uploaded_file.name
            with st.spinner("Extracting text from PDF..."):
                try:
                    resume_text = extract_text_from_pdf(uploaded_file)
                except Exception as e:
                    st.error(f"Error parsing PDF file: {str(e)}")
                    st.stop()
    else:
        resume_text = st.text_area(
            "Paste Candidate Resume Text",
            height=200,
            placeholder="Paste raw resume text, work experience, education, and skills here...",
            key="pasted_resume_text"
        )

with col_right:
    st.subheader("2. Target Job Description")
    job_description = st.text_area(
        "Paste Job Description (JD)",
        height=200,
        placeholder="Paste target job requirements, qualifications, and required skills here...",
        key="job_description",
    )

# ---------------------------------------------------------
# Processing & Evaluation Workflow
# ---------------------------------------------------------
if resume_text.strip():
    # Automated Information Extraction from Resume
    profile = parse_resume_full(resume_text)
    cleaned_resume = profile["clean_text"]

    # 1. NLP Category Prediction
    if category_model:
        try:
            predicted_category = category_model.predict([cleaned_resume])[0]
        except Exception:
            predicted_category = "General Tech Profile"
    else:
        predicted_category = "Technical Profile"

    # 2. Comprehensive Dual Matching Engine (Skills + Semantics + Experience)
    match_result = match_resume_to_job(resume_text, job_description)

    # 3. Synthesized Realistic Hiring Decision
    has_jd = bool(job_description.strip())
    final_decision = synthesize_hiring_decision(
        match_result=match_result,
        has_job_description=has_jd
    )
    decision = final_decision["decision"]
    decision_cat = final_decision["category"]
    confidence = final_decision["confidence"]
    decision_reason = final_decision["reason"]

    # 4. Explainability & Insights
    explanation = explain_prediction(match_result=match_result)

    st.markdown("---")

    # ---------------------------------------------------------
    # Results Presentation: 4 KPI Cards
    # ---------------------------------------------------------
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        score_val = match_result['composite_score']
        score_color = "#34d399" if score_val >= 70 else ("#fbbf24" if score_val >= 45 else "#f87171")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Overall Match Score</div>
            <div class="metric-value" style="color: {score_color};">{score_val}%</div>
            <div class="metric-sub">{match_result['grade']}</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Skill Coverage</div>
            <div class="metric-value">{match_result['skill_score']}%</div>
            <div class="metric-sub">{match_result['total_matched_skills']} of {match_result['total_required_skills']} skills matched</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        if decision_cat == "Hire":
            rec_color = "#34d399"
        elif decision_cat == "Review":
            rec_color = "#fbbf24"
        else:
            rec_color = "#f87171"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Screening Recommendation</div>
            <div class="metric-value" style="color: {rec_color}; font-size: 1.15rem;">{decision}</div>
            <div class="metric-sub">Confidence: {confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Candidate Domain</div>
            <div class="metric-value" style="font-size: 1.2rem;">{predicted_category}</div>
            <div class="metric-sub">NLP Domain Classifier</div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Interactive Tabs
    # ---------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "ðŸ“Š Screening Overview & Fit",
        "ðŸŽ¯ Skill & Gap Analysis",
        "ðŸ’¡ Recruiter & Candidate Insights",
        "ðŸ“„ Parsed Profile & Text"
    ])

    # ------------------ TAB 1: OVERVIEW ------------------
    with tab1:
        st.subheader("Candidate Fit Assessment")
        
        col_bar, col_details = st.columns([3, 2])
        
        with col_bar:
            st.write(f"**Composite Match Score:** {match_result['composite_score']}%")
            st.progress(int(min(max(match_result["composite_score"], 0), 100)))
            st.caption("Composite score: 50% Skill Coverage + 30% Contextual Semantics + 20% Experience Qualification.")
            
            # Recommendation Banner
            if decision_cat == "Reject":
                st.error(f"**Screening Status:** {decision}\n\n**Rationale:** {decision_reason}")
            elif decision_cat == "Review":
                st.warning(f"**Screening Status:** {decision}\n\n**Rationale:** {decision_reason}")
            else:
                st.success(f"**Screening Status:** {decision}\n\n**Rationale:** {decision_reason}")
            
        with col_details:
            st.markdown("#### Requirement Alignment")
            cand_exp_display = f"{match_result['candidate_experience']} Year(s)" if match_result['candidate_experience'] > 0 else "0 Years (Fresher)"
            align_data = [
                {"Criteria": "Skills Matched", "Status": f"{match_result['total_matched_skills']} / {match_result['total_required_skills']} required"},
                {"Criteria": "Semantic Relevance", "Status": f"{match_result['semantic_score']}% contextual overlap"},
                {"Criteria": "Candidate Experience", "Status": cand_exp_display},
                {"Criteria": "Experience Alignment", "Status": match_result['experience_status']},
                {"Criteria": "Education", "Status": match_result['candidate_education']},
            ]
            st.table(pd.DataFrame(align_data))

    # ------------------ TAB 2: SKILL ANALYSIS ------------------
    with tab2:
        st.subheader("Detailed Skill Alignment & Gap Analysis")
        
        col_s1, col_s2, col_s3 = st.columns(3)

        with col_s1:
            st.markdown(f"#### ðŸŸ¢ Matched Skills ({len(match_result['matched_skills'])})")
            if match_result["matched_skills"]:
                badges_html = "".join([f'<span class="badge-matched">{s}</span>' for s in match_result["matched_skills"]])
                st.markdown(f'<div class="badge-container">{badges_html}</div>', unsafe_allow_html=True)
            else:
                st.info("No direct skill matches found between resume and job description.")

        with col_s2:
            st.markdown(f"#### ðŸ”´ Missing Required Skills ({len(match_result['missing_skills'])})")
            if match_result["missing_skills"]:
                badges_html = "".join([f'<span class="badge-missing">{s}</span>' for s in match_result["missing_skills"]])
                st.markdown(f'<div class="badge-container">{badges_html}</div>', unsafe_allow_html=True)
            else:
                if has_jd:
                    st.success("ðŸŽ‰ All required skills from the job description are present in the resume!")
                else:
                    st.write("No job description provided.")

        with col_s3:
            st.markdown(f"#### ðŸ”µ Additional Candidate Skills ({len(match_result['extra_skills'])})")
            if match_result["extra_skills"]:
                badges_html = "".join([f'<span class="badge-extra">{s}</span>' for s in match_result["extra_skills"]])
                st.markdown(f'<div class="badge-container">{badges_html}</div>', unsafe_allow_html=True)
            else:
                st.write("No additional technical skills recorded.")

    # ------------------ TAB 3: INSIGHTS & INTERVIEW QUESTIONS ------------------
    with tab3:
        st.subheader("ðŸ’¡ Explainable AI Factors & Actionable Insights")
        
        st.markdown("#### ðŸ” Decision Factors Breakdown")
        for reason in explanation["summary_reasons"]:
            st.markdown(f"â€¢ {reason}")

        st.markdown("---")
        st.markdown("#### ðŸŽ¯ Tailored Technical Screening Questions")
        st.caption("Auto-generated questions for technical recruiters based on candidate's skill matches and identified gaps:")
        for idx, q in enumerate(explanation["interview_questions"], 1):
            st.markdown(f"**{idx}.** {q}")

        st.markdown("---")
        st.markdown("#### ðŸš€ Candidate Resume Improvement Tips")
        for tip in explanation["improvement_tips"]:
            st.markdown(f"â€¢ {tip}")

    # ------------------ TAB 4: PARSED DETAILS & RAW TEXT ------------------
    with tab4:
        st.subheader("Extracted Candidate Profile")
        exp_meta_val = f"{profile['experience_years']} Year(s)" if profile['experience_years'] > 0 else "0 Years (Fresher)"
        meta_df = pd.DataFrame([{
            "Attribute": "Document / Source", "Extracted Value": uploaded_file_name
        }, {
            "Attribute": "Total Detected Technical Skills", "Extracted Value": str(len(profile["skills"]))
        }, {
            "Attribute": "Estimated Experience", "Extracted Value": exp_meta_val
        }, {
            "Attribute": "Highest Detected Education", "Extracted Value": profile["education"]
        }, {
            "Attribute": "Detected Certifications", "Extracted Value": profile["certification"]
        }, {
            "Attribute": "Estimated Projects Count", "Extracted Value": f"{profile['projects_count']} Project(s)"
        }])
        st.table(meta_df)


        with st.expander("ðŸ“„ View Extracted Text Content", expanded=False):
            st.text_area("Resume Text Content", value=resume_text, height=300)

    # Compliance Disclaimer
    st.markdown("---")
    st.caption(
        "âš–ï¸ **Ethical AI Disclaimer:** This system provides automated decision-support metrics and candidate "
        "relevance rankings based on job requirements. Final hiring decisions must be reviewed by authorized human recruiters."
    )

else:
    # Empty State Guidance
    st.info("ðŸ‘† Please upload a PDF resume or paste resume text in the left panel to begin screening against your target Job Description.")


