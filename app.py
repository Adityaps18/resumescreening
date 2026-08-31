"""AI Resume Screening & Candidate Fit Assessment System.

A modern Streamlit application providing automated resume parsing,
NLP job category classification, dual-score job description matching,
candidate recommendation, and explainability.
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
    page_title="AI Resume Screening System",
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
        font-size: 0.85rem;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.5rem;
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
    """Loads and caches serialized classification models."""
    def find_model(model_name):
        paths = [
            os.path.join(BASE_DIR, "models", model_name),
            os.path.join(BASE_DIR, model_name)
        ]
        for p in paths:
            if os.path.exists(p):
                return joblib.load(p)
        raise FileNotFoundError(f"Model file {model_name} not found.")

    cat_model = find_model("category_model.pkl")
    scr_model = find_model("screening_model.pkl")
    return cat_model, scr_model


try:
    category_model, screening_model = load_models()
    models_ready = True
except Exception as err:
    models_ready = False
    model_load_error = str(err)


# ---------------------------------------------------------
# Sidebar: System Metadata & Controls
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/resume.png", width=64)
    st.title("System Control")
    
    if models_ready:
        st.success("ðŸŸ¢ ML Models Loaded Successfully")
    else:
        st.error(f"ðŸ”´ Model Loading Error: {model_load_error}")
        st.info("Run `python src/train.py` to train and generate models.")

    st.markdown("---")
    st.subheader("âš™ï¸ Target Role Settings")
    target_job_role = st.selectbox(
        "Default Target Job Role",
        ["Data Scientist", "Software Engineer", "AI Researcher", "Cybersecurity Analyst", "DevOps Engineer", "Web Developer"],
        index=0
    )

    st.markdown("---")
    st.subheader("ðŸ“– Quick Job Description Presets")
    sample_jds = {
        "Data Scientist / ML Engineer": (
            "We are seeking a Data Scientist with 3+ years of experience in Python, SQL, and Machine Learning. "
            "Hands-on expertise in Deep Learning, PyTorch or TensorFlow, Pandas, Scikit-Learn, and Docker. "
            "Experience with Tableau or Power BI and AWS cloud deployments is preferred."
        ),
        "Full Stack Web Developer": (
            "Looking for a Full Stack Developer proficient in JavaScript, TypeScript, React, and Node.js. "
            "Strong skills in HTML5, CSS3, REST API development, SQL databases (PostgreSQL/MySQL), and Git. "
            "Familiarity with Docker and AWS is a plus."
        ),
        "Cloud / DevOps Engineer": (
            "Seeking a DevOps Engineer with expertise in Linux, Docker, Kubernetes, CI/CD pipelines, "
            "and AWS or Azure cloud infrastructure. Proficient in Python or Bash scripting, Networking, and Git."
        ),
        "Cybersecurity Specialist": (
            "Hiring a Cybersecurity Analyst with solid knowledge in Networking, Linux, Penetration Testing, "
            "Python scripting, and Information Security best practices."
        )
    }

    def load_selected_jd():
        """Load a preset once, while leaving manually entered text editable."""
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
    st.caption("AI Resume Screening System v2.0 â€¢ Academic & Portfolio Edition")


# ---------------------------------------------------------
# Main UI Header
# ---------------------------------------------------------
st.title("ðŸ“„ Intelligent AI Resume Screening System")
st.markdown(
    "Automated resume analysis, candidate-to-job matching, job category classification, "
    "and explainable AI screening recommendations."
)

if not models_ready:
    st.stop()

# ---------------------------------------------------------
# Input Section: Resume Upload & Job Description
# ---------------------------------------------------------
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    st.subheader("1. Candidate Resume")
    uploaded_file = st.file_uploader(
        "Upload Resume PDF",
        type=["pdf"],
        help="Upload a standard PDF resume with readable text."
    )

with col_right:
    st.subheader("2. Job Description")
    job_description = st.text_area(
        "Paste Target Job Description",
        height=165,
        placeholder="Paste target job requirements and required skills here...",
        key="job_description",
    )

# ---------------------------------------------------------
# Processing & Evaluation Workflow
# ---------------------------------------------------------
if uploaded_file is not None:
    with st.spinner("Extracting text and parsing candidate profile..."):
        try:
            resume_text = extract_text_from_pdf(uploaded_file)
        except Exception as e:
            st.error(f"Error parsing PDF file: {str(e)}")
            st.stop()

    if not resume_text.strip():
        st.error(
            "âš ï¸ No readable text found in this PDF. "
            "Please ensure the PDF is text-based and not a scanned image."
        )
        st.stop()

    # Automated Information Extraction
    profile = parse_resume_full(resume_text)
    cleaned_resume = profile["clean_text"]

    # Preserve recruiter edits across Streamlit reruns; reset only for a new resume.
    resume_id = f"{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state.get("profile_resume_id") != resume_id:
        st.session_state.profile_resume_id = resume_id
        st.session_state.adj_exp = int(profile["experience_years"])
        st.session_state.adj_edu = profile["education"] if profile["education"] in ["B.Tech", "B.Sc", "M.Tech", "MBA", "PhD"] else "B.Tech"
        st.session_state.adj_projects = int(profile["projects_count"])
        st.session_state.adj_cert = profile["certification"] if profile["certification"] in ["None", "AWS Certified", "Google ML", "Deep Learning Specialization"] else "None"
        st.session_state.adj_role = "Data Scientist"

    # ---------------------------------------------------------
    # Auto-Parsed Profile & Manual Adjustment Expander
    # ---------------------------------------------------------
    with st.expander("ðŸ“ Review Auto-Extracted Candidate Profile (Click to Adjust)", expanded=False):
        st.info("The system automatically detected these attributes. You can adjust any field before final screening.")
        c1, c2, c3 = st.columns(3)
        with c1:
            adj_exp = st.number_input(
                "Experience (Years)",
                min_value=0,
                max_value=50,
                key="adj_exp",
            )
            adj_edu = st.selectbox(
                "Education",
                ["B.Tech", "B.Sc", "M.Tech", "MBA", "PhD"],
                key="adj_edu",
            )
        with c2:
            adj_projects = st.number_input(
                "Projects Count",
                min_value=0,
                max_value=50,
                key="adj_projects",
            )
            adj_cert = st.selectbox(
                "Certification",
                ["None", "AWS Certified", "Google ML", "Deep Learning Specialization"],
                key="adj_cert",
            )
        with c3:
            adj_role = st.selectbox(
                "Candidate Target Role",
                ["Data Scientist", "Software Engineer", "AI Researcher", "Cybersecurity Analyst"],
                key="adj_role",
            )

    # ---------------------------------------------------------
    # Run AI/ML Models
    # ---------------------------------------------------------
    # 1. NLP Category Prediction
    predicted_category = category_model.predict([cleaned_resume])[0]

    # 2. Candidate General Qualification Model
    candidate_data = pd.DataFrame([{
        "Skills": profile["skills_str"],
        "Experience (Years)": adj_exp,
        "Education": adj_edu,
        "Certifications": adj_cert,
        "Job Role": adj_role,
        "Projects Count": adj_projects
    }])

    raw_decision = screening_model.predict(candidate_data)[0]
    probabilities = screening_model.predict_proba(candidate_data)[0]
    classes = screening_model.named_steps["classifier"].classes_
    prob_map = dict(zip(classes, probabilities))
    raw_confidence = prob_map[raw_decision] * 100.0

    # 3. Dual Matching Engine
    match_result = match_resume_to_job(resume_text, job_description)

    # 4. Synthesized Holistic Decision (Integrates ML Profile + Specific Job Fit)
    has_jd = bool(job_description.strip())
    final_decision = synthesize_hiring_decision(
        raw_model_decision=raw_decision,
        raw_confidence=raw_confidence,
        match_result=match_result,
        has_job_description=has_jd
    )
    decision = final_decision["decision"]
    decision_cat = final_decision["category"]
    confidence = final_decision["confidence"]
    decision_reason = final_decision["reason"]

    # 5. Explainability Factors
    explanation = explain_prediction(screening_model, candidate_data, match_result=match_result)

    st.markdown("---")

    # ---------------------------------------------------------
    # Results Presentation Tabs
    # ---------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "ðŸ“Š Overview & Decision",
        "ðŸŽ¯ Skill & Fit Analysis",
        "ðŸ’¡ AI Decision Explainability",
        "ðŸ“„ Parsed Profile & Text"
    ])

    # ------------------ TAB 1: OVERVIEW ------------------
    with tab1:
        st.subheader("Screening Overview")
        
        # 4 High-Impact Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Predicted Domain</div>
                <div class="metric-value" style="font-size: 1.25rem;">{predicted_category}</div>
                <div class="metric-sub">NLP Domain Classifier</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m2:
            score_color = "#34d399" if match_result['composite_score'] >= 70 else ("#fbbf24" if match_result['composite_score'] >= 40 else "#f87171")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Job Match Score</div>
                <div class="metric-value" style="color: {score_color};">{match_result['composite_score']}%</div>
                <div class="metric-sub">{match_result['grade']}</div>
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
                <div class="metric-label">Recruiter Recommendation</div>
                <div class="metric-value" style="color: {rec_color}; font-size: 1.2rem;">{decision}</div>
                <div class="metric-sub">Role Fit + Profile Quality</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Model Confidence</div>
                <div class="metric-value">{confidence:.1f}%</div>
                <div class="metric-sub">Class Probability</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Overall Fit Breakdown")
        col_bar, col_legend = st.columns([3, 2])
        
        with col_bar:
            st.write(f"**Composite Job Match Score:** {match_result['composite_score']}%")
            st.progress(int(min(max(match_result["composite_score"], 0), 100)))
            st.caption("Weighted score combining 60% exact/synonym skill coverage + 40% TF-IDF semantic relevance.")
            
            # Show recommendation rationale alert
            if decision_cat == "Reject":
                st.error(f"**Decision Rationale:** {decision_reason}")
            elif decision_cat == "Review":
                st.warning(f"**Decision Rationale:** {decision_reason}")
            else:
                st.success(f"**Decision Rationale:** {decision_reason}")
            
        with col_legend:
            st.write(f"â€¢ **Skill Coverage:** {match_result['skill_score']}% ({match_result['total_matched_skills']}/{match_result['total_required_skills']} skills)")
            st.write(f"â€¢ **Semantic Context Relevance:** {match_result['semantic_score']}%")
            st.write(f"â€¢ **Candidate Experience:** {adj_exp} year(s)")
            st.write(f"â€¢ **General Profile ML Score:** {raw_decision} ({raw_confidence:.1f}%)")

    # ------------------ TAB 2: SKILL ANALYSIS ------------------
    with tab2:
        st.subheader("Skill Coverage & Gap Analysis")
        
        col_s1, col_s2, col_s3 = st.columns(3)

        with col_s1:
            st.markdown(f"#### ðŸŸ¢ Matched Skills ({len(match_result['matched_skills'])})")
            if match_result["matched_skills"]:
                badges_html = "".join([f'<span class="badge-matched">{s}</span>' for s in match_result["matched_skills"]])
                st.markdown(f'<div class="badge-container">{badges_html}</div>', unsafe_allow_html=True)
            else:
                st.write("No direct skill matches found.")

        with col_s2:
            st.markdown(f"#### ðŸ”´ Missing Skills ({len(match_result['missing_skills'])})")
            if match_result["missing_skills"]:
                badges_html = "".join([f'<span class="badge-missing">{s}</span>' for s in match_result["missing_skills"]])
                st.markdown(f'<div class="badge-container">{badges_html}</div>', unsafe_allow_html=True)
            else:
                st.write("No required skills missing from candidate resume!")

        with col_s3:
            st.markdown(f"#### ðŸ”µ Additional Skills ({len(match_result['extra_skills'])})")
            if match_result["extra_skills"]:
                badges_html = "".join([f'<span class="badge-extra">{s}</span>' for s in match_result["extra_skills"]])
                st.markdown(f'<div class="badge-container">{badges_html}</div>', unsafe_allow_html=True)
            else:
                st.write("No additional technical skills recorded.")

    # ------------------ TAB 3: EXPLAINABILITY ------------------
    with tab3:
        st.subheader("ðŸ’¡ Explainable AI Decision Factors")
        st.write("Transparent breakdown of the factors influencing the model's hiring recommendation:")

        # Summary bullet factors
        for reason in explanation["summary_reasons"]:
            st.markdown(f"â€¢ {reason}")

        if explanation["feature_contributions"]:
            st.markdown("#### Top Model Feature Influences")
            feat_df = pd.DataFrame(explanation["feature_contributions"])
            st.dataframe(
                feat_df,
                column_config={
                    "feature": "Evaluated Feature",
                    "impact": "Model Impact Score",
                    "type": "Contribution Direction"
                },
                use_container_width=True,
                hide_index=True
            )

    # ------------------ TAB 4: PARSED DETAILS & RAW TEXT ------------------
    with tab4:
        st.subheader("Extracted Candidate Metadata")
        meta_df = pd.DataFrame([{
            "Attribute": "File Name", "Extracted Value": uploaded_file.name
        }, {
            "Attribute": "Detected Skills Count", "Extracted Value": len(profile["skills"])
        }, {
            "Attribute": "Estimated Experience", "Extracted Value": f"{adj_exp} Years"
        }, {
            "Attribute": "Highest Education", "Extracted Value": adj_edu
        }, {
            "Attribute": "Recorded Certification", "Extracted Value": adj_cert
        }, {
            "Attribute": "Estimated Projects", "Extracted Value": f"{adj_projects} Projects"
        }])
        st.table(meta_df)

        with st.expander("ðŸ“„ View Raw Extracted Resume Text"):
            st.text_area("Resume Text Content", value=resume_text, height=350)

    # Compliance Disclaimer
    st.markdown("---")
    st.caption(
        "âš–ï¸ **Ethical AI Disclaimer:** This system provides automated decision-support metrics and candidate "
        "relevance rankings. Final hiring decisions must be reviewed by authorized human recruiters in accordance with employment regulations."
    )

else:
    # Empty State Guidance
    st.info("ðŸ‘† Please upload a PDF resume in the left panel to begin automated screening and fit evaluation.")

