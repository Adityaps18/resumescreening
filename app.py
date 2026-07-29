import re
import joblib
import pdfplumber
import pandas as pd
import streamlit as st

# Load trained models
category_model = joblib.load("category_model.pkl")
screening_model = joblib.load("screening_model.pkl")

SKILLS = [
    "python", "java", "sql", "machine learning", "deep learning",
    "tensorflow", "pytorch", "pandas", "numpy", "excel", "tableau",
    "power bi", "react", "node.js", "docker", "aws", "html", "css",
    "cybersecurity", "linux", "networking", "flask"
]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d{10,}", " ", text)
    text = re.sub(r"[^a-zA-Z+#.\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_text_from_pdf(uploaded_file):
    text = ""

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    return text

def extract_skills(text):
    text = text.lower()
    return [skill for skill in SKILLS if skill in text]

def match_resume_to_job(resume_text, job_description):
    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_description))

    matched = resume_skills.intersection(job_skills)
    missing = job_skills - resume_skills

    score = 0 if not job_skills else (len(matched) / len(job_skills)) * 100

    return {
        "score": round(score, 2),
        "matched_skills": sorted(list(matched)),
        "missing_skills": sorted(list(missing))
    }

st.set_page_config(page_title="AI Resume Screener", page_icon="📄")
st.title("📄 AI Resume Screening System")
st.write("Upload a candidate resume PDF and compare it with a job description.")

uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste Job Description",
    height=180,
    placeholder="Example: Need Python, SQL, Machine Learning and Tableau..."
)

st.subheader("Candidate Details")

col1, col2 = st.columns(2)

with col1:
    experience = st.number_input(
        "Experience (Years)",
        min_value=0,
        max_value=50,
        value=1
    )

    education = st.selectbox(
        "Education",
        ["B.Sc", "B.Tech", "M.Tech", "MBA", "PhD"]
    )

    projects = st.number_input(
        "Projects Count",
        min_value=0,
        max_value=100,
        value=1
    )

with col2:
    certification = st.selectbox(
        "Certification",
        ["None", "AWS Certified", "Google ML", "Deep Learning Specialization"]
    )

    job_role = st.selectbox(
        "Job Role",
        ["Data Scientist", "AI Researcher", "Software Engineer", "Cybersecurity Analyst"]
    )

if st.button("Screen Resume"):

    if uploaded_file is None:
        st.warning("Please upload a PDF resume first.")

    else:
        # Read text from uploaded PDF
        resume_text = extract_text_from_pdf(uploaded_file)

        if not resume_text.strip():
            st.error(
                "No readable text was found in this PDF. "
                "Please upload a text-based PDF, not a scanned image PDF."
            )

        else:
            cleaned_resume = clean_text(resume_text)

            # NLP + ML: Predict job category
            predicted_category = category_model.predict([cleaned_resume])[0]

            # NLP: Find skills and match job description
            found_skills = extract_skills(resume_text)
            match_result = match_resume_to_job(resume_text, job_description)

            # Create data for Hire / Reject model
            candidate_data = pd.DataFrame([{
                "Skills": ", ".join(found_skills),
                "Experience (Years)": experience,
                "Education": education,
                "Certifications": certification,
                "Job Role": job_role,
                "Projects Count": projects
            }])

            # ML: Predict Hire / Reject
            decision = screening_model.predict(candidate_data)[0]

            probabilities = screening_model.predict_proba(candidate_data)[0]
            classes = screening_model.named_steps["classifier"].classes_
            probability_map = dict(zip(classes, probabilities))
            confidence = probability_map[decision] * 100

            st.divider()
            st.subheader("Screening Result")

            st.write(f"**Uploaded Resume:** {uploaded_file.name}")
            st.write(f"**Predicted Category:** {predicted_category}")
            st.write(f"**Recruiter Recommendation:** {decision}")
            st.write(f"**Model Confidence:** {confidence:.2f}%")

            st.subheader("Extracted Skills")
            st.write(", ".join(found_skills) if found_skills else "No skills found.")

            if job_description.strip():
                st.subheader("Job Match")

                st.progress(int(match_result["score"]))
                st.write(f"**Job Match Score:** {match_result['score']}%")

                st.write(
                    f"**Matched Skills:** "
                    f"{', '.join(match_result['matched_skills']) or 'None'}"
                )

                st.write(
                    f"**Missing Skills:** "
                    f"{', '.join(match_result['missing_skills']) or 'None'}"
                )

            with st.expander("View extracted resume text"):
                st.write(resume_text)

            st.warning(
                "This is an AI recommendation only. "
                "A human recruiter must make the final hiring decision."
            )