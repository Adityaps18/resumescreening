# AI Resume Screening System

A Streamlit application that screens PDF resumes using trained machine-learning models. It predicts a candidate's job category, recommends a hiring decision, extracts known skills, and compares those skills with a supplied job description.

## Features

- Upload and read text-based PDF resumes
- Predict the candidate's job category
- Produce a Hire/Reject recommendation with confidence
- Extract common technical skills from the resume
- Calculate matched and missing skills against a job description

## Run locally

1. Create and activate a virtual environment.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the application:

   ```bash
   streamlit run app.py
   ```

The trained model files (`category_model.pkl` and `screening_model.pkl`) must remain in the project root.

## Note

This project provides an AI-assisted recommendation only. A human recruiter should make the final hiring decision.
