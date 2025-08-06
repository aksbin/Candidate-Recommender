import streamlit as st

# --- Page config ---
st.set_page_config(
    page_title="Candidate Recommender",
    page_icon="🕵️‍♂️",
    layout="wide"
)

# --- Title & instructions ---
st.title("🔍 Candidate Recommendation Engine")
st.markdown(
    """
    1. Paste your **job description** below.  
    2. Upload one or more **resumes** as files (PDF / DOCX / TXT).  
    3. (optional) If you don’t have files, paste resumes into the text boxes.  
    4. Hit **Run** to see your top matches!
    """
)

# --- Input area: job description ---
job_description = st.text_area(
    label="Job Description",
    help="Describe the role: responsibilities, skills, keywords, etc.",
    height=200,
)

# --- File uploader for resumes ---
uploaded_files = st.file_uploader(
    label="Upload Candidate Resumes",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

# --- Fallback: allow pasting resume text if no files ---
if not uploaded_files:
    st.markdown("**Or**, paste resume text here:")
    resume_texts = []
    for i in range(3):  # you can start with e.g. 3 slots
        txt = st.text_area(f"Resume #{i+1}", height=150, key=f"resume_text_{i}")
        resume_texts.append(txt)
else:
    resume_texts = []

# --- Run button ---
run = st.button("🚀 Run Recommendation")

if run:
    if not job_description.strip():
        st.error("Please enter a job description!")
    elif not (uploaded_files or any(r.strip() for r in resume_texts)):
        st.error("Please upload at least one resume or paste some text.")
    else:
        # Here’s where you’ll hook in your embeddings → cosine similarity → display
        st.success("Inputs look good—processing…")
        # (Placeholder for the next step)
