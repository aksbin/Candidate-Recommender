import streamlit as st
import pdfplumber
from utils.text_extraction import extract_text
from utils.embeddings import get_embeddings
from utils.similarity import get_top_matches

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
    2. Upload one or more **resumes** as files (PDF only).  
    3. (optional) If you don't have files, paste resumes into the text boxes.  
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

# Number input to choose how many top candidates to show
# top_n = st.slider("Show Top N Candidates", min_value=5, max_value=10, value=5)

top_n = st.number_input(
                "Number of top candidates",
                min_value=1,
                value=5,
                step=1
            )
    
# --- Run button ---
run = st.button("🚀 Run Recommendation")

if run:
    if not job_description.strip():
        st.error("Please enter a job description!")
    elif not (uploaded_files or any(r.strip() for r in resume_texts)):
        st.error("Please upload at least one resume or paste some text.")
    else:

        # Here’s where you’ll hook in your embeddings → cosine similarity → display
        # st.success("Inputs look good—processing…")
        # (Placeholder for the next step)

        resumes = []

        # 1) Process uploaded PDF Files
        for f in uploaded_files:
            try:
                txt = extract_text(f)
                if txt.strip():
                    resumes.append({
                        "id": f.name,
                        "text": txt
                    })
                else:
                    st.warning(f"No text founf in {f.name}.")
            except ValueError as e:
                st.warning(str(e))

        # 2) Handles pasted text slots
        for i, txt in enumerate(resume_texts):
            if txt and txt.strip():
                resumes.append({
                    "id": f"pasted_{i+1}",
                    "text": txt
                })

        

        if not resumes:
            st.error("No valid resumes to process—please upload at least one PDF.")
        else:

            with st.spinner("Generating embeddings..."):
                texts = [job_description] + [r["text"] for r in resumes]
                embs = get_embeddings(texts)
                job_emb, resume_embs = embs[0], embs[1:]


            # Compute top matches
            matches = get_top_matches(job_emb, resume_embs, resumes, top_n=top_n)

            # Display results
            st.markdown('### 🔝 Top Matches')
            st.table(matches)
