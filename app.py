import streamlit as st
from utils.text_extraction import extract_text
from utils.embeddings import get_embeddings
from utils.similarity import get_top_matches
from utils.summarize import summarize_fit

# --- Page config ---
st.set_page_config(page_title="Candidate Recommender", page_icon="🕵️‍♂️", layout="wide")

# --- Title & instructions ---
st.title("🔍 Candidate Recommendation Engine")
st.markdown(
    """
    1. Paste your **job description** below.  
    2. Upload one or more **resumes** as files (PDF only).  
    3. Hit **Run** to see your top matches!
    """
)

# --- Input area: job description ---
job_description = st.text_area(
    label="Job Description",
    help="Describe the role: responsibilities, skills, keywords, etc.",
    height=200,
)

# --- File uploader (PDF only) ---
uploaded_files = st.file_uploader(
    label="Upload Candidate Resumes (PDF only)",
    type=["pdf"],
    accept_multiple_files=True,
)

# --- How many candidates to show ---
top_n = st.number_input("Number of top candidates", min_value=1, value=5, step=1)

# --- Session state (keep results on screen across reruns) ---
if "ran" not in st.session_state:
    st.session_state.ran = False
if "summaries" not in st.session_state:
    st.session_state.summaries = {}  # {resume_id: summary_text}

# --- Run button (flips state so results persist) ---
if st.button("🚀 Run Recommendation"):
    st.session_state.ran = True
# 
# --- Main pipeline (validate -> ingest -> embed -> rank -> render) ---
if st.session_state.ran:
    
    # Validate: require JD and at least one uploaded PDF
    if not job_description.strip():
        st.error("Please enter a job description!")
    elif not uploaded_files:
        st.error("Please upload at least one PDF resume.")
        
    else:
        # Ingest resumes as [{id, text}]
        resumes = []
        for f in uploaded_files:
            try:
                txt = extract_text(f)  # PDF -> text
                if txt.strip():
                    resumes.append({"id": f.name, "text": txt})
                else:
                    st.warning(f"No text found in {f.name}.")
            except ValueError as e:
                st.warning(str(e))

        # Stop if nothing usable
        if not resumes:
            st.error("No valid resumes to process—please upload at least one readable PDF.")
        else:
            # Generate embeddings (JD first so embs[0] is the job vector)
            with st.spinner("Generating embeddings…"):
                texts = [job_description] + [r["text"] for r in resumes]
                embs = get_embeddings(texts)
                job_emb, resume_embs = embs[0], embs[1:]

            # Rank by cosine; also returns normalized score for UI
            top_n_eff = max(1, min(int(top_n), len(resumes)))  # clamp to available resumes
            matches = get_top_matches(job_emb, resume_embs, resumes, top_n=top_n_eff)

            # Display matches with Summarize buttons
            st.markdown("## Top Matches")
            resume_text_by_id = {r["id"]: r["text"] for r in resumes}  # O(1) lookup for summaries

            for rank, m in enumerate(matches, start=1):
                cols = st.columns([6, 2, 2, 2])  # name | score | raw | action
                with cols[0]:
                    st.markdown(f"**{rank}. {m['id']}**")
                with cols[1]:
                    st.markdown(f"**Score:** {round(m['score'] * 100, 1)}")
                with cols[2]:
                    st.caption(f"Raw: {m['raw']}")
                with cols[3]:
                    if st.button("Summarize", key=f"sum_{rank}_{m['id']}"):
                        try:
                            with st.spinner("Generating summary…"):
                                summary = summarize_fit(job_description, resume_text_by_id[m["id"]])
                            st.session_state.summaries[m["id"]] = summary  # cache to avoid recharges
                        except Exception as e:
                            st.error(f"Summary failed: {e}")

                # Show cached summary 
                if m["id"] in st.session_state.summaries:
                    with st.expander("Summary", expanded=True):
                        st.markdown(st.session_state.summaries[m["id"]])

