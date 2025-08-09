# Candidate Recommendation Engine

## Overview
This is a Streamlit app that takes a job description and one or more PDF resumes, and returns the top candidate matches ranked by semantic similarity. It also offers an optional AI-generated “fit summary” for each candidate.

## Approach
1. **Text Extraction**  
   - Extracts plain text from uploaded PDF resumes using `pdfplumber`.
   - Rejects non-PDFs to keep parsing consistent.

2. **Embedding Generation**  
   - Uses OpenAI’s `text-embedding-ada-002` model to convert both the job description and each resume into high-dimensional vectors.
   - All texts are sent in a batch for efficiency.

3. **Similarity & Ranking**  
   - Computes cosine similarity between the job vector and each resume vector.
   - Applies min–max normalization per query for UI-friendly scores out of 100.

4. **Summarization**   
   - On demand, generates a concise, evidence-based summary of the candidate’s fit using `gpt-4o-mini`.
   - Summaries are cached in the session to avoid repeated API calls.

## Assumptions
- Resumes are provided in **readable PDF format**.
- English is the primary language for both job descriptions and resumes.
- For this demo, accuracy is measured by semantic similarity. No additional business rules (e.g., location, salary) are applied.
- This is a minimal viable product to demonstrate the core concept, the code is not fully production ready and some features could be expanded or optimized.

## Access the App
The app is deployed on Streamlit Community Cloud: [**Launch Here**](https://candidate-recommendergit-mwxu8fikyunoqf7y5npmcz.streamlit.app/)
