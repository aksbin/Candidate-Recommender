import os
from typing import Optional
from dotenv import load_dotenv
import openai

# Load environment variables (OPENAI_API_KEY)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set in environment or .env file")
openai.api_key = api_key


def summarize_fit(job_description: str, resume_text: str, model: str = "gpt-4o-mini", max_chars: int = 6000) -> str:
    """
    Generate a concise, evidence-based summary of why the candidate fits the role.
    - Clips resume text to avoid excessive token usage.
    - Uses a low temperature for consistent, factual summaries.
    """
    resume_clip = resume_text[:max_chars]

    messages = [
        {
            "role": "system",
            "content": (
                "You are an experienced technical recruiter. Write concise, evidence-based fit summaries. "
                "Use 4-6 bullet points referencing concrete skills/experience from the resume that match the job. "
                "If there are gaps, include one short bullet on risks or missing skills. End with a one-sentence verdict starting with 'Verdict:'."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Job description:\n{job_description}\n\n"
                f"Candidate resume:\n{resume_clip}\n\n"
                "Write the summary now."
            ),
        },
    ]

    resp = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=250,
    )
    return resp.choices[0].message.content.strip()
