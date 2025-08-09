import os
from typing import Optional
from dotenv import load_dotenv
import openai

# Load environment variables 
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set in environment or .env file")

# Configure OpenAI client
openai.api_key = api_key

# Generate a concise, evidence based fit summary
def summarize_fit(job_description: str, resume_text: str, model: str = "gpt-4o-mini", max_chars: int = 40000) -> str:

    # Clip resume to control token usage
    resume_clip = resume_text[:max_chars]

    # Build chat messages 
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

    # Call chat completions with low temperature for consistent, factual output
    resp = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=250,
    )

    # Return trimmed text content
    return resp.choices[0].message.content.strip()
