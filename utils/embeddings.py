import os
from dotenv import load_dotenv
import openai
import numpy as np
from typing import List

# Load variables from .env into os.environ
load_dotenv()

oai_key = os.getenv("OPENAI_API_KEY")
if not oai_key:
    raise ValueError("OPENAI_API_KEY not set in environment or .env file")
openai.api_key = oai_key

def get_embeddings(texts: List[str]) -> np.ndarray:

    # Send all texts in one batch for efficiency
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=texts,
    )
    # Extract embeddings and convert to numpy array
    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings)

