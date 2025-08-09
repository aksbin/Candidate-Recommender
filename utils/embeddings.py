import os
from dotenv import load_dotenv
import openai
import numpy as np
from typing import List

# Load variables from .env 
load_dotenv()


# Read API key and fail fast if missing
oai_key = os.getenv("OPENAI_API_KEY")
if not oai_key:
    raise ValueError("OPENAI_API_KEY not set in environment or .env file")


# Configure OpenAI client
openai.api_key = oai_key

# Return embeddings for a list of texts as a NumPy array.
def get_embeddings(texts: List[str]) -> np.ndarray:

    # Batch request for efficiency
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=texts,
    )

    # Extract embeddings from response
    embeddings = [item.embedding for item in response.data]

    return np.array(embeddings)

