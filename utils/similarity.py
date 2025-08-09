from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Normalize scores to [0, 1] for display
def _minmax_normalize(x: np.ndarray) -> np.ndarray:
    x_min = float(x.min())
    x_max = float(x.max())
    spread = x_max - x_min

    # If spread is tiny: return 0s for everyone
    if spread <= 1e-12:
        return np.zeros_like(x)
    
    return (x - x_min) / spread

# Compute similarity between job description and resumes, return the top n
def get_top_matches(
    job_vec: np.ndarray,
    resume_vecs: np.ndarray,
    resumes: List[Dict],
    top_n: int = 5
) -> List[Dict]:

    # Calculate cosine similarity
    sims = cosine_similarity(job_vec.reshape(1, -1), resume_vecs)[0]
    norm = _minmax_normalize(sims) # Normalize for UI readability

    # Sorts, reverses, then slices array
    idxs = np.argsort(sims)[::-1][:top_n]

    # Return normalized score and raw score
    return [
        {"id": resumes[i]["id"], "score": float(norm[i]), "raw": round(float(sims[i]), 6)}
        for i in idxs
    ]