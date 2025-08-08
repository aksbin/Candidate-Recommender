from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def _minmax_normalize(x: np.ndarray) -> np.ndarray:
    """Per-query min–max to [0, 1]. If all values are equal, return zeros (no separation)."""
    x_min = float(x.min())
    x_max = float(x.max())
    denom = x_max - x_min
    if denom <= 1e-12:
        return np.zeros_like(x)
    return (x - x_min) / denom

def get_top_matches(
    job_vec: np.ndarray,
    resume_vecs: np.ndarray,
    resumes: List[Dict],
    top_n: int = 5
) -> List[Dict]:
    """
    Compute cosine similarities between `job_vec` and each row in `resume_vecs`,
    then return the top_n resumes sorted by descending score.
    
    Returns a list of dicts: [{ "id": ..., "score": ... }, …]
    """

    # sims: 1D array of shape (len(resume_vecs),)
    sims = cosine_similarity(job_vec.reshape(1, -1), resume_vecs, dense_output=True)[0]
    norm = _minmax_normalize(sims)

    # argsort descending
    idxs = np.argsort(sims)[::-1][:top_n]
    return [
        {"id": resumes[i]["id"], "score": float(norm[i]), "raw": round(float(sims[i]), 6)}
        for i in idxs
    ]