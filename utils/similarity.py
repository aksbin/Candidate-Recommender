from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
    sims = cosine_similarity(job_vec.reshape(1, -1), resume_vecs)[0]

    # argsort descending
    idxs = np.argsort(sims)[::-1][:top_n]
    return [{"id":resumes[i]["id"], "score": float(sims[i])} for i in idxs]