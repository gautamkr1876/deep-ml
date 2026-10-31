import numpy as np

def log_softmax(scores: list) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    max_score = np.max(scores)
    shifted_scores = scores - max_score
    return shifted_scores - np.log(np.sum(np.exp(shifted_scores)))