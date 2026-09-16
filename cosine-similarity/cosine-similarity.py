import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    # Write code here
    a_np = np.asarray(a, dtype = float)
    b_np = np.asarray(b, dtype = float)
    # check whether these two are vectors of same length or not 
    if a_np.ndim != 1 or b_np.ndim != 1 or a_np.shape[0] != b_np.shape[0]:
        raise ValueError("Vectors not compatible for Dot product")
        
    a_np_norm = np.sqrt(a_np@a_np)
    b_np_norm = np.sqrt(b_np@b_np)

    if a_np_norm == 0.0 or b_np_norm == 0.0:
        return 0.0
    dot_prod = a_np @ b_np
    
    return float(dot_prod/(a_np_norm*b_np_norm))