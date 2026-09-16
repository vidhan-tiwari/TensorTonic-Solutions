import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    arr = np.asarray(A)
    A_t = arr.T
    return A_t