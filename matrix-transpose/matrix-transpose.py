import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    mat = np.asarray(A)
    transposed_mat = np.moveaxis(mat,range(mat.ndim),range(mat.ndim)[::-1])
    return transposed_mat