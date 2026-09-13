import numpy as np

def relu(x) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    # Write code here
    ar = np.array(x)
    return np.asarray(np.maximum(ar,0)) # np converts scalar into 0-D array 