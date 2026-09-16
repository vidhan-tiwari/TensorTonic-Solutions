import numpy as np

def manhattan_distance(x: list, y: list) -> float:
    """
    Returns the Manhattan distance as a Python float.
    """
    # Write code here
    x_np = np.asarray(x, dtype = float)
    y_np = np.asarray(y ,dtype = float)
    return float(np.sum(np.abs(x_np-y_np)))