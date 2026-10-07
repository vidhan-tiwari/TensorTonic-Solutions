import numpy as np

def nesterov_momentum_step(w: list, v: list, grad: list, lr: float = 0.01, momentum: float = 0.9) -> dict:
    """
    Returns a dictionary with new_w and new_v.
    """
    w = np.asarray(w, dtype = float)
    v = np.asarray(v, dtype = float)
    grad = np.asarray(grad, dtype = float)
    if w.shape != grad.shape:
        raise ValueError("number of gradients are not equal to the number of parameters")
        
    if w.shape != v.shape:
        raise ValueError('moments are not correctly initialised')

    # updating moments
    v_new = (momentum*v) + (lr*grad)
    w_new = w - v_new
    
    return {'new_w' : w_new, 'new_v': v_new}