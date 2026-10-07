import numpy as np

def adagrad_step(w: list, g: list, G: list, lr: float = 0.01, eps: float = 1e-8) -> dict:
    """
    Returns a dictionary with new_w and new_G.
    """
    # Write code here
    w = np.asarray(w, dtype = float)
    g = np.asarray(g, dtype = float)
    G = np.asarray(G, dtype = float)
    
    if w.shape != g.shape:
        raise ValueError("number of gradients are not equal to the number of parameters")
        
    if w.shape != G.shape:
        raise ValueError('moments are not correctly initialised')

    # calculate G_new
    new_G = G + (g**2)
    # weight update 
    new_w = w - (lr*g)/(np.sqrt(new_G+eps))
    return {"new_w": new_w, "new_G": new_G}