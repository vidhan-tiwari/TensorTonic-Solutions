import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    # Write code here
    # convert every input into np array of float
    param = np.asarray(param, dtype = float)
    grad = np.asarray(grad, dtype = float)
    m = np.asarray(m, dtype = float)
    v = np.asarray(v, dtype = float)
    # checking if number of parameters are equal to number of gradients 
    if param.shape != grad.shape:
        raise ValueError("number of gradients are not equal to the number of parameters")
        
    if param.shape != m.shape or param.shape != v.shape:
        raise ValueError('moments are not correctly initialised')
        
    # calculate first moment
    m_new = beta1*m + (1.0-beta1)*grad
    # calculate second moment
    v_new = beta2*v + (1.0- beta2)*(grad**2)

    # bias correction of both the momentums
    # assuming t is not zero
    m_hat = m_new/(1.0-beta1**t); v_hat = v_new/(1.0-beta2**t)
    # parameter update 
    param_new = param - lr*(m_hat/(np.sqrt(v_hat) + eps))
                             
    return (param_new, m_new, v_new)
    