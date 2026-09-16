def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    # Write code here
    def gradient(x:float) -> float :
        return 2*a*x + b
    for _ in range(steps):
        x0 = x0 - (lr *gradient(x0))
    return x0