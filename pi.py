import math
import time

def pi_leibniz(n_terms):
    """Approximate pi using the Leibniz series."""
    s = 0.0
    for k in range(n_terms):
        s += (-1)**k / (2*k + 1)
    return 4 * s

def pi_nilakantha(n_terms):
    """Approximate pi using the Nilakantha series."""
    s = 3.0
    sign = 1
    for k in range(1, n_terms+1):
        term = 4.0 / ( (2*k)*(2*k+1)*(2*k+2) )
        s += sign * term
        sign *= -1
    return s

def pi_ramanujan(n_terms):
    """Approximate pi using a Ramanujan-style series."""
    # Constant factor
    factor = 2 * math.sqrt(2) / 9801
    s = 0.0
    for k in range(n_terms):
        num = math.factorial(4*k) * (1103 + 26390*k)
        den = (math.factorial(k)**4) * (396**(4*k))
        s += num / den
    return 1 / (factor * s)

if __name__ == "__main__":
    TERMS = 100000  # for Leibniz & Nilakantha you may need many terms
    R_TERMS = 3     # Ramanujan needs far fewer terms
    
    start = time.time()
    pi1 = pi_leibniz(TERMS)
    print(f"Leibniz ({TERMS} terms): {pi1:.15f}  err = {abs(pi1-math.pi):.2e}")
    print(" time:", time.time()-start)

    start = time.time()
    pi2 = pi_nilakantha(TERMS)
    print(f"Nilakantha ({TERMS} terms): {pi2:.15f}  err = {abs(pi2-math.pi):.2e}")
    print(" time:", time.time()-start)

    start = time.time()
    pi3 = pi_ramanujan(R_TERMS)
    print(f"Ramanujan ({R_TERMS} terms): {pi3:.15f}  err = {abs(pi3-math.pi):.2e}")
    print(" time:", time.time()-start)