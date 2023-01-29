import math

def lcm(a, b):
    """Least common multiple: $\mathrm{lcm}(a,b)$

    Args:
        a (int): $a$
        b (int): $b$

    Returns:
        $\mathrm{lcm}(a,b)$
    """

    return a * b / math.gcd(a, b)
