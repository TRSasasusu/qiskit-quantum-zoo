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


def decode_bin(register_bin: str) -> int:
    """Decode binaries of a register.

    Args:
        register_bin (str): binaries of the register

    Returns:
        An integer represented by the given binaries

    Examples:

    ```
    >>> decode_bin('011')
    3
    ```
    """

    return int(register_bin[-len(register_bin):], 2)
