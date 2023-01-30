import random
import math
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram
from sympy import Rational
from sympy.ntheory.continued_fraction import continued_fraction, continued_fraction_convergents

from qft import qft
from elementary import ax_modM
from order_finding import order_finding


def shor(N: int, show_hist: bool = True):
    """Shor's factoring algorithm: given $N\in\mathbb{Z}$, it finds a prime factor of $N$.

    Args:
        N (int): $N$

    Returns:
        A prime factor of $N$
    """

    if N % 2 == 0:
        return 2

    for _ in range(N):
        a = random.randint(2, N - 1)
        gcd = math.gcd(a, N)
        if gcd != 1:
            return gcd

        r = order_finding(x=a, N=N, show_hist=show_hist)

        if r % 2 == 1:
            continue
        if (x ** (r // 2) + 1) % N == 0:
            continue

        factor_candidate = math.gcd(x ** (r // 2) - 1, N)
        if N % factor_candidate == 0:
            return factor_candidate
        factor_candidate = math.gcd(x ** (r // 2) + 1, N)
        if N % factor_candidate == 0:
            return factor_candidate

    raise Exception('A prime factor is Not found!')


if __name__ == '__main__':
    print(shor(N=9))
