import random
import math
from fractions import Fraction
from itertools import combinations
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram
from sympy import Rational
from sympy.ntheory.continued_fraction import continued_fraction, continued_fraction_convergents

from qft import qft
from elementary import ax_modM


def shor(M: int, a: Optional[int] = None, show_hist=True, use_only_period=False):
    if a is None:
        random.randint(2, M - 1)
    gcd = math.gcd(a, M)
    if gcd != 1:
        print(f'Answer is found in only classical computation: {gcd}')
        return gcd

    N_len = int(np.ceil(np.log2(M ** 2)))
    N = 2 ** N_len
    #qc = QuantumCircuit(10 * N_len - 2, 2 * N_len)
    qc = QuantumCircuit(10 * N_len - 2, N_len)
    #qc = QuantumCircuit(10 * N_len - 2, 10 * N_len - 2)

    qc.h(range(N_len))
    #qc.x([0, 1])

    qc.append(ax_modM(a=a, M=M, N_len=N_len), range(10 * N_len - 2))

    qc.append(qft(n=N_len), range(N_len))

    #qc.measure(range(2 * N_len), range(2 * N_len))
    qc.measure(range(N_len), range(N_len))
    #qc.measure(range(10 * N_len - 2), range(10 * N_len - 2))

    backend = Aer.get_backend('aer_simulator_matrix_product_state')#('aer_simulator')
    qc = transpile(qc, backend)
    job = backend.run(qc, shots=10000)
    hist = job.result().get_counts()

    if show_hist:
        plot_histogram(hist)
        plt.show()

    print(sorted(hist.items(), key=lambda x: x[1], reverse=True))
    y_list = []
    for measured_key, _ in sorted(hist.items(), key=lambda x: x[1], reverse=True):
        y = int(measured_key[-N_len:], 2)
        if y == 0:
            continue

        for fraction in continued_fraction_convergents(continued_fraction(Rational(y, N))):
            maybe_r = fraction.denominator
            if maybe_r != 1 and maybe_r < M and a ** maybe_r % M == 1:
                if use_only_period:
                    return maybe_r

                if maybe_r % 2 == 1 or (a ** (maybe_r // 2) + 1) % M == 0:
                    continue
                gcd = math.gcd(a ** (maybe_r // 2) + 1, M)
                if gcd != 1:
                    return gcd
                gcd = math.gcd(a ** (maybe_r // 2) - 1, M)
                if gcd != 1:
                    return gcd

    print('gcd was not found?!')

if __name__ == '__main__':
    #print(shor(M=8, a=3))
    print(shor(M=57, a=5))
    #print(shor(M=7, a=3, use_only_period=True))
