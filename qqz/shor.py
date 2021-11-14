import random
import math
from fractions import Fraction
from itertools import combinations
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram

from qft import qft
from elementary import ax_modN


def shor(N: int, a: Optional[int] = None, show_hist=True, use_only_period=False):
    if a is None:
        random.randint(2, N - 1)
    gcd = math.gcd(a, N)
    if gcd != 1:
        print(f'Answer is found in only classical computation: {gcd}')
        return gcd

    N_len = int(np.ceil(np.log2(N)))
    qc = QuantumCircuit(10 * N_len - 2, 2 * N_len)
    #qc = QuantumCircuit(10 * N_len - 2, 10 * N_len - 2)

    qc.h(range(N_len))
    #qc.x([0, 1])

    qc.append(ax_modN(a=a, N=N), range(10 * N_len - 2))

    qc.append(qft(n=N_len), range(N_len))

    qc.measure(range(2 * N_len), range(2 * N_len))
    #qc.measure(range(10 * N_len - 2), range(10 * N_len - 2))

    backend = Aer.get_backend('aer_simulator_matrix_product_state')#('aer_simulator')
    qc = transpile(qc, backend)
    job = backend.run(qc, shots=10000)
    hist = job.result().get_counts()

    if show_hist:
        plot_histogram(hist)
        plt.show()

    y_list = []
    for measured_key, _ in sorted(hist.items(), key=lambda x: x[1]):
        y = int(measured_key[-N_len:], 2)
        if y == 0:
            continue
        y_list.append(y)

        y_per_N = Fraction(y, N)
        maybe_r = y_per_N.denominator

        if use_only_period:
            if a ** maybe_r % N == 1:
                return maybe_r
            continue

        if maybe_r % 2 == 1 or (a ** (maybe_r // 2) + 1) % N == 0:
            continue
        gcd = math.gcd(a ** (maybe_r // 2) + 1, N)
        if gcd != 1:
            return gcd
        gcd = math.gcd(a ** (maybe_r // 2) - 1, N)
        if gcd != 1:
            return gcd
    print(y_list)
    for y1, y2 in combinations(y_list, 2):
        y1_per_y2 = Fraction(y1, y2)
        m1 = y1_per_y2.numerator
        m2 = y1_per_y2.denominator
        maybe_r12 = [N * m1 // y1, N * m2 // y2]
        for maybe_r in maybe_r12:
            if use_only_period:
                if a ** maybe_r % N == 1:
                    return maybe_r
                continue

            if maybe_r % 2 == 1 or (a ** (maybe_r // 2) + 1) % N == 0:
                continue
            gcd = math.gcd(a ** (maybe_r // 2) + 1, N)
            if gcd != 1:
                return gcd
            gcd = math.gcd(a ** (maybe_r // 2) - 1, N)
            if gcd != 1:
                return gcd

    print('gcd was not found?!')

if __name__ == '__main__':
    #print(shor(N=8, a=3))
    print(shor(N=7, a=3, use_only_period=True))
