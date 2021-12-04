from typing import Optional
import math

import numpy as np
import matplotlib.pyplot as plt
from sympy import gcdex
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram

from qft import qft
from elementary import ax_modM
from shor import shor

def discrete_log(alpha: int, beta: int, p: int, N_len: Optional[int] = None, show_hist: Optional[bool] = True) -> int:
    # find q s.t. alpha^q = 1
    q = shor(M=p, a=alpha, use_only_period=True, show_hist=False)
    print(f'q is {q} ({alpha}^{q} = 1)')

    # find d s.t. alpha^d = beta
    if N_len is None:
        N_len = int(np.ceil(np.log2(p ** 2)))
    N = 2 ** N_len
    qc = QuantumCircuit(11 * N_len - 2, N_len * 2)

    qc.h(range(N_len * 2))

    qc.append(ax_modM(a=alpha, M=p, N_len=N_len), list(range(N_len)) + list(range(N_len * 2, 11 * N_len - 2)))
    qc.append(ax_modM(a=beta, M=p, N_len=N_len, x_0_at_first=False), range(N_len, 11 * N_len - 2))

    qc.append(qft(n=N_len), range(N_len))
    qc.append(qft(n=N_len), range(N_len, N_len * 2))

    qc.measure(range(N_len * 2), range(N_len * 2))

    backend = Aer.get_backend('aer_simulator_matrix_product_state')
    qc = transpile(qc, backend)
    job = backend.run(qc, shots=10000)
    hist = job.result().get_counts()

    if show_hist:
        plot_histogram(hist)
        plt.show()

    for measured_key, _ in sorted(hist.items(), key=lambda x: x[1], reverse=True):
        x = int(measured_key[-N_len:], 2)
        y = int(measured_key[-N_len * 2:-N_len], 2)
        if math.gcd(x, N) > 1:
            print(f'x:{x},N:{N},gcd:{math.gcd(x,N)}')
            continue

        d_0, d_1, _ = gcdex(x, N)
        maybe_d = x * d_0 * y % p
        print(f'maybe_d: {maybe_d}, x: {x}, y: {y}')
        if alpha ** maybe_d % p == beta:
            return maybe_d

    print('d was not found?!')

if __name__ == '__main__':
    #print(discrete_log(alpha=3, beta=6, p=7, N_len=3))
    print(discrete_log(alpha=3, beta=6, p=7))
    #discrete_log(alpha=3, beta=1, p=4, N_len=2)
