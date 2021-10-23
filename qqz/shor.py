import random
import math
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram

from qft import qft
from elementary import ax_modN


def shor(N: int, a: Optional[int] = None):
    if a is None:
        random.randint(2, N - 1)
    gcd = math.gcd(a, N)
    if gcd != 1:
        print(f'Answer is found in only classical computation: {gcd}')
        return

    N_len = int(np.ceil(np.log2(N)))
    qc = QuantumCircuit(10 * N_len - 2, 2 * N_len)

    qc.h(range(N_len))

    qc.append(ax_modN(a=a, N=N), range(10 * N_len - 2))

    qc.append(qft(n=N_len), range(N_len))

    qc.measure(range(2 * N_len), range(2 * N_len))

    backend = Aer.get_backend('aer_simulator_matrix_product_state')#('aer_simulator')
    qc = transpile(qc, backend)
    job = backend.run(qc, shots=1000)
    hist = job.result().get_counts()
    plot_histogram(hist)
    plt.show()


if __name__ == '__main__':
    shor(N=8, a=3)
