from typing import Optional

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.visualization import plot_histogram
from qiskit.circuit import Gate


def grover(N_len: int, oracle_gate: Gate, oracle_qubits_len: int, k_time: int, show_hist: Optional[bool] = True) -> int:
    qc = QuantumCircuit(N_len + oracle_qubits_len, N_len)

    qc.h(range(N_len))

    for _ in range(k_time):
        qc.append(oracle_gate, range(N_len + oracle_qubits_len))

        qc.h(range(N_len))
        qc.x(range(N_len))

        # MCZ
        qc.h(N_len - 1)
        qc.mct(list(range(N_len - 1)), N_len - 1)
        qc.h(N_len - 1)

        qc.x(range(N_len))
        qc.h(range(N_len))

    qc.measure(range(N_len), range(N_len))

    backend = Aer.get_backend('aer_simulator_matrix_product_state')
    qc = transpile(qc, backend)
    job = backend.run(qc, shots=10000)
    hist = job.result().get_counts()

    if show_hist:
        plot_histogram(hist)
        plt.show()

    return int(max(hist.items(), key=lambda x: x[1])[0], 2)


def sample_oracle(N_len: int) -> Gate:
    """
    An oracle which inverts the sign if the number of 11 is odd.
    11 が奇数個あるなら符号を反転するオラクル
    """

    qc = QuantumCircuit(N_len + 1)

    for qubit in range(N_len - 1):
        qc.ccx(qubit, qubit + 1, N_len)

    qc.z(N_len)

    for qubit in reversed(range(N_len - 1)):
        qc.ccx(qubit, qubit + 1, N_len)

    return qc.to_gate()


if __name__ == '__main__':
    print(grover(3, sample_oracle(3), 1, 1))
