import unittest

import numpy as np
from sympy.physics.quantum.qubit import matrix_to_qubit
from qiskit import QuantumCircuit, Aer, transpile

from qqz.elementary import (
        adder,
        adder_modN,
        ctrl_multi_modN,
        ax_modN,
        )


def run_qc_and_get_ket_vector(qc):
    backend = Aer.get_backend('aer_simulator_statevector')
    qc = transpile(qc, backend)
    qc.save_statevector()

    job = backend.run(qc)
    statevector = job.result().get_statevector(qc)
    return matrix_to_qubit(np.array(statevector)[:, np.newaxis])


class TestElementary(unittest.TestCase):
    def test_adder(self):
        qc = QuantumCircuit(2 + (2 + 1) + 2)
        qc.x([0, 1])
        qc.x(2)

        qc.append(adder(2), range(2 + (2 + 1) + 2))

        ket_vector = run_qc_and_get_ket_vector(qc)
        self.assertEqual(str(ket_vector), '1.0*|0010011>')

    def test_adder_modN(self):
        qc = QuantumCircuit(2 + (2 + 1) + 2 + 2 + 1)
        qc.x([0, 1])
        qc.x(2)

        qc.append(adder_modN(3), range(2 + (2 + 1) + 2 + 2 + 1))

        ket_vector = run_qc_and_get_ket_vector(qc)
        self.assertEqual(str(ket_vector), '1.0*|0000000111>')

    def test_ctrl_multi_modN(self):
        qc = QuantumCircuit(9 * 2 - 1)
        qc.h(0)
        qc.x(2)

        qc.append(ctrl_multi_modN(a=2, N=3), range(9 * 2 - 1))

        ket_vector = run_qc_and_get_ket_vector(qc)
        self.assertEqual(str(ket_vector), '0.707106781186547*|00000000000001101> + 0.707106781186548*|00000000000010100>')

    def test_ax_modN(self):
        qc = QuantumCircuit(10 * 2 - 2)
        qc.h([0, 1])

        qc.append(ax_modN(a=2, N=3), range(10 * 2 - 2))

        ket_vector = run_qc_and_get_ket_vector(qc)
        self.assertEqual(str(ket_vector), '0.5*|000000000000000100> + 0.5*|000000000000000110> + 0.5*|000000000000001001> + 0.5*|000000000000001011>')


if __name__ == '__main__':
    unittest.main()
