from typing import Optional
import math

import numpy as np
import matplotlib.pyplot as plt
from sympy import gcdex
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, transpile, assemble
from qiskit.visualization import plot_histogram
from sympy import Rational, gcdex
from sympy.ntheory.continued_fraction import continued_fraction, continued_fraction_convergents

from qft import qft
from elementary import ax_modM
from order_finding import order_finding
from classical_utils import decode_bin

def discrete_log(a: int, b: int, p: int, show_hist: Optional[bool] = False, coef_t: Optional[int] = 1) -> int:
    """Shor's discrete log algorithm: given $a,b,p\in\mathbb{Z}$, it finds $s$ such that $a^s\equiv b\pmod p$.

    Args:
        a (int): $a$
        b (int): $b$
        p (int): $p$
    """

    r = order_finding(x=a, N=p, show_hist=False)
    t = coef_t * int(np.ceil(np.log2(p)))

    first_register = QuantumRegister(t)
    second_register = QuantumRegister(t)
    third_register = QuantumRegister(2 * t)
    auxiliary_register_mid = QuantumRegister(t)
    auxiliary_register_end = QuantumRegister(6 * t - 2)
    classical_register = ClassicalRegister(2 * t)

    qc = QuantumCircuit(
        first_register,
        second_register,
        third_register,
        auxiliary_register_mid,
        auxiliary_register_end,
        classical_register,
    )

    qc.h(first_register)
    qc.h(second_register)

    qc.append(ax_modM(a=b, M=p, N_len=t), list(first_register) + list(auxiliary_register_mid) + list(third_register) + list(auxiliary_register_end))
    qc.append(ax_modM(a=a, M=p, N_len=t, x_0_at_first=False), list(second_register) + list(auxiliary_register_mid) + list(third_register) + list(auxiliary_register_end))

    qc.append(qft(n=t).inverse(), first_register)
    qc.append(qft(n=t).inverse(), second_register)

    qc.measure(list(first_register) + list(second_register), classical_register)
    #qc.measure(third_register, classical_register)

    backend = Aer.get_backend('aer_simulator_matrix_product_state')#('aer_simulator')
    qc = transpile(qc, backend)
    job = backend.run(qc, shots=10000)
    hist = job.result().get_counts()

    if show_hist:
        figsize_x = max(7 * (len(hist) // 8), 7)
        plot_histogram(hist, figsize=(figsize_x, 5))
        plt.savefig(f'img/discrete_log_a{a}_b{b}_p{p}_r{r}_t{t}.png', bbox_inches='tight')

    for measured_key, _ in sorted(hist.items(), key=lambda x: x[1], reverse=True):
        tilde_l_per_r = Rational(decode_bin(measured_key[:t]), 2 ** t) # decoded from second register: $\widetilde{l/r}$
        if tilde_l_per_r == 0:
            continue

        l = None
        for fraction in continued_fraction_convergents(continued_fraction(tilde_l_per_r)):
            if fraction.denominator == r:
                l = fraction.numerator # get correct $l$
                break

        if l is None:
            continue

        tilde_beta_per_r = Rational(decode_bin(measured_key[-t:]), 2 ** t) # decoded from first register: $\widetilde{\beta/r}$
        if tilde_beta_per_r == 0:
            if pow(a, 0, p) == b: # the case where b == 1
                return r # returning 0 is also ok
            continue

        beta = None
        for fraction in continued_fraction_convergents(continued_fraction(tilde_beta_per_r)):
            if fraction.denominator == r:
                beta = fraction.numerator # get correct $\beta$
                break

        if beta is None:
            continue

        s, alpha, _ = gcdex(l, -r)
        s = int(s * beta)
        if pow(a, s, p) == b:
            return s

    raise Exception('s is NOT found!')


if __name__ == '__main__':
    #print(discrete_log(a=2, b=4, p=7, show_hist=True))
    #print(discrete_log(a=3, b=5, p=11, show_hist=True, coef_t=2))
    print(discrete_log(a=2, b=1, p=3, show_hist=True, coef_t=1))
