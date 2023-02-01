from typing import Optional
from math import gcd

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, transpile, assemble
from qiskit.visualization import plot_histogram
from sympy import Rational
from sympy.ntheory.continued_fraction import continued_fraction, continued_fraction_convergents

from qft import qft
from elementary import ax_modM
from classical_utils import lcm

def order_finding(x: int, N: int, show_hist: Optional[bool] = False) -> int:
#def order_finding(x: int, N: int, epsilon: Optional[float] = 0.2, show_hist: Optional[bool] = False) -> int:
    r"""Order-finding algorithm: it finds $r$ of $x^r\equiv 1\pmod N$. It requires 

    Args:
        x (int): $x$
        N (int): $N$

    Returns:
        order $r$

    Examples:

    ```
    >>> order_finding(x=3, N=5, show_hist=True)
    4
    ```

    and get below image in `img` directory:  
    ![](../img/order_finding_x3_N5.png)  
    It represents $0/2^6=0$, $2^4/2^6=1/4$, $2^5/2^6=1/2$, and $(2^4+2^5)/2^6=3/4$ from the left.
    This answer is $r=4$, so $1/2$ looks wrong.
    However, $\tilde{r}=2$ is a factor of $r$, so we can get correct $r$ by lcm with another $\tilde{r}$.
    """

    L = int(np.ceil(np.log2(N)))
    t = 2 * L# + 1 + int(np.ceil(np.log2(3 + 1 / (2 * epsilon)))) # epsilon requires too many qubits to run this program...

    first_register = QuantumRegister(t)
    second_register = QuantumRegister(2 * t)
    auxiliary_register_mid = QuantumRegister(t)
    auxiliary_register_end = QuantumRegister(6 * t - 2)
    classical_register = ClassicalRegister(len(first_register))

    qc = QuantumCircuit(first_register, auxiliary_register_mid, second_register, auxiliary_register_end, classical_register)
#    qc.add_register(first_register)
#    qc.add_register(second_register)
#    qc.add_register(classical_register)

    qc.h(first_register)

    #import pdb; pdb.set_trace()
    #qc.append(ax_modM(a=x, M=N, N_len=len(first_register)), [first_register, auxiliary_register_mid, second_register, auxiliary_register_end])
    qc.append(ax_modM(a=x, M=N, N_len=len(first_register)), qc.qubits[:10 * t - 2])

    qc.append(qft(n=len(first_register)).inverse(), first_register)

    qc.measure(first_register, classical_register)

    backend = Aer.get_backend('aer_simulator_matrix_product_state')#('aer_simulator')
    qc = transpile(qc, backend)
    job = backend.run(qc, shots=10000)
    hist = job.result().get_counts()

    if show_hist:
        plot_histogram(hist)
        plt.savefig(f'img/order_finding_x{x}_N{N}.png', bbox_inches='tight')
        #plt.savefig(f'img/order_finding_x{x}_N{N}_eps{epsilon}.png', bbox_inches='tight')

    all_fractions = []
    for measured_key, _ in sorted(hist.items(), key=lambda x: x[1], reverse=True):
        tilde_s_per_r = Rational(int(measured_key[-t:], 2), 2 ** t)
        if tilde_s_per_r == 0:
            continue

        fractions = []
        for fraction in continued_fraction_convergents(continued_fraction(tilde_s_per_r)):
            if pow(x, fraction.denominator, N) == 1:
                return fraction.denominator
            fractions.append(fraction)

            for other_fraction in all_fractions:
                if math.gcd(fraction.numerator, other_fraction.numerator) == 1:
                    r_candidate = lcm(fraction.denominator, other_fraction.denominator)
                    if pow(x, r_candidate, N) == 1:
                        return r_candidate

    raise Exception('r is NOT found!')


if __name__ == '__main__':
    #print(order_finding(x=5, N=21, show_hist=True))
    print(order_finding(x=3, N=5, show_hist=True))
