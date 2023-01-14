"""
Implementing [arXiv:quant-ph/9511018](https://arxiv.org/abs/quant-ph/9511018)
"""

from typing import Optional

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate


def carry() -> Gate:
    """CARRY. It requires 4 qubits.
    
    Returns:
        its gate
    """

    qc = QuantumCircuit(4)
    qc.ccx(1, 2, 3)
    qc.cx(1, 2)
    qc.ccx(0, 2, 3)
    return qc.to_gate()

def qsum() -> Gate:
    """SUM. It requires 3 qubits.
    
    Returns:
        its gate
    """

    qc = QuantumCircuit(3)
    qc.cx(1, 2)
    qc.cx(0, 2)
    return qc.to_gate()

def adder(n: int) -> Gate:
    r"""ADDER: $a,b\to a,a+b$. It requires $3n+1$ qubits: $a$ uses $n$ qubits, $b$ uses $n+1$ qubits, and $c$ uses $n$ qubits.

    Args:
      n (int): $n$ bits for representing $a$.

    Returns:
        its gate
    """

    qubits = QuantumRegister(n + (n + 1) + n)
    a = qubits[:n]
    b = qubits[n:n + (n + 1)]
    c = qubits[n + (n + 1):]
    qc = QuantumCircuit(qubits)

    carry_gate = carry()
    carry_gate_dag = carry_gate.inverse()
    sum_gate = qsum()

    for i in range(n - 1):
        qc.append(carry_gate, [c[i]] + [a[i]] + [b[i]] + [c[i + 1]])
    qc.append(carry_gate, [c[n - 1]] + [a[-1]] + [b[n - 1]] + [b[n]])
    qc.cx(a[-1], b[n - 1])
    qc.append(sum_gate, [c[n - 1]] + [a[-1]] + [b[n - 1]])
    for i in reversed(range(n - 1)):
        qc.append(carry_gate_dag, [c[i]] + [a[i]] + [b[i]] + [c[i + 1]])
        qc.append(sum_gate, [c[i]] + [a[i]] + [b[i]])

    return qc.to_gate()

def adder_modM(M: int, N_len: int) -> Gate:
    r"""ADDER MOD: $a,b\to a,a+b\mod M$. It requires $4N_\mathit{len}+2$ qubits: $a$ uses $N_\mathit{len}$ qubits, $b$ uses $N_\mathit{len}+1$ qubits, $c$ uses $N_\mathit{len}$ qubits, $M$ uses $N_\mathit{len}$ qubits, and $t$ uses 1 qubit.

    Args:
        M (int): $N$ in the paper, but we uses $M$ instead.
        N_len (int): a number of bits for representing $a$.

    Returns:
        its gate
    """

    M_val = M

    qubits = QuantumRegister(N_len + (N_len + 1) + N_len + N_len + 1)
    a, left_qubits = qubits[:N_len], qubits[N_len:]
    b, left_qubits = left_qubits[:N_len + 1], left_qubits[N_len + 1:]
    c, left_qubits = left_qubits[:N_len], left_qubits[N_len:]
    M, left_qubits = left_qubits[:N_len], left_qubits[N_len:]
    t = left_qubits[:1]
    qc = QuantumCircuit(qubits)

    adder_gate = adder(N_len)
    adder_gate_dag = adder_gate.inverse()

    for i, char in enumerate(bin(M_val)[2:][::-1]):
        if char == '1':
            qc.x(M[i])

    qc.append(adder_gate, a + b + c)
    qc.append(adder_gate_dag, M + b + c)
    qc.x(b[-1])
    qc.cx(b[-1], t[0])
    qc.x(b[-1])
    for i, char in enumerate(bin(M_val)[2:][::-1]):
        if char == '1':
            qc.cx(t[0], M[i])
    qc.append(adder_gate, M + b + c)
    for i, char in enumerate(bin(M_val)[2:][::-1]):
        if char == '1':
            qc.cx(t[0], M[i])
    qc.append(adder_gate_dag, a + b + c)
    qc.cx(b[-1], t[0])
    qc.append(adder_gate, a + b + c)

    for i, char in enumerate(bin(M_val)[2:][::-1]):
        if char == '1':
            qc.x(M[i])

    return qc.to_gate()

def ctrl_multi_modM(a: int, M: int, N_len: int) -> Gate:
    r"""Ctrl MULT MOD: $x,0\to x,ax\mod M$ if $c=1$, otherwise $x,0\to x,x$. It requires $9N_\mathit{len}-1$ qubits: $\mathit{ctrl}$ uses 1 qubit, $x$ uses $N_\mathit{len}$ qubits, $y$ uses $2N_\mathit{len}$ qubits, $\mathit{xx}$ (which is a register in the middle of Fig. 5 in the paper) uses $2N_\mathit{len}-1$ qubits, $c$ (which is $c$ for ADDER MOD) uses $2N_\mathit{len}-1$ qubits, $M$ (which is $M$ for ADDER MOD) uses $2N_\mathit{len}-1$ qubits, and $t$ (which is $t$ for ADDER MOD) uses 1 qubit.

    Args:
        a (int): $a$
        M (int): $M$ (see `adder_modM`)
        N_len (int): a number of bits for representing $x$

    Returns:
        its gate
    """

    M_val = M
    N_len = N_len

    qubits = QuantumRegister(9 * N_len - 1)
    ctrl, left_qubits = qubits[:1], qubits[1:]
    x, left_qubits = left_qubits[:N_len], left_qubits[N_len:]
    y, left_qubits = left_qubits[:2 * N_len], left_qubits[2 * N_len:]
    xx, left_qubits = left_qubits[:2 * N_len - 1], left_qubits[2 * N_len - 1:]
    c, left_qubits = left_qubits[:2 * N_len - 1], left_qubits[2 * N_len - 1:]
    M, left_qubits = left_qubits[:2 * N_len - 1], left_qubits[2 * N_len - 1:]
    t, left_qubits = left_qubits[:1], qubits[1:]

    qc = QuantumCircuit(qubits)

    adder_modM_gate = adder_modM(M=M_val, N_len=2 * N_len - 1)

    for i in range(N_len):
        for j, char in enumerate(bin(2 ** i * a % M_val)[2:][::-1]):
            if char == '1':
                qc.ccx(ctrl[0], x[i], xx[j])
        qc.append(adder_modM_gate, xx + y + c + M + t)
        for j, char in enumerate(bin(2 ** i * a % M_val)[2:][::-1]):
            if char == '1':
                qc.ccx(ctrl[0], x[i], xx[j])
    qc.x(ctrl[0])
    for x_bit, y_bit in zip(x, y):
        qc.ccx(ctrl[0], x_bit, y_bit)
    qc.x(ctrl[0])

    return qc.to_gate()

def ax_modM(a: int, M: int, N_len: Optional[int] = None, x_0_at_first: bool = True) -> Gate:
    r"""Modular exponentiation, $a^x\mod M$. It requires $10N_\mathit{len}-2$ qubits: $x$ uses $N_\mathit{len}$ qubits, $\mathit{x\ for\ Ctrl\ MULT\ MOD}$ uses $N_\mathit{len}$ qubits, $y$ uses $2N_\mathit{len}$ qubits, $\mathit{xx}$ uses $2N_\mathit{len}-1$ qubits, $c$ (which is $c$ for ADDER MOD) uses $2N_\mathit{len}-1$ qubits, $M$ (which is $M$ for ADDER MOD) uses $2N_\mathit{len}-1$ qubits, and $t$ (which is $t$ for ADDER MOD) uses 1 qubit.

    Args:
        a (int): $a$
        M (int): $M$ (see `adder_modM`)
        N_len (int): a number of bits for representing $x$
        x_0_at_first (bool): if True, it adds 1 into the target register before calculating modular exponentiation

    Returns:
        its gate
    """

    M_val = M
    if N_len is None:
        N_len = int(np.ceil(np.log2(M)))

    qubits = QuantumRegister(10 * N_len - 2)
    x, left_qubits = qubits[:N_len], qubits[N_len:]
    x_for_ctrl_multi_modM_gate, left_qubits = left_qubits[:N_len], left_qubits[N_len:]
    y, left_qubits = left_qubits[:2 * N_len], left_qubits[2 * N_len:]
    xx, left_qubits = left_qubits[:2 * N_len - 1], left_qubits[2 * N_len - 1:]
    c, left_qubits = left_qubits[:2 * N_len - 1], left_qubits[2 * N_len - 1:]
    M, left_qubits = left_qubits[:2 * N_len - 1], left_qubits[2 * N_len - 1:]
    t, left_qubits = left_qubits[:1], qubits[1:]

    qc = QuantumCircuit(qubits)

    if x_0_at_first:
        qc.x(x_for_ctrl_multi_modM_gate[0])
    for i in range(N_len):
        ctrl_multi_modM_gate = ctrl_multi_modM(pow(a, 2 ** i, M_val), M_val, N_len)
        ctrl_multi_modM_gate_dag = ctrl_multi_modM(pow(a, -2 ** i, M_val), M_val, N_len).inverse()

        qc.append(ctrl_multi_modM_gate, [x[i]] + x_for_ctrl_multi_modM_gate + y + xx + c + M + t)
        for j in range(N_len):
            qc.cswap(x[i], x_for_ctrl_multi_modM_gate[j], y[j])
        qc.append(ctrl_multi_modM_gate_dag, [x[i]] + x_for_ctrl_multi_modM_gate + y + xx + c + M + t)

    return qc.to_gate()
