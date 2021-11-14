"""
Implementing arXiv:quant-ph/9511018
"""

from typing import Optional

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate


def carry() -> Gate:
    qc = QuantumCircuit(4)
    qc.ccx(1, 2, 3)
    qc.cx(1, 2)
    qc.ccx(0, 2, 3)
    return qc.to_gate()

def qsum() -> Gate:
    qc = QuantumCircuit(3)
    qc.cx(1, 2)
    qc.cx(0, 2)
    return qc.to_gate()

def adder(n: int) -> Gate:
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

def adder_modN(N: int, N_len: Optional[int] = None) -> Gate:
    N_val = N
    if N_len is None:
        N_len = int(np.ceil(np.log2(N)))

    qubits = QuantumRegister(N_len + (N_len + 1) + N_len + N_len + 1)
    a, left_qubits = qubits[:N_len], qubits[N_len:]
    b, left_qubits = left_qubits[:N_len + 1], left_qubits[N_len + 1:]
    c, left_qubits = left_qubits[:N_len], left_qubits[N_len:]
    N, left_qubits = left_qubits[:N_len], left_qubits[N_len:]
    t = left_qubits[:1]
    qc = QuantumCircuit(qubits)

    adder_gate = adder(N_len)
    adder_gate_dag = adder_gate.inverse()

    for i, char in enumerate(bin(N_val)[2:][::-1]):
        if char == '1':
            qc.x(N[i])

    qc.append(adder_gate, a + b + c)
    qc.append(adder_gate_dag, N + b + c)
    qc.x(b[-1])
    qc.cx(b[-1], t[0])
    qc.x(b[-1])
    for i, char in enumerate(bin(N_val)[2:][::-1]):
        if char == '1':
            qc.cx(t[0], N[i])
    qc.append(adder_gate, N + b + c)
    for i, char in enumerate(bin(N_val)[2:][::-1]):
        if char == '1':
            qc.cx(t[0], N[i])
    qc.append(adder_gate_dag, a + b + c)
    qc.cx(b[-1], t[0])
    qc.append(adder_gate, a + b + c)

    for i, char in enumerate(bin(N_val)[2:][::-1]):
        if char == '1':
            qc.x(N[i])

    return qc.to_gate()

def adder_modN_hoge(N: int, N_len: Optional[int] = None) -> Gate:
    N_val = N
    if N_len is None:
        N_len = int(np.ceil(np.log2(N)))

    qubits = QuantumRegister(N_len + (N_len + 1) + N_len + N_len + 1)
    a, left_qubits = qubits[:N_len], qubits[N_len:]
    b, left_qubits = left_qubits[:N_len + 1], left_qubits[N_len + 1:]
    c, left_qubits = left_qubits[:N_len], left_qubits[N_len:]
    N, left_qubits = left_qubits[:N_len], left_qubits[N_len:]
    t = left_qubits[:1]
    qc = QuantumCircuit(qubits)

    adder_gate = adder(N_len)
    adder_gate_dag = adder_gate.inverse()

    for i, char in enumerate(bin(N_val)[2:][::-1]):
        if char == '1':
            qc.x(N[i])
    qc.append(adder_gate_dag, a + b + c)
    return qc.to_gate()
    qc.cx(b[-1], t[0])
    qc.append(adder_gate, a + b + c)

def ctrl_multi_modN(a: int, N: int) -> Gate:
    N_val = N
    N_len = int(np.ceil(np.log2(N)))

    qubits = QuantumRegister(9 * N_len - 1)
    ctrl, left_qubits = qubits[:1], qubits[1:]
    x, left_qubits = left_qubits[:N_len], left_qubits[N_len:]
    y, left_qubits = left_qubits[:2 * N_len], left_qubits[2 * N_len:]
    xx, left_qubits = left_qubits[:2 * N_len - 1], left_qubits[2 * N_len - 1:]
    c, left_qubits = left_qubits[:2 * N_len - 1], left_qubits[2 * N_len - 1:]
    N, left_qubits = left_qubits[:2 * N_len - 1], left_qubits[2 * N_len - 1:]
    t, left_qubits = left_qubits[:1], qubits[1:]

    qc = QuantumCircuit(qubits)

    adder_modN_gate = adder_modN(N=N_val, N_len=2 * N_len - 1)

    for i in range(N_len):
        for j, char in enumerate(bin(2 ** i * a % N_val)[2:][::-1]):
            if char == '1':
                qc.ccx(ctrl[0], x[i], xx[j])
        qc.append(adder_modN_gate, xx + y + c + N + t)
        for j, char in enumerate(bin(2 ** i * a % N_val)[2:][::-1]):
            if char == '1':
                qc.ccx(ctrl[0], x[i], xx[j])
    qc.x(ctrl[0])
    for x_bit, y_bit in zip(x, y):
        qc.ccx(ctrl[0], x_bit, y_bit)
    qc.x(ctrl[0])

    return qc.to_gate()

def ctrl_multi_modN_hoge(a: int, N: int) -> Gate:
    N_val = N
    N_len = int(np.ceil(np.log2(N)))

    qubits = QuantumRegister(9 * N_len - 1)
    ctrl, left_qubits = qubits[:1], qubits[1:]
    x, left_qubits = left_qubits[:N_len], left_qubits[N_len:]
    y, left_qubits = left_qubits[:2 * N_len], left_qubits[2 * N_len:]
    xx, left_qubits = left_qubits[:2 * N_len - 1], left_qubits[2 * N_len - 1:]
    c, left_qubits = left_qubits[:2 * N_len - 1], left_qubits[2 * N_len - 1:]
    N, left_qubits = left_qubits[:2 * N_len - 1], left_qubits[2 * N_len - 1:]
    t, left_qubits = left_qubits[:1], qubits[1:]

    qc = QuantumCircuit(qubits)

    adder_modN_gate_dag = adder_modN(N=N_val, N_len=2 * N_len - 1).inverse()

    qc.x(ctrl[0])
    for x_bit, y_bit in zip(x, y):
        qc.ccx(ctrl[0], x_bit, y_bit)
    qc.x(ctrl[0])
    for i in reversed(range(N_len)):
        for j, char in enumerate(bin(2 ** i * a)[2:][::-1]):
            if char == '1':
                qc.ccx(ctrl[0], x[i], xx[j])
        if i == 0:
            adder_modN_gate_dag = adder_modN_hoge(N=N_val, N_len=2 * N_len - 1)
        qc.append(adder_modN_gate_dag, xx + y + c + N + t)
        if i == 0:
            return qc.to_gate()
        for j, char in enumerate(bin(2 ** i * a)[2:][::-1]):
            if char == '1':
                qc.ccx(ctrl[0], x[i], xx[j])

def ax_modN(a: int, N: int) -> Gate:
    N_val = N
    N_len = int(np.ceil(np.log2(N)))

    qubits = QuantumRegister(10 * N_len - 2)
    x, left_qubits = qubits[:N_len], qubits[N_len:]
    x_for_ctrl_multi_modN_gate, left_qubits = left_qubits[:N_len], left_qubits[N_len:]
    y, left_qubits = left_qubits[:2 * N_len], left_qubits[2 * N_len:]
    xx, left_qubits = left_qubits[:2 * N_len - 1], left_qubits[2 * N_len - 1:]
    c, left_qubits = left_qubits[:2 * N_len - 1], left_qubits[2 * N_len - 1:]
    N, left_qubits = left_qubits[:2 * N_len - 1], left_qubits[2 * N_len - 1:]
    t, left_qubits = left_qubits[:1], qubits[1:]

    qc = QuantumCircuit(qubits)

    qc.x(x_for_ctrl_multi_modN_gate[0])
    for i in range(N_len):
        ctrl_multi_modN_gate = ctrl_multi_modN(pow(a, 2 ** i, N_val), N_val)
        ctrl_multi_modN_gate_dag = ctrl_multi_modN(pow(a, -2 ** i, N_val), N_val).inverse()
        #ctrl_multi_modN_gate_dag = ctrl_multi_modN_hoge(pow(a, -2 ** i, N_val), N_val)

        qc.append(ctrl_multi_modN_gate, [x[i]] + x_for_ctrl_multi_modN_gate + y + xx + c + N + t)
        for j in range(N_len):
            qc.cswap(x[i], x_for_ctrl_multi_modN_gate[j], y[j])
        qc.append(ctrl_multi_modN_gate_dag, [x[i]] + x_for_ctrl_multi_modN_gate + y + xx + c + N + t)

    return qc.to_gate()
