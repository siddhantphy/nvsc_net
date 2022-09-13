import netsquid.qubits as qubits
from gates.gates import *
from netsquid.nodes import Node
from netsquid.qubits import operators as ops
from netsquid.qubits.qubitapi import assign_qstate

def logical_initialization(theta: float, phi: float, node_a: Node):
    c1 = node_a.subcomponents["A data memory"].peek(0)[0]
    c3 = node_a.subcomponents["A data memory"].peek(1)[0]
    ns.qubits.operate(c1, Ry(theta))
    ns.qubits.operate(c3, Ry(theta))
    ns.qubits.operate(c3, Rz(phi))

    return


def zz_stabilizer(node: Node, name: str):
    c1 = node.subcomponents[f"{name} data memory"].peek(0)[0]
    c3 = node.subcomponents[f"{name} data memory"].peek(1)[0]
    e1 = node.subcomponents[f"{name} electron memory"].peek(0)[0]

    ns.qubits.operate(e1, ops.H)
    ns.qubits.operate([e1,c1], ops.CZ)
    ns.qubits.operate([e1,c3], ops.CZ)

    ns.qubits.measure(e1)
    assign_qstate([e1], np.array([[1,0],[0,0]]))


def xxxx_stabilizer():
    pass