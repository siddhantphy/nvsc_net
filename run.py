# Netsquid imports
import netsquid as ns
from gates.gates import *
from netsquid.nodes import Node
from netsquid.components import QuantumMemory
from netsquid.qubits.qformalism import QFormalism


# Standard packages
import numpy as np
import pandas as pd



# Set Density Matrices as the working representation
ns.get_qstate_formalism()
ns.set_qstate_formalism(QFormalism.DM)

# Create the network components
c1,c2,c3,c4,e1,e2 = ns.qubits.create_qubits(6)

node_a = Node("A")
node_b = Node("B")

qma_d = QuantumMemory("Node A data memory", num_positions=2)
qma_e = QuantumMemory("Node A electron memory", num_positions=1)
node_a.add_subcomponent(qma_d, name = "A data memory")
node_a.add_subcomponent(qma_e, name = "A electron memory")
qma_d.put([c1, c3])
qma_e.put(e1)


qmb_d = QuantumMemory("Node B data memory", num_positions=2)
qmb_e = QuantumMemory("Node B electron memory", num_positions=1)
node_b.add_subcomponent(qmb_d, name = "B data memory")
node_b.add_subcomponent(qmb_e, name = "B electron memory")
qmb_d.put([c2, c4])
qmb_e.put(e2)




# Quantum Circuit
ns.qubits.operate(c1, H)
ns.qubits.operate(c1, Rz(np.pi/4))




print(ns.qubits.reduced_dm([c1]))
# print(Rz(np.pi).arr)

print(node_a.subcomponents)
print(qma_d.peek)