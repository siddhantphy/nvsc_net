import netsquid as ns
from gates.gates import *
from netsquid.qubits.qformalism import QFormalism

import numpy as np

# Set Density Matrices as the working representation
ns.get_qstate_formalism()
ns.set_qstate_formalism(QFormalism.DM)

# Create the network components
c1,c2,c3,c4,e1,e2 = ns.qubits.create_qubits(6)


# Quantum Circuit
ns.qubits.operate(c1, H)
ns.qubits.operate(c1, Rz(np.pi/4))




print(ns.qubits.reduced_dm([c1]))
print(Rz(np.pi).arr)