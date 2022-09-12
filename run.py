import netsquid as ns
from netsquid.qubits.qformalism import QFormalism

ns.get_qstate_formalism()
ns.set_qstate_formalism(QFormalism.DM)

c1,c2,c3,c4,e1,e2 = ns.qubits.create_qubits(6)

print(ns.qubits.reduced_dm([c1,c2,c3,c4,e1,e2]))