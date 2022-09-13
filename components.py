import logging
import numpy as np
import netsquid as ns
import netsquid.qubits.operators as ops
from netsquid_nv.nv_center import NVQuantumProcessor
from netsquid_nv.magic_distributor import NVDoubleClickMagicDistributor, NVSingleClickMagicDistributor
from netsquid.nodes import Node
from netsquid.protocols import Protocol
from netsquid.components.qprogram import *
from netsquid.qubits.qubitapi import reduced_dm
import netsquid.components.instructions as instr

ns.logger.setLevel(logging.DEBUG)
node_A = Node("Node: A")
processor_A = NVQuantumProcessor(num_positions=3, noiseless=True)
node_A.add_subcomponent(processor_A, name="Node A processor")
node_A.add_ports(['Q_in_Ent'])
node_A.ports['Q_in_Ent'].forward_input(node_A.qmemory.ports['qin'])
e1, c1, c3 = ns.qubits.create_qubits(3)

processor_A.put([e1,c1,c3])


node_B = Node("Node: B")
processor_B = NVQuantumProcessor(num_positions=3, noiseless=True)
node_B.add_subcomponent(processor_B, name="Node B processor")
node_B.add_ports(['Q_in_Ent'])
node_B.ports['Q_in_Ent'].forward_input(node_B.qmemory.ports['qin'])
e2, c2, c4 = ns.qubits.create_qubits(3)
processor_B.put([e2,c2,c4])


entanglement_gen = NVDoubleClickMagicDistributor(nodes=[node_A, node_B], length_A=0.001, length_B=0.001,
                                                 coin_prob_ph_ph=1., coin_prob_ph_dc=0., coin_prob_dc_dc=0.)
# entanglement_gen = NVSingleClickMagicDistributor(nodes=[node_A, node_B], length_A=0.001, length_B=0.001, alpha_A=0.1, alpha_B=0.1)

# Protocols
class zz_stabilizer(Protocol):
    pass


class xxxx_stabilizer(Protocol):
    pass


entanglement_gen.add_delivery({node_A.ID: 0, node_B.ID: 0})
ns.sim_run()

class CustomQProgram(QuantumProgram):
    default_num_qubits = 3

    def program(self):
        e1, c1, c3 = self.get_qubit_indices(3)
        self.apply(instr.INSTR_CXDIR, [e1, c1], angle=np.pi)
        self.apply(instr.INSTR_CXDIR, [e1, c3], angle=np.pi)
        self.apply(instr.INSTR_MEASURE, e1, output_key="mA")
        yield self.run()


quantum_prog_A = CustomQProgram(num_qubits=3)
quantum_prog_B = CustomQProgram(num_qubits=3)

node_A.qmemory.execute_program(quantum_prog_A, qubit_mapping=[0, 1, 2])
ns.sim_run()
node_B.qmemory.execute_program(quantum_prog_B, qubit_mapping=[0, 1, 2])
ns.sim_run()

print(quantum_prog_A.output["mA"])
print(quantum_prog_B.output["mA"])
# electron_1 = node_A.qmemory.peek([0])[0]
# electron_2 = node_B.qmemory.peek([0])[0]
# print(electron_2)
# print(reduced_dm([electron_1, electron_2]))


# print(processor_A.measure(positions=[0]))
print("")