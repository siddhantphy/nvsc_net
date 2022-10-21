# Netsquid imports
import netsquid as ns
from netsquid_nv.nv_center import NVQuantumProcessor
from netsquid.nodes import Node


from native_gates_and_parameters import add_native_gates


"""
    Network components creation
"""
def create_two_node_setup(no_noise: bool = True, parameters: dict={}):
    # Component creation
    node_A = Node("Node: A")
    processor_A = NVQuantumProcessor(num_positions=3, noiseless=no_noise, **parameters)
    node_A.add_subcomponent(processor_A, name="Node A processor")
    node_A.add_ports(['Q_in_Ent'])
    node_A.ports['Q_in_Ent'].forward_input(node_A.qmemory.ports['qin'])
    e1, c1, c3 = ns.qubits.create_qubits(3)
    processor_A.put([e1,c1,c3])


    node_B = Node("Node: B")
    processor_B = NVQuantumProcessor(num_positions=3, noiseless=no_noise, **parameters)
    node_B.add_subcomponent(processor_B, name="Node B processor")
    node_B.add_ports(['Q_in_Ent'])
    node_B.ports['Q_in_Ent'].forward_input(node_B.qmemory.ports['qin'])
    e2, c2, c4 = ns.qubits.create_qubits(3)
    processor_B.put([e2,c2,c4])

    add_native_gates(processor_A)
    add_native_gates(processor_B)

    return node_A, node_B

def create_physical_qubit_single_node_setup(no_noise: bool = True, parameters: dict={}):
    # Component creation

    node = Node("Node: A")
    processor = NVQuantumProcessor(num_positions=3, noiseless=no_noise, **parameters)
    node.add_subcomponent(processor, name="Node processor")
    node.add_ports(['Q_in_Ent'])
    node.ports['Q_in_Ent'].forward_input(node.qmemory.ports['qin'])
    e, c = ns.qubits.create_qubits(2)
    processor.put([e,c])

    add_native_gates(processor)

    return node