from netsquid_nv.nv_center import NVQuantumProcessor
from netsquid.nodes import Node


from native_gates_and_parameters import add_native_gates


def create_system():
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

        add_native_gates(processor_A)
        add_native_gates(processor_B)

        return node_A, node_B