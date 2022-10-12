import numpy as np
import qiskit as qk
import netsquid as ns
from netsquid_nv.magic_distributor import NVDoubleClickMagicDistributor, NVSingleClickMagicDistributor
from netsquid.nodes import Node
from netsquid.qubits.ketstates import BellIndex
import netsquid.components.instructions as instr
from netsquid_nv.nv_center import NVQuantumProcessor

from q_programs import Rotate_Bell_Pair, Phase_Correction
from native_gates_and_parameters import add_native_gates

KET_0 = np.array([[1], [0]])
KET_1 = np.array([[0], [1]])

KET_PLUS = (KET_0 + KET_1)/np.sqrt(2)
KET_MINUS = (KET_0 - KET_1)/np.sqrt(2)

KET_i_PLUS = (KET_0 + 1j * KET_1)/np.sqrt(2)
KET_i_MINUS = (KET_0 - 1j * KET_1)/np.sqrt(2)

PAULI_X = np.array([[0, 1],
                    [1, 0]])
PAULI_Y = np.array([[0, -1j],
                    [1j, 0]])
PAULI_Z = np.array([[1, 0],
                    [0, -1]])


def create_cardinal_states_distance_2():
    ket_0L = (np.kron(np.kron(KET_0, KET_0) , np.kron(KET_0, KET_0)) + np.kron(np.kron(KET_1, KET_1) , np.kron(KET_1, KET_1)))/np.sqrt(2)
    ket_1L = (np.kron(np.kron(KET_0, KET_1) , np.kron(KET_0, KET_1)) + np.kron(np.kron(KET_1, KET_0) , np.kron(KET_1, KET_0)))/np.sqrt(2)

    ket_plus_L = (ket_0L + ket_1L)/np.sqrt(2)
    ket_minus_L = (ket_0L - ket_1L)/np.sqrt(2)

    ket_iplus_L = (ket_0L + 1j * ket_1L)/np.sqrt(2)
    ket_iminus_L = (ket_0L - 1j * ket_1L)/np.sqrt(2)

    z_L = np.outer(ket_0L, ket_0L) - np.outer(ket_1L, ket_1L)
    x_l = np.outer(ket_0L, ket_1L) + np.outer(ket_1L, ket_0L)
    y_L = -1j * np.outer(ket_0L, ket_1L) + 1j* np.outer(ket_1L, ket_0L)

    return [ket_0L, ket_1L, ket_plus_L, ket_minus_L, ket_iplus_L, ket_iminus_L], [x_l, y_L, z_L]



def create_theoretical_rho(theta:float=0, phi:float=0):
    logical_0 = (np.kron(np.kron(KET_0, KET_0) , np.kron(KET_0, KET_0)) + np.kron(np.kron(KET_1, KET_1) , np.kron(KET_1, KET_1)))/np.sqrt(2)
    logical_1 = (np.kron(np.kron(KET_0, KET_1) , np.kron(KET_0, KET_1)) + np.kron(np.kron(KET_1, KET_0) , np.kron(KET_1, KET_0)))/np.sqrt(2)
    psi_logical = (logical_0 * (np.cos(theta/2))**2 + logical_1 * (np.exp(-1j*phi)*np.sin(theta/2))**2)/(np.sqrt((np.cos(theta/2))**4+(np.sin(theta/2))**4))
    rho_logical = np.outer(psi_logical, psi_logical)

    z_L = np.outer(logical_0, logical_0) - np.outer(logical_1, logical_1)
    x_l = np.outer(logical_0, logical_1) + np.outer(logical_1, logical_0)
    y_L = -1j * np.outer(logical_0, logical_1) + 1j* np.outer(logical_1, logical_0)
    return rho_logical, x_l, y_L, z_L





def physical_initialization_cardinal_states(node_A: Node, node_B: Node, command:str = "0000"):
    if command[0] == '1':
        node_A.qmemory.execute_instruction(instr.INSTR_ROT_X, qubit_mapping=[1], angle=np.pi)
        ns.sim_run()
    
    if command[1] == '1':
        node_B.qmemory.execute_instruction(instr.INSTR_ROT_X, qubit_mapping=[1], angle=np.pi)
        ns.sim_run()

    if command[2] == '1':
        node_A.qmemory.execute_instruction(instr.INSTR_ROT_X, qubit_mapping=[2], angle=np.pi)
        ns.sim_run()

    if command[3] == '1':
        node_B.qmemory.execute_instruction(instr.INSTR_ROT_X, qubit_mapping=[2], angle=np.pi)
        ns.sim_run()

    return

def state_tomography_physical():
    pass

def physical_PTM_ideal(op, num_qubits):
    if op == "X":
        pass
    return

def logical_PTM():
    pass


"""
    Network components creation
"""
def create_two_node_setup():
    # Component creation
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


"""
    Node Operations    
"""

def reset_node(node: Node):
    num_qubits = len(node.qmemory.mem_positions) - 1
    for i in range(num_qubits):
        node.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[i])
        ns.sim_run()
    return

def create_Bell_Pair(node_A: Node, node_B: Node):
    entanglement_gen = NVDoubleClickMagicDistributor(nodes=[node_A, node_B], length_A=0.00001, length_B=0.00001,
                                                 coin_prob_ph_ph=1., coin_prob_ph_dc=0., coin_prob_dc_dc=0.)
    
    event = entanglement_gen.add_delivery({node_A.ID: 0, node_B.ID: 0})
    label = entanglement_gen.get_label(event)
    ns.sim_run()
    rotate = Rotate_Bell_Pair(num_qubits=3)
    phase_gate = Phase_Correction(num_qubits=3)
    program = rotate

    # Apply phase correction based on the detector click, to make \Phi+>
    if label[1] == BellIndex.PSI_MINUS:
        program += phase_gate
    node_A.qmemory.execute_program(program, qubit_mapping=[0, 1, 2], check_qubit_mapping=True)
    ns.sim_run()