# Math imports
import numpy as np

# Netsquid imports
import netsquid as ns
from netsquid_nv.magic_distributor import NVDoubleClickMagicDistributor, NVSingleClickMagicDistributor
from netsquid.nodes import Node
from netsquid.qubits.ketstates import BellIndex
import netsquid.components.instructions as instr
from netsquid_nv.nv_center import NVQuantumProcessor
from netsquid.qubits.qubitapi import reduced_dm, assign_qstate

# Local imports
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

# Theoretical and analytical functions for calculations

def create_cardinal_states_distance_2():
    """ Create the vectors for logical states and matrices for logical Pauli operators for the distance-2 code. """

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
    """ Create a logical pure target state to compare with, for state preparation. Parameterized by theta and phi angles. """

    logical_0 = (np.kron(np.kron(KET_0, KET_0) , np.kron(KET_0, KET_0)) + np.kron(np.kron(KET_1, KET_1) , np.kron(KET_1, KET_1)))/np.sqrt(2)
    logical_1 = (np.kron(np.kron(KET_0, KET_1) , np.kron(KET_0, KET_1)) + np.kron(np.kron(KET_1, KET_0) , np.kron(KET_1, KET_0)))/np.sqrt(2)
    psi_logical = (logical_0 * (np.cos(theta/2))**2 + logical_1 * (np.exp(-1j*phi)*np.sin(theta/2))**2)/(np.sqrt((np.cos(theta/2))**4+(np.sin(theta/2))**4))
    rho_logical = np.outer(psi_logical, psi_logical)

    z_L = np.outer(logical_0, logical_0) - np.outer(logical_1, logical_1)
    x_l = np.outer(logical_0, logical_1) + np.outer(logical_1, logical_0)
    y_L = -1j * np.outer(logical_0, logical_1) + 1j* np.outer(logical_1, logical_0)
    return rho_logical, x_l, y_L, z_L

"""
    Node initialization modules
"""

def physical_initialization_cardinal_states(node_A: Node, node_B: Node, command:str = "0000"):
    """ Used for calculating the input-output matrix. Generates a state in the Z computational
    basis over all the physical qubits on the code. """

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



"""
    Node Operations    
"""

def reset_node(node: Node):
    """ Resets any input node to resample in some experiment. By reinitializing all the qubits to |0⟩ state each. All the electrons and carbons. """
    num_qubits = len(node.qmemory.mem_positions) - 1
    for i in range(num_qubits):
        node.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[i])
        ns.sim_run()
    return

def create_Bell_Pair(node_A: Node, node_B: Node):
    """ Creates Bell pair between any two input nodes using the DoubleClickMagicDistributor entity. Just call this function
    for any two nodes when Bell state is needed. Noise is already modelled by the global noise parameters, which can be overwritten. """

    entanglement_gen = NVDoubleClickMagicDistributor(nodes=[node_A, node_B], length_A=0.00001, length_B=0.00001,
                                                 coin_prob_ph_ph=1., coin_prob_ph_dc=0., coin_prob_dc_dc=0.)
    
    event = entanglement_gen.add_delivery({node_A.ID: 0, node_B.ID: 0}) # add the delivery, by attempting probabilistically 
    label = entanglement_gen.get_label(event) # Get the labdel of the detector which clicked, to  correct for the rotation to \Phi+>
    ns.sim_run()
    rotate = Rotate_Bell_Pair(num_qubits=3) # Rotate once
    phase_gate = Phase_Correction(num_qubits=3) # Apply phase correction based on which detector clicked
    program = rotate

    # Apply phase correction based on the detector click, to make \Phi+>
    if label[1] == BellIndex.PSI_MINUS:
        program += phase_gate
    node_A.qmemory.execute_program(program, qubit_mapping=[0, 1, 2], check_qubit_mapping=True)
    ns.sim_run()

def get_instantaneous_data_qubit_density_matrix(nodes: list(Node)):
    """
    Qubit placement is D1, D3 in node A and D2, D4 in node B. But the denisty matrix follows D1-D2-D3-D4 ordering for representation.
    """
    if len(nodes) > 2:
        print("Computational error! Only two nodes allowed with two carbon data qubits each.")

    carbon_1 = nodes[0].qmemory.peek([1])[0]
    carbon_3 = nodes[0].qmemory.peek([2])[0]
    carbon_2 = nodes[1].qmemory.peek([1])[0]
    carbon_4 = nodes[1].qmemory.peek([2])[0]
    data_density_matrix = reduced_dm([carbon_1, carbon_2, carbon_3, carbon_4])

    return data_density_matrix

"""
    Tomography and gate benchmarking functions
"""

def physical_cardinal_state_init(node: Node, state: str = "0"):
    """ Prepares the carbon qubit (physical qubit) in the desired state for state tomography. """

    node.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[1])
    ns.sim_run()
    if state == "0":
        pass
    elif state == "1":
        node.qmemory.execute_instruction(instr.INSTR_ROT_X, qubit_mapping=[1], angle=np.pi)
        ns.sim_run()
    elif state == "+":
        node.qmemory.execute_instruction(instr.INSTR_ROT_Y, qubit_mapping=[1], angle=np.pi/2)
        ns.sim_run()
    elif state == "-":
        node.qmemory.execute_instruction(instr.INSTR_ROT_Y, qubit_mapping=[1], angle=-np.pi/2)
        ns.sim_run()
    elif state == "+i":
        node.qmemory.execute_instruction(instr.INSTR_ROT_X, qubit_mapping=[1], angle=np.pi/2)
        ns.sim_run()
    elif state == "-i":
        node.qmemory.execute_instruction(instr.INSTR_ROT_X, qubit_mapping=[1], angle=-np.pi/2)
        ns.sim_run()
    else:
        raise RuntimeError("Invalid initialization parameter!")

    return


def physical_pauli_measure(node: Node, basis: str = "Z"):
    """ Measure the physical carbon qubit in the desired basis as commanded. These repeated state preparation and measurement are used to
    calculate the expectation values at the end. """
    if basis == "Z":
        
    pass

def state_tomography_physical():
    pass

def physical_PTM_ideal(op, num_qubits):
    if op == "X":
        pass
    return

def logical_PTM():
    pass
