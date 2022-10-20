# Math imports
import numpy as np

# utilities import
from itertools import product
import copy

# Plotting imports
import matplotlib.pyplot as plt

# Netsquid imports
import netsquid as ns
from netsquid_nv.magic_distributor import NVDoubleClickMagicDistributor, NVSingleClickMagicDistributor
from netsquid.nodes import Node
from netsquid.qubits.ketstates import BellIndex
import netsquid.components.instructions as instr
from netsquid_nv.move_circuits import reverse_move_using_CXDirections # Note that this results into a Hadamrd being applied on the electron, so there is a change of basis
from netsquid_nv.nv_center import NVQuantumProcessor
from netsquid.qubits.qubitapi import reduced_dm, assign_qstate

# Local imports
from q_programs import *
from native_gates_and_parameters import add_native_gates
from network_model import *

import time
timestr = time.strftime("%Y%m%d-%H%M%S")
# ns.logger.setLevel(logging.DEBUG)



KET_0 = np.array([[1], [0]])
KET_1 = np.array([[0], [1]])

KET_PLUS = (KET_0 + KET_1)/np.sqrt(2)
KET_MINUS = (KET_0 - KET_1)/np.sqrt(2)

KET_i_PLUS = (KET_0 + 1j * KET_1)/np.sqrt(2)
KET_i_MINUS = (KET_0 - 1j * KET_1)/np.sqrt(2)

IDENTITY = np.array([[1, 0],
                    [0, 1]])
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

def get_instantaneous_data_qubit_density_matrix(nodes):
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
        node.qmemory.execute_instruction(instr.INSTR_ROT_X, qubit_mapping=[1], angle=-np.pi/2)
        ns.sim_run()
    elif state == "-i":
        node.qmemory.execute_instruction(instr.INSTR_ROT_X, qubit_mapping=[1], angle=np.pi/2)
        ns.sim_run()
    else:
        raise RuntimeError("Invalid initialization parameter!")

    return


def physical_pauli_measure(node: Node, basis: str = "Z"):
    """ Measure the physical carbon qubit in the desired basis as commanded. These repeated state preparation and measurement are used to
    calculate the expectation values at the end. """

    measurement_result = "NA"

    if basis == "Z":
        z_m = Z_Measurement(num_qubits=2)
        reverse_move_using_CXDirections(z_m, 0, 1)
        node.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[0])
        ns.sim_run()
        node.qmemory.execute_program(z_m, qubit_mapping=[0, 1])
        ns.sim_run()
        measurement_result = int(z_m.output["M"][0])

    elif basis == "Y":
        y_m = Y_Measurement(num_qubits=2)
        reverse_move_using_CXDirections(y_m, 0, 1)
        node.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[0])
        ns.sim_run()
        node.qmemory.execute_program(y_m, qubit_mapping=[0, 1])
        ns.sim_run()
        measurement_result = int(y_m.output["M"][0])
        # invert the measurment outputs to correct those!
        if measurement_result == 0:
            measurement_result = 1
        elif measurement_result == 1:
            measurement_result = 0

    elif basis == "X":
        x_m = X_Measurement(num_qubits=2)
        reverse_move_using_CXDirections(x_m, 0, 1)
        node.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[0])
        ns.sim_run()
        node.qmemory.execute_program(x_m, qubit_mapping=[0, 1])
        ns.sim_run()
        measurement_result = int(x_m.output["M"][0])
    else:
        raise RuntimeError("Invalid measurement basis chosen!")

    if measurement_result == 0:
        return 1
    if measurement_result == 1:
        return -1

def create_physical_input_density_matrix(node: Node, input_state: str = "0", iters: int = 10):
    """ Create the input density matrix by doing tomography, to mimic the actual experiment for state preparation! """

    p = [0, 0, 0]
    for _ in range(iters):
        reset_node(node=node)
        physical_cardinal_state_init(node=node, state=input_state)
        res = physical_pauli_measure(node=node, basis="X")
        p[0] += res
    p[0] = p[0]/iters

    for _ in range(iters):
        reset_node(node=node)
        physical_cardinal_state_init(node=node, state=input_state)
        res = physical_pauli_measure(node=node, basis="Y")
        p[1] += res
    p[1] = p[1]/iters

    for _ in range(iters):
        reset_node(node=node)
        physical_cardinal_state_init(node=node, state=input_state)
        res = physical_pauli_measure(node=node, basis="Z")
        p[2] += res
    p[2] = p[2]/iters

    rho = (IDENTITY + p[0]*PAULI_X + p[1]*PAULI_Y + p[2]*PAULI_Z)/2

    return p, rho

def create_physical_output_density_matrix(node: Node, operation: str = "I", iters: int = 10):
    """ Create the output density matrix by doing tomography again, to mimic the actual experiment for an operation. First applies that
    operation and then does state tomography to reconstruct the output state and expectation values vector. """

    if 



def get_input_outputexpectation_values(node: Node, operation: str = "I"):
    pass


def state_tomography_physical():
    pass

def physical_PTM_ideal(op, num_qubits):
    if op == "X":
        pass
    return

def logical_PTM():
    pass


""" Main logical circuit and experiments! """

def logical_state_preparation(theta:float=0, phi:float=0, logical_measure = "Z_L"):
    node_A, node_B = create_two_node_setup()

    xxxx_A = XXXX_Stabilizer(num_qubits=3)
    xxxx_B = XXXX_Stabilizer(num_qubits=3)

    zz_A = ZZ_Stabilizer(num_qubits=3)
    zz_B = ZZ_Stabilizer(num_qubits=3)


    """ Run actual physical sequence """

    physical_init = Physical_Initialization(num_qubits=3)
    node_A.qmemory.execute_program(physical_init, qubit_mapping=[0, 1, 2], theta=theta, phi=phi)
    ns.sim_run()

    electron_1 = node_A.qmemory.peek([0])[0]
    electron_2 = node_B.qmemory.peek([0])[0]
    carbon_1 = node_A.qmemory.peek([1])[0]
    carbon_3 = node_A.qmemory.peek([2])[0]
    carbon_2 = node_B.qmemory.peek([1])[0]
    carbon_4 = node_B.qmemory.peek([2])[0]


    node_A.qmemory.execute_program(zz_A, qubit_mapping=[0, 1, 2])
    ns.sim_run()

    node_B.qmemory.execute_program(zz_B, qubit_mapping=[0, 1, 2])
    ns.sim_run()

    node_A.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[0])
    ns.sim_run()

    node_B.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[0])
    ns.sim_run()

    create_Bell_Pair(node_A=node_A, node_B=node_B)

    node_A.qmemory.execute_program(xxxx_A, qubit_mapping=[0, 1, 2])
    ns.sim_run()
    node_B.qmemory.execute_program(xxxx_B, qubit_mapping=[0, 1, 2])
    ns.sim_run()
    
    measurement_results = [zz_A.output["M"][0], zz_B.output["M"][0], xxxx_A.output["M"][0], xxxx_B.output["M"][0]]
    data_density_matrix = reduced_dm([carbon_1, carbon_2, carbon_3, carbon_4])

    data_measure = "Nothing measured!"

    if logical_measure == "Z_L":
        data_measure = logical_Z_measurement(node_A=node_A, node_B=node_B)
    elif logical_measure == "X_L":
        data_measure = logical_X_measurement(node_A=node_A, node_B=node_B)
    elif logical_measure == "Y_L":
        data_measure = logical_Y_measurement(node_A=node_A, node_B=node_B)
    else:
        raise RuntimeError("Invalid logical measurement basis chosen!")

    return data_density_matrix, measurement_results, data_measure




    node_A.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[0])
    ns.sim_run()
    node_A.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[1])
    ns.sim_run()
    node_A.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[2])
    ns.sim_run()

    node_B.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[0])
    ns.sim_run()
    node_B.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[1])
    ns.sim_run()
    node_B.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[2])
    ns.sim_run()

    return

def logical_Z_measurement(node_A: Node, node_B: Node):
    zl_data_results = [None, None, None, None]
    

    zl_A = ZL_measure(num_qubits=3)
    reverse_move_using_CXDirections(zl_A, 0, 1)
    zl_B = ZL_measure(num_qubits=3)
    reverse_move_using_CXDirections(zl_B, 0, 1)

    node_A.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[0])
    ns.sim_run()
    node_A.qmemory.execute_program(zl_A, qubit_mapping=[0, 1, 2])
    ns.sim_run()
    zl_data_results[0] = zl_A.output["M"][0]
    node_B.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[0])
    ns.sim_run()
    node_B.qmemory.execute_program(zl_B, qubit_mapping=[0, 1, 2])
    ns.sim_run()
    zl_data_results[1] = zl_B.output["M"][0]



    zl_A = ZL_measure(num_qubits=3)
    reverse_move_using_CXDirections(zl_A, 0, 2)
    zl_B = ZL_measure(num_qubits=3)
    reverse_move_using_CXDirections(zl_B, 0, 2)

    node_A.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[0])
    ns.sim_run()
    node_A.qmemory.execute_program(zl_A, qubit_mapping=[0, 1, 2])
    ns.sim_run()
    zl_data_results[2] = zl_A.output["M"][0]
    node_B.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[0])
    ns.sim_run()
    node_B.qmemory.execute_program(zl_B, qubit_mapping=[0, 1, 2])
    ns.sim_run()
    zl_data_results[3] = zl_B.output["M"][0]

    return zl_data_results


def logical_Y_measurement(node_A: Node, node_B: Node):
    node_A_YL = YL_Measurement(num_qubits=3)
    node_B_YL = YL_Measurement(num_qubits=3)

    yl_data_results = []

    node_A.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[0])
    ns.sim_run()
    reverse_move_using_CXDirections(node_A_YL, 0, 1)
    node_A.qmemory.execute_instruction(instr.INSTR_S, qubit_mapping=[0])
    ns.sim_run()
    node_A.qmemory.execute_program(node_A_YL, qubit_mapping=[0, 1, 2])
    ns.sim_run()
    yl_data_results.append(node_A_YL.output["M"][0])

    node_B.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[0])
    ns.sim_run()
    reverse_move_using_CXDirections(node_B_YL, 0, 1)
    node_B.qmemory.execute_instruction(instr.INSTR_H, qubit_mapping=[0])
    ns.sim_run()
    node_B.qmemory.execute_program(node_B_YL, qubit_mapping=[0, 1, 2])
    ns.sim_run()
    yl_data_results.append(node_B_YL.output["M"][0])

    node_A.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[0])
    ns.sim_run()
    reverse_move_using_CXDirections(node_A_YL, 0, 2)
    ns.sim_run()
    node_A.qmemory.execute_program(node_A_YL, qubit_mapping=[0, 1, 2])
    ns.sim_run()
    yl_data_results.append(node_A_YL.output["M"][0])

    node_B.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[0])
    ns.sim_run()
    reverse_move_using_CXDirections(node_B_YL, 0, 2)
    node_B.qmemory.execute_instruction(instr.INSTR_H, qubit_mapping=[0])
    ns.sim_run()
    node_B.qmemory.execute_program(node_B_YL, qubit_mapping=[0, 1, 2])
    ns.sim_run()
    yl_data_results.append(node_B_YL.output["M"][0])

    return yl_data_results


def logical_X_measurement(node_A: Node, node_B: Node):
    xl_data_results = [None, None, None, None]
    

    xl_A = XL_Measurement(num_qubits=3)
    reverse_move_using_CXDirections(xl_A, 0, 1)
    xl_B = XL_Measurement(num_qubits=3)
    reverse_move_using_CXDirections(xl_B, 0, 1)

    node_A.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[0])
    ns.sim_run()
    node_A.qmemory.execute_program(xl_A, qubit_mapping=[0, 1, 2])
    ns.sim_run()
    xl_data_results[0] = xl_A.output["M"][0]
    node_B.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[0])
    ns.sim_run()
    node_B.qmemory.execute_program(xl_B, qubit_mapping=[0, 1, 2])
    ns.sim_run()
    xl_data_results[1] = xl_B.output["M"][0]



    xl_A = XL_Measurement(num_qubits=3)
    reverse_move_using_CXDirections(xl_A, 0, 2)
    xl_B = XL_Measurement(num_qubits=3)
    reverse_move_using_CXDirections(xl_B, 0, 2)

    node_A.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[0])
    ns.sim_run()
    node_A.qmemory.execute_program(xl_A, qubit_mapping=[0, 1, 2])
    ns.sim_run()
    xl_data_results[2] = xl_A.output["M"][0]
    node_B.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[0])
    ns.sim_run()
    node_B.qmemory.execute_program(xl_B, qubit_mapping=[0, 1, 2])
    ns.sim_run()
    xl_data_results[3] = xl_B.output["M"][0]

    return xl_data_results


def create_input_output_matrix(iters: int = 10):
    """ Components creation"""
    io_matrix = np.zeros((16,16))
    serial = range(16)
    input_states = [''.join(comb) for comb in product(['0','1'], repeat=4)]

    input_states = dict(zip(input_states, serial))


    for input_state in input_states.keys():
        for iteration in range(iters):
            node_A, node_B = create_two_node_setup()
            reset_node(node=node_A)
            reset_node(node=node_B)
            physical_initialization_cardinal_states(node_A=node_A, node_B=node_B, command = input_state)
            data_measure = logical_Z_measurement(node_A=node_A, node_B=node_B)
            data_meas = ''.join([''.join(str(val)) for val in data_measure])
            reset_node(node=node_A)
            reset_node(node=node_B)

            # print(data_measure, input_state)
            # print(ns.sim_time())

            io_matrix[input_states[f"{data_meas}"], input_states[f"{input_state}"]] += 1
    
    io_matrix = np.around(io_matrix/iters, decimals=2)


    fig = plt.figure(figsize=(10, 10))
    fig.set_facecolor("w")
    ax = fig.add_subplot()

    plot = ax.matshow(io_matrix, cmap=plt.cm.Blues)
    fig.colorbar(plot, ax=ax)

    state_label = [f"|{str}⟩" for str in list(input_states.keys())]
    meas_strings = [f'({str[0]}1,{str[1]}1,{str[2]}1,{str[3]}1,)' for str in [''.join(comb) for comb in product(['+','-'], repeat=4)]]

    plt.xticks(range(16), state_label, rotation=60)
    plt.yticks(range(16), meas_strings)

    for i in range(16):
        for j in range(16):
            c = io_matrix[j,i]
            ax.text(i, j, str(c), va='center', ha='center')

    plt.savefig(f'io_matrix_{timestr}.pdf')

    return io_matrix

