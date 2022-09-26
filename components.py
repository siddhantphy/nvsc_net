from itertools import product
import logging
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import netsquid as ns
import netsquid.qubits.operators as ops
from netsquid_nv.nv_center import NVQuantumProcessor
from netsquid_nv.move_circuits import reverse_move_using_CXDirections # Note that this results into a Hadamrd being applied on the electron, so there is a change of basis
from netsquid_nv.magic_distributor import NVDoubleClickMagicDistributor, NVSingleClickMagicDistributor
from netsquid.nodes import Node
from netsquid.protocols import Protocol
from netsquid.components.qprogram import *
from netsquid.qubits.qubitapi import reduced_dm, assign_qstate
import netsquid.components.instructions as instr
from netsquid.components.qprocessor import QuantumProcessor, PhysicalInstruction
from netsquid.qubits.ketstates import BellIndex
from netsquid.components.instructions import INSTR_X, INSTR_Y, INSTR_Z, INSTR_ROT_X, INSTR_ROT_Y, INSTR_ROT_Z, INSTR_H, INSTR_S,\
    INSTR_MEASURE, INSTR_SWAP, INSTR_INIT, INSTR_CXDIR, INSTR_EMIT


timestr = time.strftime("%Y%m%d-%H%M%S")
# ns.logger.setLevel(logging.DEBUG)


"""
    Classes and global functions for operations
"""

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


class Phase_Correction(QuantumProgram):
    default_num_qubits = 3
    def program(self):
        e1, c1, c3 = self.get_qubit_indices(3)
        self.apply(instr.INSTR_Z, e1)
        yield self.run()

class Rotate_Bell_Pair(QuantumProgram):
    default_num_qubits = 3

    def program(self):
        e1, c1, c3 = self.get_qubit_indices(3)
        self.apply(instr.INSTR_X, e1)
        yield self.run()


class Logical_Initialization(QuantumProgram):
    default_num_qubits = 3

    def program(self, theta: float=0, phi: float=0):

        e1, c1, c3 = self.get_qubit_indices(3)
        self.apply(instr.INSTR_ROT_Y, c1, angle=theta)
        self.apply(instr.INSTR_ROT_Y, c3, angle=theta)
        self.apply(instr.INSTR_ROT_Z, c3, angle=phi)
        yield self.run()

class XXXX_Stabilizer(QuantumProgram):
    default_num_qubits = 3

    def program(self):
        e1, c1, c3 = self.get_qubit_indices(3)
        self.apply(instr.INSTR_CX, [e1, c1])
        self.apply(instr.INSTR_CX, [e1, c3])
        self.apply(instr.INSTR_H, e1)
        self.apply(instr.INSTR_MEASURE, e1, output_key="M")
        yield self.run()

class ZZ_Stabilizer(QuantumProgram):
    default_num_qubits = 3

    def program(self):
        e1, c1, c3 = self.get_qubit_indices(3)
        self.apply(instr.INSTR_H, e1)
        self.apply(instr.INSTR_CZ, [e1, c1]) # Perform Controlled Phase
        self.apply(instr.INSTR_CZ, [e1, c3])
        self.apply(instr.INSTR_H, e1)
        self.apply(instr.INSTR_MEASURE, e1, output_key="M")
        yield self.run()

class ZL_Measurement(QuantumProgram):
    default_num_qubits = 3

    def program(self):
        e1, c1, c3 = self.get_qubit_indices(3)
        self.apply(instr.INSTR_MEASURE, e1, output_key="M")
        yield self.run()

class XL_Measurement(QuantumProgram):
    default_num_qubits = 3

    def program(self):
        e1, c1, c3 = self.get_qubit_indices(3)
        self.apply(instr.INSTR_MEASURE, e1, output_key="M")
        yield self.run()

class YL_Measurement(QuantumProgram):
    default_num_qubits = 3

    def program(self):
        e1, c1, c3 = self.get_qubit_indices(3)
        self.apply(instr.INSTR_MEASURE, e1, output_key="M")
        yield self.run()



"""
    Adding native gates as instructions
"""
def add_native_gates(NV_Center: NVQuantumProcessor):
    physical_instructions = []

    # Add all arbitrary rotations on the carbon
    physical_instructions.append(PhysicalInstruction(instr.INSTR_ROT_X, 
                                                    parallel=False,
                                                    topology=NV_Center.carbon_positions,
                                                    q_noise_model=NV_Center.models["carbon_init_noise"],
                                                    apply_q_noise_after=True,
                                                    duration=NV_Center.properties["carbon_z_rot_duration"]))

    physical_instructions.append(PhysicalInstruction(instr.INSTR_ROT_Y, 
                                                    parallel=False,
                                                    topology=NV_Center.carbon_positions,
                                                    q_noise_model=NV_Center.models["carbon_init_noise"],
                                                    apply_q_noise_after=True,
                                                    duration=NV_Center.properties["carbon_z_rot_duration"]))
    
    physical_instructions.append(PhysicalInstruction(instr.INSTR_ROT_Z, 
                                                    parallel=False,
                                                    topology=NV_Center.carbon_positions,
                                                    q_noise_model=NV_Center.models["carbon_init_noise"],
                                                    apply_q_noise_after=True,
                                                    duration=NV_Center.properties["carbon_z_rot_duration"]))
    
    physical_instructions.append(PhysicalInstruction(instr.INSTR_S, 
                                                    parallel=False,
                                                    topology=[NV_Center.electron_position],
                                                    q_noise_model=NV_Center.models["electron_single_qubit_noise"],
                                                    duration=NV_Center.properties["electron_single_qubit_duration"]))
    

    physical_instructions.append(
            PhysicalInstruction(instr.INSTR_CX,
                                parallel=False,
                                topology=[(0, 1), (0, 2)],
                                q_noise_model=NV_Center.models["ec_noise"],
                                apply_q_noise_after=True,
                                duration=NV_Center.properties["ec_two_qubit_gate_duration"]))
    
    physical_instructions.append(
            PhysicalInstruction(instr.INSTR_CZ,
                                parallel=False,
                                topology=[(0, 1), (0, 2)],
                                q_noise_model=NV_Center.models["ec_noise"],
                                apply_q_noise_after=True,
                                duration=NV_Center.properties["ec_two_qubit_gate_duration"]))
    


    for instruction in physical_instructions:
            NV_Center.add_physical_instruction(instruction)
    return

def create_theoretical_rho(theta:float=0, phi:float=0):
    ket_0 = np.array([[1], [0]])
    ket_1 = np.array([[0], [1]])
    logical_0 = (np.kron(np.kron(ket_0, ket_0) , np.kron(ket_0, ket_0)) + np.kron(np.kron(ket_1, ket_1) , np.kron(ket_1, ket_1)))/np.sqrt(2)
    logical_1 = (np.kron(np.kron(ket_0, ket_1) , np.kron(ket_0, ket_1)) + np.kron(np.kron(ket_1, ket_0) , np.kron(ket_1, ket_0)))/np.sqrt(2)
    psi_logical = (logical_0 * (np.cos(theta/2))**2 + logical_1 * (np.exp(-1j*phi)*np.sin(theta/2))**2)/(np.sqrt((np.cos(theta/2))**4+(np.sin(theta/2))**4))
    rho_logical = np.outer(psi_logical, psi_logical)

    z_L = np.outer(logical_0, logical_0) - np.outer(logical_1, logical_1)
    x_l = np.outer(logical_0, logical_1) + np.outer(logical_1, logical_0)
    y_L = -1j * np.outer(logical_0, logical_1) + 1j* np.outer(logical_1, logical_0)
    return rho_logical, x_l, y_L, z_L


""" Main logical circuit and experiment! """

def logical_state_preparation(theta:float=0, phi:float=0, logical_measure = "Z_L"):
    """ Components creation"""
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


    """ Logic of the tasks"""
    add_native_gates(processor_A)
    add_native_gates(processor_B)

    xxxx_A = XXXX_Stabilizer(num_qubits=3)
    xxxx_B = XXXX_Stabilizer(num_qubits=3)

    zz_A = ZZ_Stabilizer(num_qubits=3)
    zz_B = ZZ_Stabilizer(num_qubits=3)


    """ Run actual physical sequence """

    physical_init = Logical_Initialization(num_qubits=3)
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

def physical_initialization(node_A: Node, node_B: Node, command:str = "0000"):
    if command[0] == '0':
        node_A.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[1])
        ns.sim_run()
    elif command[0] == '1':
        node_A.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[1])
        ns.sim_run()
        node_A.qmemory.execute_instruction(instr.INSTR_ROT_X, qubit_mapping=[1], angle=np.pi)
        ns.sim_run()
    
    if command[1] == '0':
        node_B.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[1])
        ns.sim_run()
    elif command[1] == '1':
        node_B.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[1])
        ns.sim_run()
        node_B.qmemory.execute_instruction(instr.INSTR_ROT_X, qubit_mapping=[1], angle=np.pi)
        ns.sim_run()

    if command[2] == '0':
        node_A.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[2])
        ns.sim_run()
    elif command[2] == '1':
        node_A.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[2])
        ns.sim_run()
        node_A.qmemory.execute_instruction(instr.INSTR_ROT_X, qubit_mapping=[2], angle=np.pi)
        ns.sim_run()

    if command[3] == '0':
        node_B.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[1])
        ns.sim_run()
    elif command[3] == '1':
        node_B.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[1])
        ns.sim_run()
        node_B.qmemory.execute_instruction(instr.INSTR_ROT_X, qubit_mapping=[2], angle=np.pi)
        ns.sim_run()

def reset_nodes(node_A: Node, node_B: Node):
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
    node_A_ZL = ZL_Measurement(num_qubits=3)
    node_B_ZL = ZL_Measurement(num_qubits=3)

    zl_data_results = []

    node_A.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[0])
    ns.sim_run()
    reverse_move_using_CXDirections(node_A_ZL, 0, 1)
    node_A.qmemory.execute_instruction(instr.INSTR_H, qubit_mapping=[0])
    ns.sim_run()
    node_A.qmemory.execute_program(node_A_ZL, qubit_mapping=[0, 1, 2])
    ns.sim_run()
    zl_data_results.append(node_A_ZL.output["M"][0])

    node_B.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[0])
    ns.sim_run()
    reverse_move_using_CXDirections(node_B_ZL, 0, 1)
    node_B.qmemory.execute_instruction(instr.INSTR_H, qubit_mapping=[0])
    ns.sim_run()
    node_B.qmemory.execute_program(node_B_ZL, qubit_mapping=[0, 1, 2])
    ns.sim_run()
    zl_data_results.append(node_B_ZL.output["M"][0])

    node_A.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[0])
    ns.sim_run()
    reverse_move_using_CXDirections(node_A_ZL, 0, 2)
    node_A.qmemory.execute_instruction(instr.INSTR_H, qubit_mapping=[0])
    ns.sim_run()
    node_A.qmemory.execute_program(node_A_ZL, qubit_mapping=[0, 1, 2])
    ns.sim_run()
    zl_data_results.append(node_A_ZL.output["M"][0])

    node_B.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[0])
    ns.sim_run()
    reverse_move_using_CXDirections(node_B_ZL, 0, 2)
    node_B.qmemory.execute_instruction(instr.INSTR_H, qubit_mapping=[0])
    ns.sim_run()
    node_B.qmemory.execute_program(node_B_ZL, qubit_mapping=[0, 1, 2])
    ns.sim_run()
    zl_data_results.append(node_B_ZL.output["M"][0])

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
    node_A_XL = XL_Measurement(num_qubits=3)
    node_B_XL = XL_Measurement(num_qubits=3)

    xl_data_results = []

    node_A.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[0])
    ns.sim_run()
    reverse_move_using_CXDirections(node_A_XL, 0, 1)
    ns.sim_run()
    node_A.qmemory.execute_program(node_A_XL, qubit_mapping=[0, 1, 2])
    ns.sim_run()
    xl_data_results.append(node_A_XL.output["M"][0])

    node_B.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[0])
    ns.sim_run()
    reverse_move_using_CXDirections(node_B_XL, 0, 1)
    ns.sim_run()
    node_B.qmemory.execute_program(node_B_XL, qubit_mapping=[0, 1, 2])
    ns.sim_run()
    xl_data_results.append(node_B_XL.output["M"][0])

    node_A.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[0])
    ns.sim_run()
    reverse_move_using_CXDirections(node_A_XL, 0, 2)
    ns.sim_run()
    node_A.qmemory.execute_program(node_A_XL, qubit_mapping=[0, 1, 2])
    ns.sim_run()
    xl_data_results.append(node_A_XL.output["M"][0])

    node_B.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[0])
    ns.sim_run()
    reverse_move_using_CXDirections(node_B_XL, 0, 2)
    ns.sim_run()
    node_B.qmemory.execute_program(node_B_XL, qubit_mapping=[0, 1, 2])
    ns.sim_run()
    xl_data_results.append(node_B_XL.output["M"][0])

    return xl_data_results


def create_input_output_matrix(iters: int = 10):
    """ Components creation"""
    io_matrix = np.zeros((16,16))
    input_states = []
    input_states = [''.join(comb) for comb in product(['0','1'], repeat=4)]
    meas_strings = [''.join(comb) for comb in product(['+','-'], repeat=4)]


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

    for input_state in input_states:
        for iteration in range(iters):
            physical_initialization(node_A=node_A, node_B=node_B, command = input_state)
            data_measure = logical_Z_measurement(node_A=node_A, node_B=node_B)

            print(data_measure)

    return io_matrix

""" 
    Plotting and results
"""
def plot_logical_post_theta(iters:int=1, steps:int=10):
    post_selection = []
    for theta in np.arange(0,np.pi, np.pi/steps):
        sum=0
        for iter in range(iters):
            rho, meas_results = logical_state_preparation(theta=theta, phi=0)
            # theoretical_rho, x_l, y_l, z_l = create_theoretical_rho(theta=theta, phi=0)
            print(iter)
            os.system('cls||clear')
            if meas_results[0]==0 and meas_results[1]==0:
                if meas_results[2]==meas_results[3]:
                    sum = sum+1 
        post_selection.append(sum/iters)
    print(post_selection)

    theta = np.arange(0,np.pi, np.pi/steps)
    theory = 0.5* ((np.cos(theta/2))**4+(np.sin(theta/2))**4)

    fig = plt.figure(figsize=(10,5))
    fig.set_facecolor("w")
    ax1 = fig.add_subplot()
    ax1.set_title('Post selection fraction for logical initialization')
    ax1.set_ylabel('P(θ)')
    ax1.set_xlabel('θ in radians')
    plt.grid()

    plt.plot(theta,post_selection,'o', label='post_selection')
    plt.plot(theta,theory,'r', label='post_selection')
    plt.savefig('Post selection.pdf')

    return


def logical_state_fidelity_theta(iters:int=1, steps:int=10, logical_measure="Z_L"):
    o_L_avg = []
    for theta in np.arange(0,np.pi, np.pi/steps):
        sum=0
        overlap = 0
        for iter in range(iters):
            rho, meas_results, data_measure = logical_state_preparation(theta=theta, phi=0, logical_measure=logical_measure)
            theoretical_rho, x_l, y_l, z_l = create_theoretical_rho(theta=theta, phi=0)
            print(iter)
            os.system('cls||clear')
            if meas_results[0]==0 and meas_results[1]==0:
                if meas_results[2]==meas_results[3]:
                    sum = sum+1
                    if logical_measure == "Z_L":
                        new_overlap =  round(np.trace(np.matmul(z_l,rho)).real, 12)
                        overlap +=new_overlap
                    elif logical_measure == "X_L":
                        new_overlap =  round(np.trace(np.matmul(x_l,rho)).real, 12)
                        overlap +=new_overlap
                    elif logical_measure == "Y_L":
                        new_overlap =  round(np.trace(np.matmul(y_l,rho)).real, 12)
                        overlap +=new_overlap
                    print(new_overlap)
        overlap = overlap/sum
        o_L_avg.append(overlap)

    print(o_L_avg)
    theta = np.arange(0,np.pi, np.pi/steps)

    if logical_measure == "Z_L":
        theory = ((np.cos(theta/2))**4 - (np.sin(theta/2))**4)/((np.cos(theta/2))**4 + (np.sin(theta/2))**4)
    elif logical_measure == "X_L":
        theory = (2 * (np.cos(theta/2))**2 * (np.sin(theta/2))**2)/((np.cos(theta/2))**4 + (np.sin(theta/2))**4)
    elif logical_measure == "Y_L":
        theory = 0 * theta

    fig = plt.figure(figsize=(10,5))
    fig.set_facecolor("w")
    ax1 = fig.add_subplot()
    ax1.set_title(f'Logical {logical_measure} initialization')
    ax1.set_ylabel(f'<{logical_measure}>')
    ax1.set_xlabel('θ in radians')
    plt.grid()

    plt.plot(theta,o_L_avg,'o', label=f'{logical_measure} assignment data')
    plt.plot(theta,theory,'r', label=f'{logical_measure} assignment theory')
    plt.savefig(f'{logical_measure}_theta_assignment_{timestr}.pdf')


def logical_state_fidelity_phi(iters:int=1, steps:int=10, logical_measure="Z_L"):
    o_L_avg = []
    for phi in np.arange(0,2 * np.pi, np.pi/steps):
        sum=0
        overlap = 0
        for iter in range(iters):
            rho, meas_results, data_measure = logical_state_preparation(theta=np.pi/2, phi=phi, logical_measure=logical_measure)
            theoretical_rho, x_l, y_l, z_l = create_theoretical_rho(theta=np.pi/2, phi=phi)
            print(iter)
            os.system('cls||clear')
            if meas_results[0]==0 and meas_results[1]==0:
                if meas_results[2]==meas_results[3]:
                    sum = sum+1
                    if logical_measure == "Z_L":
                        new_overlap =  round(np.trace(np.matmul(z_l,rho)).real, 12)
                        overlap +=new_overlap
                    elif logical_measure == "X_L":
                        new_overlap =  round(np.trace(np.matmul(x_l,rho)).real, 12)
                        overlap +=new_overlap
                    elif logical_measure == "Y_L":
                        new_overlap =  round(np.trace(np.matmul(y_l,rho)).real, 12)
                        overlap +=new_overlap
                    print(new_overlap)
        overlap = overlap/sum
        o_L_avg.append(overlap)

    print(o_L_avg)
    phi = np.arange(0, 2 * np.pi, np.pi/steps)

    if logical_measure == "Z_L":
        theory = 0 * phi
    elif logical_measure == "X_L":
        theory = np.cos(phi)
    elif logical_measure == "Y_L":
        theory = np.sin(phi)

    fig = plt.figure(figsize=(10,5))
    fig.set_facecolor("w")
    ax1 = fig.add_subplot()
    ax1.set_title(f'Logical {logical_measure} initialization')
    ax1.set_ylabel(f'<{logical_measure}>')
    ax1.set_xlabel('ϕ in radians')
    plt.grid()

    plt.plot(phi,o_L_avg,'o', label=f'{logical_measure} assignment data')
    plt.plot(phi,theory,'r', label=f'{logical_measure} assignment theory')
    plt.savefig(f'{logical_measure}_phi_assignment_{timestr}.pdf')


"""
############################################################################
############################################################################
############################################################################
"""

"""
    Main script
"""


iters = 25
steps = 20

# logical_state_fidelity_theta(iters=iters, steps=steps, logical_measure="Z_L")
# logical_state_fidelity_phi(iters=iters, steps=steps, logical_measure="Y_L")

print(create_input_output_matrix(iters=10))

""" Trash data from before"""
# [0.48333333333333334, 0.5333333333333333, 0.49666666666666665, 0.45666666666666667, 0.43333333333333335, 0.45666666666666667, 0.47333333333333333, 0.39, 0.37333333333333335, 0.31, 0.32666666666666666, 0.2966666666666667, 0.24333333333333335, 0.3, 0.22333333333333333, 0.25333333333333335, 0.28, 0.2633333333333333, 0.2833333333333333, 0.30666666666666664, 0.31, 0.37, 0.37666666666666665, 0.4066666666666667, 0.38666666666666666, 0.4, 0.45666666666666667, 0.4633333333333333, 0.47, 0.46]


 