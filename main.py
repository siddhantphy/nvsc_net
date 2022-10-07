# Utilities imports
from itertools import product
import logging
import os
import time

# Mathematics imports
import numpy as np

# Plotting imports
import matplotlib.pyplot as plt

# Netsquid imports
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

# Qiskit imports
import qiskit

# Local imports
from basic_operations import *
from q_programs import *
from native_gates_and_parameters import *
from network_model import *


###############################################################
###############################################################
###############################################################

timestr = time.strftime("%Y%m%d-%H%M%S")
# ns.logger.setLevel(logging.DEBUG)




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

    for input_state in input_states.keys():
        for iteration in range(iters):
            node_A, node_B = create_system()
            reset_nodes(node_A=node_A, node_B=node_B)
            physical_initialization_cardinal_states(node_A=node_A, node_B=node_B, command = input_state)
            data_measure = logical_Z_measurement(node_A=node_A, node_B=node_B)
            data_meas = ''.join([''.join(str(val)) for val in data_measure])
            reset_nodes(node_A=node_A, node_B=node_B)

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


""" 
    Plotting and results
"""
def plot_logical_post_theta(iters:int=1, steps:int=10):
    post_selection = []
    for theta in np.arange(0,np.pi, np.pi/steps):
        sum=0
        for iter in range(iters):
            rho, meas_results, _ = logical_state_preparation(theta=theta, phi=0)
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
    plt.savefig(f'Post selection_{timestr}.pdf')

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


iters = 50
steps = 25


# plot_logical_post_theta(iters=iters, steps=steps)
# logical_state_fidelity_theta(iters=iters, steps=steps, logical_measure="X_L")
# logical_state_fidelity_phi(iters=iters, steps=steps, logical_measure="Z_L")

create_input_output_matrix(iters=iters)

""" Trash data from before"""
# [0.48333333333333334, 0.5333333333333333, 0.49666666666666665, 0.45666666666666667, 0.43333333333333335, 0.45666666666666667, 0.47333333333333333, 0.39, 0.37333333333333335, 0.31, 0.32666666666666666, 0.2966666666666667, 0.24333333333333335, 0.3, 0.22333333333333333, 0.25333333333333335, 0.28, 0.2633333333333333, 0.2833333333333333, 0.30666666666666664, 0.31, 0.37, 0.37666666666666665, 0.4066666666666667, 0.38666666666666666, 0.4, 0.45666666666666667, 0.4633333333333333, 0.47, 0.46]