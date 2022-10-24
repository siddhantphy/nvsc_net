# Utilities imports
from itertools import product
import itertools
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


""" To calculate the physical gate fidelity! """
def get_the_physical_gate_fidelity(depolar_rates: list, operation: str = "NA", iterations: int = 10):
    fid_gate = []
    for depolar in depolar_rates:
    # Parameters dictionary for properties
        parameters = {"electron_T1": np.inf, "electron_T2": np.inf, "carbon_T1": np.inf, "carbon_T2": np.inf, "electron_init_depolar_prob": depolar,
        "electron_single_qubit_depolar_prob": depolar, "carbon_init_depolar_prob": depolar, "carbon_z_rot_depolar_prob": depolar,
        "ec_gate_depolar_prob": depolar}
        fidelity = 0
        for i in range(iterations):
            node_noiseless = create_physical_qubit_single_node_setup(no_noise=True)
            noiseless = np.array(create_analytical_physical_PTM(node=node_noiseless, operation="Rx_pi"))

            node_noisy = create_physical_qubit_single_node_setup(no_noise=False, parameters=parameters)
            noisy = np.array(create_analytical_physical_PTM(node=node_noisy, operation="Rx_pi"))

            fidelity += (np.trace(noiseless.conj().T @ noisy)+2)/6
        fidelity = fidelity/iterations
        fid_gate.append(fidelity)

    print(fid_gate)


def get_the_logical_gate_fidelity(depolar_rates: list, operation: str = "NA", iterations: int = 10, post_select: bool = False):
    fid_gate = []
    trashed = 0
    for depolar in depolar_rates:
    # Parameters dictionary for properties
        parameters = {"electron_T1": np.inf, "electron_T2": np.inf, "carbon_T1": np.inf, "carbon_T2": np.inf, "electron_init_depolar_prob": depolar,
        "electron_single_qubit_depolar_prob": depolar, "carbon_init_depolar_prob": depolar, "carbon_z_rot_depolar_prob": depolar,
        "ec_gate_depolar_prob": depolar}
        fidelity = 0
        node_A, node_B = create_two_node_setup(no_noise=True)
        lptm_noiseless = np.array(create_analytical_logical_PTM(node_A=node_A, node_B=node_B, operation=operation,iterations=iterations, post_select=post_select))

        node_A_noisy, node_B_noisy = create_two_node_setup(no_noise=False, parameters=parameters)
        lptm_noisy = np.array(create_analytical_logical_PTM(node_A=node_A_noisy, node_B=node_B_noisy, operation=operation,iterations=iterations, post_select=post_select))
        
        fidelity = (np.trace(lptm_noiseless.conj().T @ lptm_noisy)+2)/6

        fid_gate.append(fidelity)

    print(fid_gate)

