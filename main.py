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
# import qiskit

# Local imports
from basic_operations import *
from q_programs import *
from native_gates_and_parameters import *
from network_model import *
from plot import *


###############################################################
###############################################################
###############################################################

timestr = time.strftime("%Y%m%d-%H%M%S")
# ns.logger.setLevel(logging.DEBUG)







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

# get_the_physical_gate_fidelity([0.01, 0.02, 0.05, 0.07, 0.1, 0.12, 0.15], operation="T", iterations=iters)
get_the_logical_gate_fidelity([0.01, 0.02, 0.05, 0.07, 0.1, 0.12, 0.15], operation="Rx_pi", iterations=iters)

# node_A, node_B = create_two_node_setup()
# perform_first_stabilizer_measurements(node_A=node_A, node_B=node_B, state="0_L")
# print(get_instantaneous_data_qubit_density_matrix([node_A, node_B]))