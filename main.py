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

# create_input_output_matrix(iters=iters)
# print(np.outer(KET_i_PLUS, KET_PLUS)+np.outer(KET_i_MINUS, KET_MINUS))
# print(np.array([[1,1],[1,-1]])@np.array([[1,0],[0,1j]]))
# print(np.array([[1,0],[0,1j]])@np.array([[1,1],[1,-1]]))

node = create_physical_qubit_single_node_setup(no_noise=True)
print(create_physical_input_density_matrix(node=node, input_state="+i", iters=200))
# sumx=0

# for i in range(100):
#     physical_cardinal_state_init(node=node, state="0")
#     # print(reduced_dm([node.qmemory.peek([1])[0]]))
#     res = physical_pauli_measure(node=node, basis="Z")
#     sumx = sumx+res
# # print(reduced_dm([node.qmemory.peek([1])[0]]))
# print(sumx/100)


""" Trash data from before"""
# [0.48333333333333334, 0.5333333333333333, 0.49666666666666665, 0.45666666666666667, 0.43333333333333335, 0.45666666666666667, 0.47333333333333333, 0.39, 0.37333333333333335, 0.31, 0.32666666666666666, 0.2966666666666667, 0.24333333333333335, 0.3, 0.22333333333333333, 0.25333333333333335, 0.28, 0.2633333333333333, 0.2833333333333333, 0.30666666666666664, 0.31, 0.37, 0.37666666666666665, 0.4066666666666667, 0.38666666666666666, 0.4, 0.45666666666666667, 0.4633333333333333, 0.47, 0.46]
