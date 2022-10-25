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


iters = 100
steps = 25


# plot_logical_post_theta(iters=iters, steps=steps)
# logical_state_fidelity_theta(iters=iters, steps=steps, logical_measure="X_L")
# logical_state_fidelity_phi(iters=iters, steps=steps, logical_measure="Z_L")

get_the_physical_gate_fidelity([0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.003], operation="Rx_pi", iterations=iters)
# get_the_logical_gate_fidelity([0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.003], operation="Rx_pi", iterations=iters, post_select=True)



""" Simulated Data! """

# FOR RX_PI [Circuit noise on gates and initialization!]
# Rates [0.005, 0.01, 0.015, 0.02, 0.1, 0.2, 0.4]
# POST SELECTED FIDELITIES [0.956943433721451, 0.9382038293130955, 0.9211240368948278, 0.9354005636070851, 0.6338118580765639, 0.5200757575757575, 0.48983134920634924]
# WITHOUT POST SELECTED [0.9233333333333332, 0.8583333333333333, 0.82, 0.7633333333333333, 0.5283333333333333, 0.5083333333333333, 0.49499999999999994]
# PHYSICAL GATE FIDELITIES [0.9966666666666667, 0.9799999999999999, 0.9733333333333334, 0.9633333333333328, 0.8933333333333331, 0.733333333333333, 0.6383333333333332]


# FOR RZ_PI [Circuit noise on gates and initialization!]
# Rates [0.001, 0.002, 0.003, 0.004, 0.05, 0.006, 0.007, 0.008, 0.009, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
# POST SELECTED FIDELITIES [0.9683333333333333, 0.9433333333333332, 0.9466666666666667, 0.9183333333333333, 0.6366666666666666, 0.9216666666666665, 0.9083333333333332, 0.8933333333333332, 0.8816666666666667, 0.8816666666666665, 0.8516666666666666, 0.8266666666666665, 0.8133333333333334, 0.7933333333333333, 0.7766666666666667, 0.725, 0.6866666666666665, 0.68, 0.6616666666666666, 0.6366666666666666, 0.64]
# PHYSICAL GATE FIDELITIES [0.9983333333333333, 1.0, 0.9949999999999999, 0.9933333333333333, 0.9416666666666668, 0.9933333333333333, 0.9916666666666666, 0.9916666666666667, 0.995, 0.9883333333333332, 0.9766666666666665, 0.983333333333333, 0.9649999999999995, 0.9716666666666666, 0.9849999999999998, 0.9599999999999999, 0.9499999999999997, 0.9416666666666668, 0.9466666666666664, 0.925, 0.9416666666666665]


# FOR RX_PI/2 [Only circuit noise!]
# POST SELECTED FIDELITIES [0.9999999999999999, 1.0, 0.9999999999999999, 1.0, 0.9970959595959595, 0.9971264367816092, 0.9926251161046773, 0.9986338797814206, 1.0, 0.995714007403078, 0.9983660130718953, 0.9936456063907043]
# WITHOUT POST SELECTED
# PHYSICAL GATE FIDELITIES 