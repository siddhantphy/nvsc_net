import logging
from traceback import print_tb
import numpy as np
import netsquid as ns
import netsquid.qubits.operators as ops
from netsquid_nv.nv_center import NVQuantumProcessor
from netsquid_nv.magic_distributor import NVDoubleClickMagicDistributor, NVSingleClickMagicDistributor
from netsquid.nodes import Node
from netsquid.protocols import Protocol
from netsquid.components.qprogram import *
from netsquid.qubits.qubitapi import reduced_dm, assign_qstate
import netsquid.components.instructions as instr
from netsquid.components.qprocessor import QuantumProcessor, PhysicalInstruction
from netsquid.qubits.ketstates import BellIndex
from netsquid.components.instructions import INSTR_X, INSTR_Y, INSTR_Z, INSTR_ROT_X, INSTR_ROT_Y, INSTR_ROT_Z, INSTR_H,\
    INSTR_MEASURE, INSTR_SWAP, INSTR_INIT, INSTR_CXDIR, INSTR_EMIT



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
        self.apply(instr.INSTR_ROT_Y, c1, angle=theta)
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

class Tomography():
    pass


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
    psi_logical = (logical_0 * (np.cos(theta/2))**2 + logical_1 * (np.sin(theta/2))**2)/(np.sqrt((np.cos(theta/2))**4+(np.sin(theta/2))**4))
    rho_logical = np.outer(psi_logical, psi_logical)
    return rho_logical

""" Main logical circuit and experiment! """

def logical_state_preparation(theta:float=0, phi:float=0):
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
    node_A.qmemory.execute_program(physical_init, qubit_mapping=[0, 1, 2], theta=0, phi=0)
    ns.sim_run()

    create_Bell_Pair(node_A=node_A, node_B=node_B)

    electron_1 = node_A.qmemory.peek([0])[0]
    electron_2 = node_B.qmemory.peek([0])[0]
    carbon_1 = node_A.qmemory.peek([1])[0]
    carbon_3 = node_A.qmemory.peek([2])[0]
    carbon_2 = node_B.qmemory.peek([1])[0]
    carbon_4 = node_B.qmemory.peek([2])[0]

    # print(reduced_dm([electron_1, electron_2]))
    # print(c1.qstate)

    node_A.qmemory.execute_instruction(instr.INSTR_INIT, qubit_mapping=[0])
    ns.sim_run()

    node_A.qmemory.execute_program(zz_A, qubit_mapping=[0, 1, 2])
    ns.sim_run()

    node_B.qmemory.execute_program(zz_B, qubit_mapping=[0, 1, 2])
    ns.sim_run()

    node_A.qmemory.execute_program(xxxx_A, qubit_mapping=[0, 1, 2])
    ns.sim_run()
    node_B.qmemory.execute_program(xxxx_B, qubit_mapping=[0, 1, 2])
    ns.sim_run()

    measurement_results = [zz_A.output["M"], zz_B.output["M"], xxxx_A.output["M"], xxxx_B.output["M"]]
    data_density_matrix = reduced_dm([carbon_1, carbon_2, carbon_3, carbon_4])

    return data_density_matrix, measurement_results

"""
############################################################################
############################################################################
############################################################################
"""

"""
    Main script
"""

rho, meas_results = logical_state_preparation(theta=1, phi=1)
theoretical_rho = create_theoretical_rho(theta=1, phi=1)

print(np.trace(np.dot(theoretical_rho, rho)))
# print(rho)