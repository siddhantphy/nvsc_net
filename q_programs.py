import netsquid.components.instructions as instr
from netsquid.components.qprogram import QuantumProgram


"""
    Classes and quantum programs for node operations and initialization
"""

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


class Physical_Initialization(QuantumProgram):
    default_num_qubits = 3

    def program(self, theta: float=0, phi: float=0):

        e1, c1, c3 = self.get_qubit_indices(3)
        self.apply(instr.INSTR_ROT_Y, c1, angle=theta)
        self.apply(instr.INSTR_ROT_Y, c3, angle=theta)
        self.apply(instr.INSTR_ROT_Z, c3, angle=phi)
        yield self.run()

""" Quantum programs for stabilizer measurements. """

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

""" Quantum programs for logical state measurement. """

class ZL_measure(QuantumProgram):
    default_num_qubits = 3

    def program(self):
        e1, c1, c3 = self.get_qubit_indices(3)
        self.apply(instr.INSTR_H, e1)
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

""" Quantum programs for physical state measurement. """

class X_Measurement(QuantumProgram):
    default_num_qubits = 2

    def program(self):
        e, c = self.get_qubit_indices(2)
        self.apply(instr.INSTR_MEASURE, e, output_key="M")
        yield self.run()

class Y_Measurement(QuantumProgram):
    default_num_qubits = 2

    def program(self):
        e, c = self.get_qubit_indices(2)
        self.apply(instr.INSTR_H, e)
        self.apply(instr.INSTR_S, e)
        self.apply(instr.INSTR_H, e)
        self.apply(instr.INSTR_MEASURE, e, output_key="M")
        yield self.run()

class Z_Measurement(QuantumProgram):
    default_num_qubits = 2

    def program(self):
        e, c = self.get_qubit_indices(2)
        self.apply(instr.INSTR_H, e)
        self.apply(instr.INSTR_MEASURE, e, output_key="M")
        yield self.run()