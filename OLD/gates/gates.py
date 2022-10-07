import numpy as np
import netsquid as ns
from netsquid.qubits.operators import *



# Single qubit general rotation gates

def Rx(theta:float):
    r = Operator("Rx", np.cos(theta/2) * I.arr - 1j * np.sin(theta/2) * X.arr)
    return r

def Ry(theta:float):
    r = Operator("Ry", np.cos(theta/2) * I.arr - 1j * np.sin(theta/2) * Y.arr)
    return r

def Rz(phi:float):
    r = Operator("Rz", np.cos(phi/2) * I.arr - 1j * np.sin(phi/2) * Z.arr)
    return r


# Logical operations
