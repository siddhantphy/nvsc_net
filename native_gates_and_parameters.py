from netsquid_nv.nv_center import NVQuantumProcessor
from netsquid.components.qprocessor import PhysicalInstruction
import netsquid.components.instructions as instr


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

