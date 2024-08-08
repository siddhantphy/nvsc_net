import numpy as np
from itertools import product
import cvxpy as cp
from numpy.linalg import pinv 


def pauli_matrices():
    """
    Returns the Pauli matrices I, X, Y, Z as a dictionary.
    """
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    return {'I': I, 'X': X, 'Y': Y, 'Z': Z}

def generate_pauli_strings(num_qubits: int) -> list:
    """
    Generates all possible Pauli strings for a given number of qubits.

    Parameters
    ----------
    num_qubits : int
        The number of qubits for which to generate Pauli strings.

    Returns
    -------
    list of tuple
        A list of tuples, each representing a Pauli string of length `num_qubits`. 
        Each tuple contains Pauli operators ('I', 'X', 'Y', 'Z') for the respective qubits.
    
    Example
    -------
    For `num_qubits = 2`, the output will be:
    [('I', 'I'), ('I', 'X'), ('I', 'Y'), ('I', 'Z'), 
     ('X', 'I'), ('X', 'X'), ('X', 'Y'), ('X', 'Z'),
     ('Y', 'I'), ('Y', 'X'), ('Y', 'Y'), ('Y', 'Z'),
     ('Z', 'I'), ('Z', 'X'), ('Z', 'Y'), ('Z', 'Z')]
    """
    paulis = ['I', 'X', 'Y', 'Z']
    pauli_strings = list(product(paulis, repeat=num_qubits))
    return pauli_strings

def construct_pauli_operator(pauli_string: tuple, pauli_dict: dict) -> np.ndarray:
    """
    Constructs a tensor product of Pauli matrices based on the given Pauli string.

    Parameters
    ----------
    pauli_string : tuple of str
        A tuple of Pauli labels ('I', 'X', 'Y', 'Z') for each qubit. Each element of the tuple
        corresponds to a Pauli operator acting on a specific qubit.
    pauli_dict : dict
        A dictionary mapping Pauli labels ('I', 'X', 'Y', 'Z') to their respective 2x2 matrices.
        Example: {'I': np.eye(2), 'X': np.array([[0, 1], [1, 0]]), ...}

    Returns
    -------
    np.ndarray
        The tensor product of the Pauli matrices corresponding to the given Pauli string.

    Example
    -------
    Given `pauli_string = ('X', 'I')` and a corresponding `pauli_dict`, this function 
    returns the tensor product of the X matrix acting on the first qubit and the I 
    (identity) matrix acting on the second qubit.

    For `pauli_string = ('X', 'Z')`, the function returns the tensor product of the 
    X matrix for the first qubit and the Z matrix for the second qubit.

    Notes
    -----
    The function uses the Kronecker product to combine the Pauli matrices as per the order
    of qubits specified in `pauli_string`.
    """
    operator = pauli_dict[pauli_string[0]]
    for i in range(1, len(pauli_string)):
        operator = np.kron(operator, pauli_dict[pauli_string[i]])
    return operator

def planar_412_code_states_and_logical_operators()-> list:
    """
    Generate basic quantum states, logical states for a distance-2 code, and Pauli matrices.

    This function creates the following:
    - Basic quantum states: |0⟩, |1⟩, |+⟩, |−⟩, |i+⟩, |i−⟩.
    - Logical states and logical Pauli operators for a distance-2 code.
    - Pauli matrices I, X, Y, Z.

    Returns
    -------
    tuple
        A tuple containing two elements:
        - A list of basic quantum states: [ket_0, ket_1, ket_plus, ket_minus, ket_i_plus, ket_i_minus].
        - A list of logical states and logical Pauli operators for the distance-2 code:
          [ket_0L, ket_1L, ket_plus_L, ket_minus_L, ket_iplus_L, ket_iminus_L], 
          [I_L, X_L, Y_L, Z_L].
    """
    # Define basic quantum states
    ket_0 = np.array([[1], [0]])
    ket_1 = np.array([[0], [1]])

    ket_0L = (np.kron(np.kron(ket_0, ket_0), np.kron(ket_0, ket_0)) +
               np.kron(np.kron(ket_1, ket_1), np.kron(ket_1, ket_1))) / np.sqrt(2)
    ket_1L = (np.kron(np.kron(ket_0, ket_1), np.kron(ket_0, ket_1)) +
               np.kron(np.kron(ket_1, ket_0), np.kron(ket_1, ket_0))) / np.sqrt(2)

    ket_plus_L = (ket_0L + ket_1L) / np.sqrt(2)
    ket_minus_L = (ket_0L - ket_1L) / np.sqrt(2)

    ket_iplus_L = (ket_0L + 1j * ket_1L) / np.sqrt(2)
    ket_iminus_L = (ket_0L - 1j * ket_1L) / np.sqrt(2)

    I_L = np.outer(ket_0L, ket_0L) + np.outer(ket_1L, ket_1L)
    Z_L = np.outer(ket_0L, ket_0L) - np.outer(ket_1L, ket_1L)
    X_L = np.outer(ket_0L, ket_1L) + np.outer(ket_1L, ket_0L)
    Y_L = -1j * np.outer(ket_0L, ket_1L) + 1j * np.outer(ket_1L, ket_0L)

    logical_states = [ket_0L, ket_1L, ket_plus_L, ket_minus_L, ket_iplus_L, ket_iminus_L]
    logical_pauli_operators = [I_L, X_L, Y_L, Z_L]

    return logical_states, logical_pauli_operators

def estimate_density_matrix(measurement_outcomes: list[float], num_qubits: int) -> np.ndarray:
    """
    Estimate the density matrix of a quantum system using full state tomography.

    This function reconstructs the density matrix from measurement outcomes for 
    each Pauli string. It assumes that the measurement outcomes correspond to the 
    expectation values of Pauli operators in a full tomography setup.

    Parameters
    ----------
    measurement_outcomes : list of float
        A list containing the measurement outcomes (expectation values) for each Pauli string. 
        The length of this list should be equal to the number of Pauli strings for the given number of qubits.
    num_qubits : int
        The number of qubits in the quantum system. This determines the dimension of the density matrix.

    Returns
    -------
    np.ndarray
        The estimated density matrix as a 2D complex numpy array of shape (2^num_qubits, 2^num_qubits).

    Notes
    -----
    The density matrix is computed by summing the contributions of each Pauli operator 
    weighted by its corresponding measurement outcome and then normalizing by the dimension.
    """
    pauli_dict = pauli_matrices()
    pauli_strings = generate_pauli_strings(num_qubits=num_qubits)
    dimension = 2 ** num_qubits
    density_matrix = np.zeros((dimension, dimension), dtype=complex)
    
    for i, pauli_string in enumerate(pauli_strings):
        operator = construct_pauli_operator(pauli_string, pauli_dict)
        expectation_value = measurement_outcomes[i]
        density_matrix += expectation_value * operator
    
    density_matrix /= dimension
    return density_matrix

def get_density_matrix(raw_density_matrix: np.ndarray):
    """
    Computes and returns the corrected 4-qubit density matrix.

    Parameters
    ----------
    raw_density_matrix: The raw density matrix from the averaged measurement outcomes

    Returns
    -------
    np.ndarray
        The corrected 4-qubit density matrix.
    """
    
    # Ensure the density matrix is Hermitian and positive semi-definite
    raw_density_matrix = (raw_density_matrix + raw_density_matrix.conj().T) / 2
    w, v = np.linalg.eigh(raw_density_matrix)
    w = np.maximum(w, 0)  # Clip negative eigenvalues to zero
    raw_density_matrix = (v * w) @ v.conj().T
    
    # Normalize the density matrix
    density_matrix = raw_density_matrix/np.trace(raw_density_matrix)
    
    return density_matrix

def perform_mle_optimization(rho: np.ndarray, pauli_strings: list, measurement_outcomes: np.ndarray, num_qubits: int) -> np.ndarray:
    """
    Perform maximum-likelihood estimation (MLE) optimization to find the physical density matrix that satisfies the physical definitions.

    This function optimizes the density matrix of a quantum system using MLE. The optimization process
    is guided by the initial density matrix `rho` obtained from state tomography. The goal is to find
    the density matrix that best explains the given measurement outcomes based on Pauli string measurements.

    Parameters
    ----------
    rho : np.ndarray
        The initial density matrix (4-qubit) obtained from state tomography. It is used as a starting point
        for the optimization process and included in the objective function to guide the optimization.
    pauli_strings : list of str
        A list of Pauli strings corresponding to the measurements performed. Each Pauli string represents
        a different Pauli operator used in the state tomography.
    measurement_outcomes : np.ndarray
        An array of measurement outcomes, where each element corresponds to the measurement result for a
        respective Pauli string. These outcomes are used to compute the expectation values in the optimization.
    num_qubits : int
        The number of qubits in the quantum system. This determines the dimensions of the density matrix
        and should match the number of qubits used in the Pauli strings and measurement outcomes.

    Returns
    -------
    np.ndarray
        The optimized physical density matrix `rho_ph` that best fits the given measurement outcomes.
    
    Notes
    -----
    The function defines a convex optimization problem where the density matrix `rho_ph` is optimized to
    minimize the difference between the expected measurement outcomes and those predicted by the Pauli operators.
    Additionally, a regularization term penalizes deviations from the initial density matrix `rho` to ensure
    the result remains close to the initial estimate.
    """
    dim = 2 ** num_qubits
    
    # Define the variable rho_ph as a Hermitian matrix with dimensions (dim, dim)
    rho_ph = cp.Variable((dim, dim), complex=True)
    
    # Constraints: rho_ph must be Hermitian, positive semi-definite, and have trace 1
    constraints = [rho_ph == rho_ph.H, cp.trace(rho_ph) == 1, rho_ph >> 0]
    
    # Calculate the objective function: sum of squared differences, incorporating the initial rho
    objective_terms = []
    for i, pauli_str in enumerate(pauli_strings):
        pauli_op = construct_pauli_operator(pauli_str, pauli_matrices())
        expectation_value = measurement_outcomes[i]
        # Incorporate the initial rho into the objective function
        objective_terms.append(cp.square(expectation_value - cp.real(cp.trace(rho_ph @ pauli_op))))
    
    # Define the objective to minimize the sum of these terms, and include a term to penalize deviations from initial rho
    rho_initial = cp.Constant(rho)
    initial_rho_term = cp.norm(rho_ph - rho_initial, 'fro')**2
    objective = cp.Minimize(cp.sum(objective_terms) + initial_rho_term)
    
    # Solve the optimization problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # Extract the optimized density matrix
    rho_ph_optimized = rho_ph.value
    
    # Ensure the matrix is Hermitian
    rho_ph_optimized = (rho_ph_optimized + rho_ph_optimized.conj().T) / 2
    
    return rho_ph_optimized

def logical_density_matrix(rho_ph_optimized: np.ndarray, code: str = "NA") -> np.ndarray:
    """
    Compute the logical density matrix by projecting the optimized physical density matrix
    onto the logical codespace.

    This function computes the logical density matrix by projecting the given physical density
    matrix onto the logical subspace defined by the logical Pauli operators for the specified code.

    Parameters
    ----------
    rho_ph_optimized : np.ndarray
        The optimized physical density matrix, which should be a square matrix of dimensions
        consistent with the number of qubits in the system.
    code : str, optional
        The code for which the logical Pauli operators should be used. Currently, only "412_planar"
        is supported. If the provided code is not recognized, an error will be raised.

    Returns
    -------
    np.ndarray
        The logical density matrix rho_L, which is the projection of the physical density matrix
        onto the logical subspace.

    Raises
    ------
    ValueError
        If the provided code is not recognized, a ValueError is raised.
    """
    # Initialize the logical density matrix
    rho_L = np.zeros_like(rho_ph_optimized, dtype=complex)
    
    # Retrieve logical Pauli operators based on the specified code
    if code == "412_planar":
        _, logical_paulis = planar_412_code_states_and_logical_operators()
    else:
        raise ValueError(f"No valid error-detection code used: {code}. Supported code: '412_planar'.")

    # Project the physical density matrix onto the logical codespace
    for pauli_op in logical_paulis:
        trace_term = np.trace(rho_ph_optimized @ pauli_op)
        normalization = np.trace(rho_ph_optimized @ logical_paulis[0])
        rho_L += (trace_term / normalization) * pauli_op
    
    rho_L /= 2  # Normalize by the number of logical Pauli operators
    
    # Ensure the matrix is Hermitian
    rho_L = (rho_L + rho_L.conj().T) / 2
    
    return rho_L

def logical_pauli_transfer_matrix()->np.ndarray:
    """
    Computes the logical Pauli Transfer Matrix (PTM) for a quantum error-correcting code.

    The function uses randomly generated measurement outcomes to simulate state tomography data 
    and estimates the corresponding logical density matrices for different initial states (|0⟩, |1⟩, |+⟩, |−⟩, |i+⟩, |i−⟩).
    It then constructs the logical Pauli vectors for both input and output states, and solves a linear equation to find the 
    logical Pauli Transfer Matrix that best describes the transformation of the logical qubits.

    Returns
    -------
    np.ndarray
        The 4x4 logical Pauli Transfer Matrix that maps the input logical Pauli vectors to the output logical Pauli vectors.
    """
    np.random.seed(42)  # For reproducibility and testing of functions!

    # The following will be replaced by actual data from the scripts output files in appropriate format as evident for "measurement_outcomes" variable
    meas_0_input_data = np.random.uniform(-1, 1, 256)
    meas_0_output_data = np.random.uniform(-1, 1, 256)
    meas_1_input_data = np.random.uniform(-1, 1, 256)
    meas_1_output_data = np.random.uniform(-1, 1, 256)
    meas_plus_input_data = np.random.uniform(-1, 1, 256)
    meas_plus_output_data = np.random.uniform(-1, 1, 256)
    meas_minus_input_data = np.random.uniform(-1, 1, 256)
    meas_minus_output_data = np.random.uniform(-1, 1, 256)
    meas_iplus_input_data = np.random.uniform(-1, 1, 256)
    meas_iplus_output_data = np.random.uniform(-1, 1, 256)
    meas_iminus_input_data = np.random.uniform(-1, 1, 256)
    meas_iminus_output_data = np.random.uniform(-1, 1, 256)

    # Note that the data qubit order is D1-D2-D3-D4
    rho_0_input= logical_density_matrix(rho_ph_optimized=perform_mle_optimization(rho=get_density_matrix(estimate_density_matrix(measurement_outcomes=meas_0_input_data,num_qubits=4)),measurement_outcomes=meas_0_input_data,pauli_strings=generate_pauli_strings(num_qubits=4),num_qubits=4),code="412_planar")
    rho_0_output= logical_density_matrix(rho_ph_optimized=perform_mle_optimization(rho=get_density_matrix(estimate_density_matrix(measurement_outcomes=meas_0_output_data,num_qubits=4)),measurement_outcomes=meas_0_output_data,pauli_strings=generate_pauli_strings(num_qubits=4),num_qubits=4),code="412_planar")
    rho_1_input= logical_density_matrix(rho_ph_optimized=perform_mle_optimization(rho=get_density_matrix(estimate_density_matrix(measurement_outcomes=meas_1_input_data,num_qubits=4)),measurement_outcomes=meas_1_input_data,pauli_strings=generate_pauli_strings(num_qubits=4),num_qubits=4),code="412_planar")
    rho_1_output= logical_density_matrix(rho_ph_optimized=perform_mle_optimization(rho=get_density_matrix(estimate_density_matrix(measurement_outcomes=meas_1_output_data,num_qubits=4)),measurement_outcomes=meas_1_output_data,pauli_strings=generate_pauli_strings(num_qubits=4),num_qubits=4),code="412_planar")
    rho_plus_input= logical_density_matrix(rho_ph_optimized=perform_mle_optimization(rho=get_density_matrix(estimate_density_matrix(measurement_outcomes=meas_plus_input_data,num_qubits=4)),measurement_outcomes=meas_plus_input_data,pauli_strings=generate_pauli_strings(num_qubits=4),num_qubits=4),code="412_planar")
    rho_plus_output= logical_density_matrix(rho_ph_optimized=perform_mle_optimization(rho=get_density_matrix(estimate_density_matrix(measurement_outcomes=meas_plus_output_data,num_qubits=4)),measurement_outcomes=meas_plus_output_data,pauli_strings=generate_pauli_strings(num_qubits=4),num_qubits=4),code="412_planar")
    rho_minus_input= logical_density_matrix(rho_ph_optimized=perform_mle_optimization(rho=get_density_matrix(estimate_density_matrix(measurement_outcomes=meas_minus_input_data,num_qubits=4)),measurement_outcomes=meas_minus_input_data,pauli_strings=generate_pauli_strings(num_qubits=4),num_qubits=4),code="412_planar")
    rho_minus_output= logical_density_matrix(rho_ph_optimized=perform_mle_optimization(rho=get_density_matrix(estimate_density_matrix(measurement_outcomes=meas_minus_output_data,num_qubits=4)),measurement_outcomes=meas_minus_output_data,pauli_strings=generate_pauli_strings(num_qubits=4),num_qubits=4),code="412_planar")
    rho_iplus_input= logical_density_matrix(rho_ph_optimized=perform_mle_optimization(rho=get_density_matrix(estimate_density_matrix(measurement_outcomes=meas_iplus_input_data,num_qubits=4)),measurement_outcomes=meas_iplus_input_data,pauli_strings=generate_pauli_strings(num_qubits=4),num_qubits=4),code="412_planar")
    rho_iplus_output= logical_density_matrix(rho_ph_optimized=perform_mle_optimization(rho=get_density_matrix(estimate_density_matrix(measurement_outcomes=meas_iplus_output_data,num_qubits=4)),measurement_outcomes=meas_iplus_output_data,pauli_strings=generate_pauli_strings(num_qubits=4),num_qubits=4),code="412_planar")
    rho_iminus_input= logical_density_matrix(rho_ph_optimized=perform_mle_optimization(rho=get_density_matrix(estimate_density_matrix(measurement_outcomes=meas_iminus_input_data,num_qubits=4)),measurement_outcomes=meas_iminus_input_data,pauli_strings=generate_pauli_strings(num_qubits=4),num_qubits=4),code="412_planar")
    rho_iminus_output= logical_density_matrix(rho_ph_optimized=perform_mle_optimization(rho=get_density_matrix(estimate_density_matrix(measurement_outcomes=meas_iminus_output_data,num_qubits=4)),measurement_outcomes=meas_iminus_output_data,pauli_strings=generate_pauli_strings(num_qubits=4),num_qubits=4),code="412_planar")

    _, logical_paulis = planar_412_code_states_and_logical_operators()

    p_L_0_input = [ np.trace(rho_0_input @ logical_paulis[i]) for i in range(len(logical_paulis))]
    p_L_0_output = [ np.trace(rho_0_output @ logical_paulis[i]) for i in range(len(logical_paulis))]
    p_L_1_input = [ np.trace(rho_1_input @ logical_paulis[i]) for i in range(len(logical_paulis))]
    p_L_1_output = [ np.trace(rho_1_output @ logical_paulis[i]) for i in range(len(logical_paulis))]
    p_L_plus_input = [ np.trace(rho_plus_input @ logical_paulis[i]) for i in range(len(logical_paulis))]
    p_L_plus_output = [ np.trace(rho_plus_output @ logical_paulis[i]) for i in range(len(logical_paulis))]
    p_L_minus_input = [ np.trace(rho_minus_input @ logical_paulis[i]) for i in range(len(logical_paulis))]
    p_L_minus_output = [ np.trace(rho_minus_output @ logical_paulis[i]) for i in range(len(logical_paulis))]
    p_L_iplus_input = [ np.trace(rho_iplus_input @ logical_paulis[i]) for i in range(len(logical_paulis))]
    p_L_iplus_output = [ np.trace(rho_iplus_output @ logical_paulis[i]) for i in range(len(logical_paulis))]
    p_L_iminus_input = [ np.trace(rho_iminus_input @ logical_paulis[i]) for i in range(len(logical_paulis))]
    p_L_iminus_output = [ np.trace(rho_iminus_output @ logical_paulis[i]) for i in range(len(logical_paulis))]

    logical_pauli_vectors_input = np.array([p_L_0_input, p_L_1_input, p_L_plus_input, p_L_minus_input, p_L_iplus_input, p_L_iminus_input])
    logical_pauli_vectors_output = np.array([p_L_0_output, p_L_1_output, p_L_plus_output, p_L_minus_output, p_L_iplus_output, p_L_iminus_output])

    # To construct the PTM R, solve the linear equation p'_j = sum_i R_ji * p_i
    # For each input-output pair, we have a matrix equation
    # Stack all input-output equations together to form a large linear system
    # Reshape the input and output matrices
    input_matrix = np.vstack(logical_pauli_vectors_input).T
    output_matrix = np.vstack(logical_pauli_vectors_output).T

    # Using the pseudo-inverse to find the best fit for R
    input_pinv = pinv(input_matrix)  # 6x4 matrix

    # Compute R (4x4 matrix) as the logical pauli transfer matrix
    R = np.dot(output_matrix, input_pinv)

    return R

def choi_from_ptm(R: np.ndarray)->np.ndarray:
    """
    Construct the Choi matrix from the Pauli Transfer Matrix (PTM) for a single qubit.
    
    Parameters
    ----------
    R : np.ndarray
        Pauli Transfer Matrix (PTM) of dimensions (4, 4).
    
    Returns
    -------
    np.ndarray
        The Choi matrix rho^R corresponding to the quantum channel described by R.
    """
    # Pauli matrices for a single qubit
    pauli_matrices = [
        np.array([[1, 0], [0, 1]]),  # I
        np.array([[0, 1], [1, 0]]),  # X
        np.array([[0, -1j], [1j, 0]]), # Y
        np.array([[1, 0], [0, -1]])   # Z
    ]
    
    # Initialize the Choi matrix
    rho_R = np.zeros((4, 4), dtype=complex)
    
    # Compute the Choi matrix using the given formula
    for i in range(4):
        for j in range(4):
            sigma_i = pauli_matrices[i]
            sigma_j = pauli_matrices[j]
            rho_R += R[i, j] * np.kron(sigma_j.T, sigma_i)
    
    rho_R /= 4.0
    return rho_R

def partial_trace_first_qubit(rho: np.ndarray)->np.ndarray:
    """
    Compute the partial trace over the first qubit of a 4x4 matrix.
    
    Parameters
    ----------
    rho : np.ndarray
        A 4x4 matrix (Choi matrix).
    
    Returns
    -------
    np.ndarray
        The resulting 2x2 matrix after taking the partial trace over the first qubit.
    """
    return np.array([
        [rho[0, 0] + rho[2, 2], rho[0, 1] + rho[2, 3]],
        [rho[1, 0] + rho[3, 2], rho[1, 1] + rho[3, 3]]
    ])

def partial_trace_first_qubit_cvxpy(rho: np.ndarray)->np.ndarray:
    """
    Compute the partial trace over the first qubit of a 4x4 matrix within cvxpy framework.
    
    Parameters
    ----------
    rho : cvxpy.Variable
        A 4x4 cvxpy variable representing the Choi matrix.
    
    Returns
    -------
    cp.Expression
        The resulting 2x2 matrix after taking the partial trace over the first qubit.
    """
    # Manually compute the partial trace
    rho_A = cp.vstack([
        cp.hstack([rho[0, 0] + rho[2, 2], rho[0, 1] + rho[2, 3]]),
        cp.hstack([rho[1, 0] + rho[3, 2], rho[1, 1] + rho[3, 3]])
    ])
    
    return rho_A

def optimize_choi_matrix(rho_R: np.ndarray)->np.ndarray:
    """
    Perform convex optimization to find the physical Choi matrix satisfying TPCP constraints.
    
    Parameters
    ----------
    rho_R : np.ndarray
        The Choi matrix of Pauli Transfer Matrix (PTM) of dimensions (4, 4).
    
    Returns
    -------
    np.ndarray
        The optimized physical Choi matrix rho_ph^R_opt.
    """
    # Define optimization variable for the Choi matrix
    rho_ph = cp.Variable((4, 4), complex=True)
    
    # Constraints
    constraints = [
        rho_ph == rho_ph.H,  # Hermitian
        cp.trace(rho_ph) == 1,  # Trace 1
        rho_ph >> 0  # Positive semi-definite
    ]
    
    # Partial trace condition: Tr1(rho_ph) = 1/2 * I_2 (where I_2 is 2x2 identity matrix)
    rho_A = partial_trace_first_qubit_cvxpy(rho_ph)
    constraints.append(rho_A == 0.5 * np.eye(2))
    
    # Objective function: minimize the Frobenius norm between rho_ph and rho_R
    objective = cp.Minimize(cp.norm(rho_ph - rho_R, 'fro'))
    
    # Solve the optimization problem
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    # Return the optimized Choi matrix
    return rho_ph.value

def calculate_gate_fidelity(R_ideal: np.ndarray, R_opt_ph: np.ndarray)->float:
    """
    Calculate the average logical-gate fidelity.
    
    Parameters
    ----------
    R_ideal : np.ndarray
        The ideal Pauli Transfer Matrix of the noiseless case.
    R_opt_ph : np.ndarray
        The optimized Pauli Transfer Matrix obtained from the optimization for the chosen noisy case.
    
    Returns
    -------
    float
        The average logical-gate fidelity.
    """
    fidelity = (np.trace(R_ideal.conj().T @ R_opt_ph) + 2) / 6.0
    return np.real(fidelity)


# Test run
print(optimize_choi_matrix(choi_from_ptm(logical_pauli_transfer_matrix())))