from typing import Tuple, List
import numpy as np
from numpy.typing import NDArray

def simplex(c: NDArray, A: NDArray, b: NDArray, constraints: List[str], method: str, M: float = 1e6) -> Tuple[NDArray, NDArray, bool]:
    """
    Solve the linear program using the simplex method
    
    Args:
        c: Coefficients in the objective function
        A: Coefficients in the constraints
        b: Right-hand side of the constraints
        constraints: List of constraint types ('<=', '>=', or '=')
        method: Optimization method ('max' or 'min')
        M: Penalty coefficient for artificial variables (default is 1e6)
        
    Returns:
        Tuple[NDArray, NDArray, bool]: A tuple containing:
            NDArray: Optimal solution
            NDArray: Optimal value of the objective function
            bool: A boolean flag indicating successful termination
    """
    # Convert the linear program into standard form
    c_std, A_std, b_std = convert_to_standard_form(c, A, b, constraints, method, M)
    
    # Initialize the tableau
    tableau, artificial_indices = initialize_tableau(c_std, A_std, b_std, M)
    
    # Perform the simplex method
    optimal_solution, optimal_value, success = perform_simplex(tableau, artificial_indices, c.shape[0])
    if success:
        if method == "min":
            optimal_value = -optimal_value  # Negate the value if the original problem was a minimization problem

        # Sanity check
        if not np.allclose(A @ optimal_solution, b, atol=1e-6):
            success = False
    
    return optimal_solution, optimal_value, success

def convert_to_standard_form(c: NDArray, A: NDArray, b: NDArray, constraints: List[str], method: str, M: float) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Convert the linear program into standard form
    
    Args:
        c: Coefficients in the objective function
        A: Coefficients in the constraints
        b: Right-hand side of the constraints
        constraints: List of constraint types ('<=', '>=', or '=')
        method: Optimization method ('max' or 'min')
        M: Penalty coefficient for artificial variables
        
    Returns:
        Tuple[NDArray, NDArray, NDArray]: A tuple containing the following:
            NDArray: Coefficients in the objective function
            NDArray: Coefficients in the constraints
            NDArray: Right-hand side of the constraints
    """
    # Get the number of constraints
    num_constraints = A.shape[0]

    # Initialize lists to store slack, surplus, and artificial variables
    slack_vars, surplus_vars, artificial_vars = [], [], []

    # Check if the RHS of the constraints is negative
    for i in range(num_constraints):
        if b[i] < 0:
            # Negate everything in the row
            A[i] = -A[i]
            b[i] = -b[i]

            # Change the constraint type
            if constraints[i] == "<=":
                constraints[i] = ">="
            elif constraints[i] == ">=":
                constraints[i] = "<="

    # Iterate over the constraints to identify their types and add slack, surplus, and artificial variables
    for i, constraint in enumerate(constraints):
        if constraint == "<=":
            slack_var = np.zeros(num_constraints)
            slack_var[i] = 1
            slack_vars.append(slack_var)
        elif constraint == ">=":
            surplus_var = np.zeros(num_constraints)
            surplus_var[i] = -1
            surplus_vars.append(surplus_var)
            artificial_var = np.zeros(num_constraints)
            artificial_var[i] = 1
            artificial_vars.append(artificial_var)
        elif constraint == "=":
            artificial_var = np.zeros(num_constraints)
            artificial_var[i] = 1
            artificial_vars.append(artificial_var)
        else:
            raise ValueError("Invalid constraint type")  # Raise an error if the constraint type is invalid
        
    # Convert the lists into NumPy arrays
    if slack_vars:
        slack_vars = np.vstack(slack_vars).T
    else:
        slack_vars = np.zeros((num_constraints, 0))
    if surplus_vars:
        surplus_vars = np.vstack(surplus_vars).T
    else:
        surplus_vars = np.zeros((num_constraints, 0))
    if artificial_vars:
        artificial_vars = np.vstack(artificial_vars).T
    else:
        artificial_vars = np.zeros((num_constraints, 0))

    # Convert a minimization problem into a maximization problem
    if method == "min":
        c = -c
    elif method != "max":
        raise ValueError("Invalid optimization method")  # Raise an error if the optimization method is invalid

    # Update the objective function coefficients
    c_std = np.concatenate((c, np.zeros(slack_vars.shape[1] + surplus_vars.shape[1]), -M * np.ones(artificial_vars.shape[1])))

    # Update the constraint coefficients with slack, surplus, and artificial variables
    A_std = np.hstack((A, slack_vars, surplus_vars, artificial_vars))

    return c_std, A_std, b

def initialize_tableau(c: NDArray, A: NDArray, b: NDArray, M: float) -> Tuple[NDArray, List[int]]:
    """
    Initialize the tableau for the simplex method
    
    Args:
        c: Coefficients in the objective function
        A: Coefficients in the constraints
        b: Right-hand side of the constraints
        M: Penalty coefficient for artificial variables
        
    Returns:
        Tuple[NDArray, List[int]]: A tuple containing the following:
            NDArray: Initial tableau
            List[int]: Indices of the artificial variables in the tableau
    """
    # Get the number of constraints and variables
    num_constraints, num_variables = A.shape

    # Initialize the tableau with zeros
    tableau = np.zeros((num_constraints + 1, num_variables + 1))

    # Bottom row should contain the coefficients of the objective function minus the basis vectors multiplied by the constraints
    # First we need to find a basis from the column vectors of A, the coefficients of the constraints matching the identity matrix
    identity = np.eye(num_constraints)
    basis = []
    artificial_indices = []
    count = 0
    for i in range(num_variables):
        if np.array_equal(A[:, i], identity[:, count]):
            basis.append(c[i])
            count += 1
            if c[i] == -M:
                artificial_indices.append(i)

    # Calculate the z-row for the initial tableau
    basis = np.array(basis)
    z_row = np.zeros(num_variables + 1)
    for col in range(num_variables):
        z_row[col] = c[col] - np.sum(basis * A[:, col])
    z_row[-1] = -np.sum(basis * b)

    tableau[-1, :] = z_row

    # Fill in the coefficients of the constraints
    tableau[:-1, :-1] = A

    # Fill in the right-hand side of the constraints
    tableau[:-1, -1] = b

    return tableau, artificial_indices

def perform_simplex(tableau: NDArray, artificial_indices: List[int], num_vars: int) -> Tuple[NDArray, NDArray, bool]:
    """
    Perform the simplex method
    
    Args:
        tableau: Initial tableau
        num_vars: Number of variables in the original problem (excludes slack, surplus, and artificial variables)
        
    Returns:
        Tuple[NDArray, NDArray, bool]: A tuple containing:
            NDArray: Optimal solution, or None if the solution is unbounded
            NDArray: Optimal value of the objective function, or None if the solution is unbounded
            bool: A boolean flag indicating successful termination
    """
    # Perform the simplex method until the solution is optimal
    while not is_optimal(tableau):
        # Select the pivot column and row
        pivot_column = select_pivot_column(tableau)
        pivot_row = select_pivot_row(tableau, pivot_column)

        # Check if the solution is unbounded
        if pivot_row is None:
            return None, None, False
        
        # Remove the artificial variable from the basis and its column from the tableau
        for col in artificial_indices:
            if np.count_nonzero(tableau[:, col]) == 1 and np.sum(tableau[:, col]) == 1:
                row = np.where(tableau[:, col] == 1)[0][0]
                if row == pivot_row:
                    tableau = np.delete(tableau, col, axis=1)
                    artificial_indices.remove(col)
                    for i in range(len(artificial_indices)):
                        if artificial_indices[i] != col:
                            artificial_indices[i] -= 1
                    break

        # Perform the pivot operation
        tableau = pivot_operation(tableau, pivot_row, pivot_column)
    
    # Extract the optimal solution and value
    optimal_solution, optimal_value = extract_optimal(tableau, num_vars)
    
    return optimal_solution, optimal_value, True
    
def is_optimal(tableau: NDArray) -> bool:
    """
    Check if the current solution is optimal
    
    Args:
        tableau: Current tableau
        
    Returns:
        bool: True if the current solution is optimal, False otherwise
    """
    return np.all(tableau[-1, :-1] <= 0)

def select_pivot_column(tableau: NDArray) -> int:
    """
    Select the pivot column for the next iteration of the simplex method
    
    Args:
        tableau: Current tableau
        
    Returns:
        int: Index of the pivot column
    """
    return np.argmax(tableau[-1, :-1])

def select_pivot_row(tableau: NDArray, pivot_column: int) -> int:
    """
    Select the pivot row for the next iteration of the simplex method
    
    Args:
        tableau: Current tableau
        pivot_column: Index of the pivot column
        
    Returns:
        int: Index of the pivot row, or None if the solution is unbounded
    """
    # Calculate the ratios of the right-hand side to the pivot column
    with np.errstate(divide="ignore"):  # Ignore division by zero
        ratios = tableau[:-1, -1] / tableau[:-1, pivot_column]
    valid_ratios = np.where(tableau[:-1, pivot_column] > 0, ratios, np.inf)

    # Check if the solution is unbounded
    if np.all(valid_ratios == np.inf):
        return None
    
    return np.argmin(valid_ratios)

def pivot_operation(tableau: NDArray, pivot_row: int, pivot_column: int) -> NDArray:
    """
    Perform the pivot operation to update the tableau
    
    Args:
        tableau: Current tableau
        pivot_row: Index of the pivot row
        pivot_column: Index of the pivot column
        
    Returns:
        NDArray: Updated tableau
    """
    # Create a new tableau to store the updated values
    new_tableau = np.copy(tableau)
    new_tableau[pivot_row, :] /= tableau[pivot_row, pivot_column]

    # Perform the row operations
    for i in range(tableau.shape[0]):
        if i != pivot_row:  # Skip the pivot row
            new_tableau[i, :] -= new_tableau[pivot_row, :] * tableau[i, pivot_column]

    return new_tableau

def extract_optimal(tableau: NDArray, num_vars: int) -> Tuple[NDArray, float]:
    """
    Extract the optimal solution and value from the final tableau
    
    Args:
        tableau: Final tableau
        num_vars: Number of variables in the original problem (excludes slack, surplus, and artificial variables)
        
    Returns:
        NDArray: Optimal solution
        float: Optimal value of the objective function
    """
    solution = np.zeros(num_vars)
    for i in range(num_vars):
        column = tableau[:-1, i]

        # Check if the column is a unit vector with a single 1 - this indicates a basic variable
        if np.count_nonzero(column) == 1 and np.sum(column) == 1:
            row = np.where(column == 1)[0][0]
            solution[i] = tableau[row, -1]

    value = -tableau[-1, -1]  # Negate the value because the objective function was negated
    return solution, value
