import warnings
import time
from simplex import simplex
from scipy.optimize import linprog
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)  # Suppress deprecation warnings for scipy's simplex method

def test(c, A, b, constraints, method):
    start_time = time.time()
    optimal_solution, optimal_value, success = simplex(c, A, b, constraints, method)
    end_time = time.time()
    total_time = round(end_time - start_time, 5)

    print("Our Simplex Method:")
    if success:
        print("Optimal Solution: ", optimal_solution)
        print("Optimal Value: ", optimal_value)
    print("Success: ", success)
    print(f"Time taken: {total_time} seconds")

def test_scipy(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, lb=None, ub=None):
    start_time = time.time()
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
    end_time = time.time()
    total_time = round(end_time - start_time, 5)

    print("Scipy Simplex Method:")
    if res.success:
        print("Optimal Solution: ", res.x)
        print("Optimal Value: ", -res.fun)  # linprog takes minimization problems, so we need to negate the value
    print("Success: ", res.success)
    print(f"Time taken: {total_time} seconds")

def test1():
    """
    This represents the following problem:
    Maximize 3*x1 + 2*x2
    Subject to:
    x1 + x2 <= 4
    x1 - x2 >= 1
    x1, x2 >= 0 (implicit)

    Expected optimal solution: [4, 0]
    Expected optimal value: 12
    """
    c = np.array([3, 2])
    A = np.array([[1, 1], [1, -1]])
    b = np.array([4, 1])
    constraints = ['<=', '>=']
    method = 'max'
    test(c, A, b, constraints, method)

def test1_scipy():
    """
    Same problem as test1, but using scipy's linprog function
    """
    c = np.array([-3, -2])
    A_ub = np.array([[1, 1], [-1, 1]])
    b_ub = np.array([4, -1])
    test_scipy(c, A_ub, b_ub)

def test2():
    """
    This represents the following problem:
    Minimize 2*x1 + 3*x2
    Subject to:
    2*x1 + x2 >= 8
    x1 + 2*x2 >= 6
    x1, x2 >= 0 (implicit)

    Expected optimal solution: [3 1/3, 1 1/3]
    Expected optimal value: 10 2/3
    """
    c = np.array([2, 3])
    A = np.array([[2, 1], [1, 2]])
    b = np.array([8, 6])
    constraints = ['>=', '>=']
    method = 'min'
    test(c, A, b, constraints, method)

def test2_scipy():
    """
    Same problem as test2, but using scipy's linprog function
    """
    c = np.array([2, 3])
    A = np.array([[2, 1], [1, 2]])
    b = np.array([8, 6])
    test_scipy(c, A, b)

def test_unbounded():
    """
    This represents the following problem:
    Minimize: -2*x1 - 3*x2 - 4*x3
    Subject to:
    3*x1 + 2*x2 + x3 >= 10
    2*x1 + 5*x2 + 3*x3 >= 15
    x1 + 2*x2 + 2*x3 >= 8
    x1, x2, x3 >= 0 (implicit)

    Expected success: False
    """
    c = np.array([-2, -3, -4])
    A = np.array([[3, 2, 1], [2, 5, 3], [1, 2, 2]])
    b = np.array([10, 15, 8])
    constraints = ['>=', '>=', '>=']
    method = 'min'
    test(c, A, b, constraints, method)

def test_unbounded_scipy():
    """
    Same problem as test_unbounded, but using scipy's linprog function
    """
    c = np.array([2, 3, 4])
    A_ub = np.array([[3, 2, 1], [2, 5, 3], [1, 2, 2]])
    b_ub = np.array([10, 15, 8])
    test_scipy(c, A_ub, b_ub)

def test3():
    """
    This represents the following problem:
    Maximize 7*x1 + 6*x2
    Subject to:
    2*x1 + 4*x2 <= 16
    3*x1 + 2*x2 <= 12
    x1, x2 >= 0 (implicit)

    Expected optimal solution: [2, 3]
    Expected optimal value: 32
    """
    c = np.array([7, 6])
    A = np.array([[2, 4], [3, 2]])
    b = np.array([16, 12])
    constraints = ['<=', '<=']
    method = 'max'
    test(c, A, b, constraints, method)

def test3_scipy():
    """
    Same problem as test3, but using scipy's linprog function
    """
    c = np.array([-7, -6])
    A_ub = np.array([[2, 4], [3, 2]])
    b_ub = np.array([16, 12])
    test_scipy(c, A_ub, b_ub)

# Uncomment the following lines to run the tests
# test1()
# test1_scipy()
# test2()
# test2_scipy()
# test_unbounded()
# test_unbounded_scipy()
# test3()
# test3_scipy()