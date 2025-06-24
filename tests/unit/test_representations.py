import pytest
import numpy as np
from boolfunc.core.representations.truth_table import TruthTableRepresentation
from boolfunc.core.representations.fourier_expansion import FourierExpansionRepresentation
from boolfunc.core.representations.symbolic import SymbolicRepresentation
from boolfunc.core.spaces import Space

# Dummy BooleanFunction wrapper for testing
class DummyBooleanFunction:
    def __init__(self, repr_type, data, space, n_vars):
        self.repr_type = repr_type
        self.data = data
        self.n_vars = n_vars
        self.space = space

    def get_n_vars(self):
        return self.n_vars

    def evaluate(self, inputs, rep_type=None, **kwargs):
        return self.repr_type.evaluate(inputs, self.data, self.space, self.n_vars)


AND_TRUTH_TABLE = np.array([0, 0, 0, 1])  # [00, 01, 10, 11]
AND_FOURIER_COEFFS = np.array([0.5, 0.5, 0.5, -0.5])  # ∅, {0}, {1}, {0,1}

XOR_TRUTH_TABLE = np.array([0, 1, 1, 0])
XOR_FOURIER_COEFFS = np.array([0, 0, 0, 1])  # Only {0,1} term

# Fixtures for common test objects

@pytest.fixture
def sym_rep():
    return SymbolicRepresentation()

@pytest.fixture
def tt_rep():
    return TruthTableRepresentation()

@pytest.fixture
def fourier_rep():
    return FourierExpansionRepresentation()

@pytest.fixture
def boo_cube():
    return Space.BOOLEAN_CUBE


## Truth Table Representation Tests ##

def test_truth_table_evaluate_single(tt_rep):
    """Test evaluation of single inputs"""
    # AND function
    assert tt_rep.evaluate(np.array(0), AND_TRUTH_TABLE, space = boo_cube, n_vars = 2) == 0
    assert tt_rep.evaluate(np.array(1), AND_TRUTH_TABLE, space = boo_cube, n_vars = 2) == 0
    assert tt_rep.evaluate(np.array(2), AND_TRUTH_TABLE, space = boo_cube, n_vars = 2) == 0
    
    # XOR function
    assert tt_rep.evaluate(np.array(0), XOR_TRUTH_TABLE, space = boo_cube, n_vars = 2) == 0
    assert tt_rep.evaluate(np.array(1), XOR_TRUTH_TABLE, space = boo_cube, n_vars = 2) == 1
    assert tt_rep.evaluate(np.array(2), XOR_TRUTH_TABLE, space = boo_cube, n_vars = 2) == 1
    assert tt_rep.evaluate(np.array(3), XOR_TRUTH_TABLE, space = boo_cube, n_vars = 2) == 0

def test_truth_table_evaluate_batch(tt_rep):
    """Test batch evaluation"""
    inputs = np.array([0, 1, 2, 3])
    
    # AND function
    results = tt_rep.evaluate(inputs, AND_TRUTH_TABLE, space = boo_cube, n_vars = 2)
    assert np.array_equal(results, np.array([0, 0, 0, 1]))
    
    # XOR function
    results = tt_rep.evaluate(inputs, XOR_TRUTH_TABLE, space = boo_cube, n_vars = 2)
    assert np.array_equal(results, np.array([0, 1, 1, 0]))

def test_truth_table_dump(tt_rep):
    """Test serialization of truth table"""
    dumped = tt_rep.dump(AND_TRUTH_TABLE)
    assert dumped['type'] == 'truth_table'
    assert dumped['values'] == [0, 0, 0, 1]
    assert dumped['n'] == 2
    assert dumped['size'] == 4

def test_truth_table_create_empty(tt_rep):
    """Test empty truth table creation"""
    empty_tt = tt_rep.create_empty(3)
    assert len(empty_tt) == 8
    assert np.all(empty_tt == 0)

def test_truth_table_storage_requirements(tt_rep):
    """Test storage requirements calculation"""
    requirements = tt_rep.get_storage_requirements(3)
    assert requirements['entries'] == 8
    assert requirements['bytes'] == 1  # packed bits


## Fourier Expansion Representation Tests ##
def test_fourier_evaluate_single(fourier_rep):
    """Test evaluation of single inputs"""
    # AND function in ±1 domain
    assert fourier_rep.evaluate(0, AND_FOURIER_COEFFS, space = boo_cube, n_vars = 2) == 1.0
    assert fourier_rep.evaluate(1, AND_FOURIER_COEFFS, space = boo_cube, n_vars = 2) == 1.0
    assert fourier_rep.evaluate(2, AND_FOURIER_COEFFS, space = boo_cube, n_vars = 2) == 1.0
    assert fourier_rep.evaluate(3, AND_FOURIER_COEFFS, space = boo_cube, n_vars = 2) == -1.0
    
    # XOR function in ±1 domain
    assert fourier_rep.evaluate(0, XOR_FOURIER_COEFFS, space = boo_cube, n_vars = 2) == 1.0
    assert fourier_rep.evaluate(1, XOR_FOURIER_COEFFS, space = boo_cube, n_vars = 2) == -1.0
    assert fourier_rep.evaluate(2, XOR_FOURIER_COEFFS, space = boo_cube, n_vars = 2) == -1.0
    assert fourier_rep.evaluate(3, XOR_FOURIER_COEFFS, space = boo_cube, n_vars = 2) == 1.0

def test_fourier_evaluate_batch(fourier_rep):
    """Test batch evaluation"""
    inputs = np.array([0, 1, 2, 3])
    
    # AND function
    results = fourier_rep.evaluate(inputs, AND_FOURIER_COEFFS, space = boo_cube, n_vars = 2) 
    expected = np.array([1.0, 1.0, 1.0, -1.0])
    assert np.allclose(results, expected)
    
    # XOR function
    results = fourier_rep.evaluate(inputs, XOR_FOURIER_COEFFS, space = boo_cube, n_vars = 2) 
    expected = np.array([1.0, -1.0, -1.0, 1.0])
    assert np.allclose(results, expected)

def test_fourier_dump(fourier_rep):
    """Test serialization of Fourier coefficients"""
    dumped = fourier_rep.dump(AND_FOURIER_COEFFS)
    assert dumped['type'] == 'fourier_expansion'
    assert dumped['coefficients'] == [0.5, 0.5, 0.5, -0.5]
    assert dumped['metadata']['num_vars'] == 2
    assert dumped['metadata']['norm'] == pytest.approx(1.0)

def test_fourier_create_empty(fourier_rep):
    """Test empty Fourier coefficients creation"""
    empty_coeffs = fourier_rep.create_empty(2)
    assert len(empty_coeffs) == 4
    assert np.all(empty_coeffs == 0.0)

def test_fourier_storage_requirements(fourier_rep):
    """Test storage requirements calculation"""
    requirements = fourier_rep.get_storage_requirements(4)
    assert requirements['elements'] == 16
    assert requirements['bytes'] == 128  # 16 floats * 8 bytes each


## Symbolic representation tests
def test_symbolic_evaluate_single(sym_rep):
    """Test evaluating a symbolic expression with one wrapped function"""
    expr = "x0"
    vars = [DummyBooleanFunction(TruthTableRepresentation(), AND_TRUTH_TABLE, Space.BOOLEAN_CUBE, 2)]
    
    # should evaluate AND(x0, x1)
    result = sym_rep.evaluate(np.array(3), (expr, vars), space=Space.BOOLEAN_CUBE, n_vars=2)
    assert result == 1

    result = sym_rep.evaluate(np.array(1), (expr, vars), space=Space.BOOLEAN_CUBE, n_vars=2)
    assert result == 0

def test_symbolic_evaluate_sum_two_functions(sym_rep):
    """Test evaluating a symbolic expression that sums two subfunctions"""
    expr = "x0 + x1"  # sum of outputs from two subfunctions

    # Two identical AND functions
    vars = [
        DummyBooleanFunction(TruthTableRepresentation(), AND_TRUTH_TABLE, Space.BOOLEAN_CUBE, 2),
        DummyBooleanFunction(TruthTableRepresentation(), AND_TRUTH_TABLE, Space.BOOLEAN_CUBE, 2)
    ]

    # Evaluate input [0, 0, 0, 0] (2 functions * 2 bits each)
    result = sym_rep.evaluate(np.array(0), (expr, vars), space=Space.BOOLEAN_CUBE, n_vars=4)
    assert result == 0

    # Evaluate input [1, 1, 1, 1] → each AND([1, 1]) = 1, so 1 + 1 = 2
    result = sym_rep.evaluate(np.array(15), (expr, vars), space=Space.BOOLEAN_CUBE, n_vars=4)
    assert result == 2


def test_symbolic_evaluate_batch(sym_rep):
    """Test batch evaluation of symbolic expressions"""
    expr = "x0"
    vars = [DummyBooleanFunction(TruthTableRepresentation(), AND_TRUTH_TABLE, Space.BOOLEAN_CUBE, 2)]

    inputs = np.array([0, 1, 2, 3])
    expected = np.array([0, 0, 0, 1])
    result = sym_rep.evaluate(inputs, (expr, vars), space=Space.BOOLEAN_CUBE, n_vars=2)
    assert np.array_equal(result, expected)


def test_symbolic_create_empty(sym_rep):
    """Test empty symbolic creation"""
    empty_expr, var_list = sym_rep.create_empty(3)
    assert empty_expr == ""
    assert var_list == ["x0", "x1", "x2"]

def test_symbolic_dump(sym_rep):
    """Test dumping symbolic data"""
    expr = "x0 and x1"
    vars = ["x0", "x1"]
    dumped = sym_rep.dump((expr, vars))
    assert dumped["expression"] == "x0 and x1"
    assert dumped["variables"] == ["x0", "x1"]


## Conversion Tests ##
def test_truth_table_to_fourier_conversion(tt_rep, fourier_rep):
    """Test conversion from truth table to Fourier coefficients"""
    # Convert AND truth table to Fourier coefficients
    fourier_coeffs = fourier_rep.convert_from(tt_rep, AND_TRUTH_TABLE, space = boo_cube, n_vars = 2)
    
    # Should match known Fourier coefficients
    assert fourier_coeffs.shape == (4,)
    assert np.allclose(fourier_coeffs, AND_FOURIER_COEFFS, atol=1e-5)

def test_fourier_to_truth_table_conversion(tt_rep, fourier_rep):
    """Test conversion from Fourier coefficients to truth table"""
    # Convert Fourier coefficients to truth table
    truth_table = tt_rep.convert_from(fourier_rep, AND_FOURIER_COEFFS, space = boo_cube, n_vars = 2)
    
    # Should match known truth table
    assert truth_table.shape == (4,)
    assert np.array_equal(truth_table, AND_TRUTH_TABLE)



## Edge Case Tests ##


