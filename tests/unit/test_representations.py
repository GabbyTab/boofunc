import pytest
import numpy as np
from representations import (
    TruthTableRepresentation,
    FourierExpansionRepresentation,
    BooleanFunctionRepresentation
)

# Test data for 2-variable functions
AND_TRUTH_TABLE = np.array([0, 0, 0, 1])  # [00, 01, 10, 11]
AND_FOURIER_COEFFS = np.array([0.5, 0.5, 0.5, -0.5])  # ∅, {0}, {1}, {0,1}

XOR_TRUTH_TABLE = np.array([0, 1, 1, 0])
XOR_FOURIER_COEFFS = np.array([0, 0, 0, 1])  # Only {0,1} term

# Fixtures for common test objects
@pytest.fixture
def tt_rep():
    return TruthTableRepresentation()

@pytest.fixture
def fourier_rep():
    return FourierExpansionRepresentation()

## Truth Table Representation Tests ##

def test_truth_table_evaluate_single(tt_rep):
    """Test evaluation of single inputs"""
    # AND function
    assert tt_rep.evaluate(np.array([0, 0]), AND_TRUTH_TABLE) == 0
    assert tt_rep.evaluate(np.array([0, 1]), AND_TRUTH_TABLE) == 0
    assert tt_rep.evaluate(np.array([1, 0]), AND_TRUTH_TABLE) == 0
    assert tt_rep.evaluate(np.array([1, 1]), AND_TRUTH_TABLE) == 1
    
    # XOR function
    assert tt_rep.evaluate(np.array([0, 0]), XOR_TRUTH_TABLE) == 0
    assert tt_rep.evaluate(np.array([0, 1]), XOR_TRUTH_TABLE) == 1
    assert tt_rep.evaluate(np.array([1, 0]), XOR_TRUTH_TABLE) == 1
    assert tt_rep.evaluate(np.array([1, 1]), XOR_TRUTH_TABLE) == 0

def test_truth_table_evaluate_batch(tt_rep):
    """Test batch evaluation"""
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    
    # AND function
    results = tt_rep.evaluate(inputs, AND_TRUTH_TABLE)
    assert np.array_equal(results, np.array([0, 0, 0, 1]))
    
    # XOR function
    results = tt_rep.evaluate(inputs, XOR_TRUTH_TABLE)
    assert np.array_equal(results, np.array([0, 1, 1, 0]))

def test_truth_table_dump(tt_rep):
    """Test serialization of truth table"""
    dumped = tt_rep.dump(AND_TRUTH_TABLE)
    assert dumped['type'] == 'truth_table'
    assert dumped['values'] == [0, 0, 0, 1]
    assert dumped['metadata']['num_vars'] == 2
    assert dumped['metadata']['size'] == 4

def test_truth_table_create_empty(tt_rep):
    """Test empty truth table creation"""
    empty_tt = tt_rep.create_empty(3)
    assert len(empty_tt) == 8
    assert np.all(empty_tt == 0)

def test_truth_table_storage_requirements(tt_rep):
    """Test storage requirements calculation"""
    requirements = tt_rep.get_storage_requirements(3)
    assert requirements['elements'] == 8
    assert requirements['bytes'] == 8  # 8 bytes for 8 booleans

## Fourier Expansion Representation Tests ##

def test_fourier_evaluate_single(fourier_rep):
    """Test evaluation of single inputs"""
    # AND function in ±1 domain
    assert fourier_rep.evaluate(np.array([0, 0]), AND_FOURIER_COEFFS) == 1.0
    assert fourier_rep.evaluate(np.array([0, 1]), AND_FOURIER_COEFFS) == 1.0
    assert fourier_rep.evaluate(np.array([1, 0]), AND_FOURIER_COEFFS) == 1.0
    assert fourier_rep.evaluate(np.array([1, 1]), AND_FOURIER_COEFFS) == -1.0
    
    # XOR function in ±1 domain
    assert fourier_rep.evaluate(np.array([0, 0]), XOR_FOURIER_COEFFS) == 1.0
    assert fourier_rep.evaluate(np.array([0, 1]), XOR_FOURIER_COEFFS) == -1.0
    assert fourier_rep.evaluate(np.array([1, 0]), XOR_FOURIER_COEFFS) == -1.0
    assert fourier_rep.evaluate(np.array([1, 1]), XOR_FOURIER_COEFFS) == 1.0

def test_fourier_evaluate_batch(fourier_rep):
    """Test batch evaluation"""
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    
    # AND function
    results = fourier_rep.evaluate(inputs, AND_FOURIER_COEFFS)
    expected = np.array([1.0, 1.0, 1.0, -1.0])
    assert np.allclose(results, expected)
    
    # XOR function
    results = fourier_rep.evaluate(inputs, XOR_FOURIER_COEFFS)
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

## Conversion Tests ##

def test_truth_table_to_fourier_conversion(tt_rep, fourier_rep):
    """Test conversion from truth table to Fourier coefficients"""
    # Convert AND truth table to Fourier coefficients
    fourier_coeffs = fourier_rep.convert_from(tt_rep, AND_TRUTH_TABLE)
    
    # Should match known Fourier coefficients
    assert fourier_coeffs.shape == (4,)
    assert np.allclose(fourier_coeffs, AND_FOURIER_COEFFS, atol=1e-5)

def test_fourier_to_truth_table_conversion(tt_rep, fourier_rep):
    """Test conversion from Fourier coefficients to truth table"""
    # Convert Fourier coefficients to truth table
    truth_table = tt_rep.convert_from(fourier_rep, AND_FOURIER_COEFFS)
    
    # Should match known truth table
    assert truth_table.shape == (4,)
    assert np.array_equal(truth_table, AND_TRUTH_TABLE)

## Edge Case Tests ##

def test_empty_evaluation(tt_rep, fourier_rep):
    """Test evaluation with empty inputs"""
    # Truth table
    assert tt_rep.evaluate(np.array([]), AND_TRUTH_TABLE) == 0
    
    # Fourier expansion
    assert fourier_rep.evaluate(np.array([]), AND_FOURIER_COEFFS) == 0.5

def test_invalid_input_dimensions(tt_rep, fourier_rep):
    """Test invalid input dimensions"""
    with pytest.raises(ValueError):
        # 3D inputs
        tt_rep.evaluate(np.ones((2, 2, 2)), AND_TRUTH_TABLE)
    
    with pytest.raises(ValueError):
        fourier_rep.evaluate(np.ones((2, 2, 2)), AND_FOURIER_COEFFS)

def test_invalid_data_size(tt_rep, fourier_rep):
    """Test mismatched data size"""
    with pytest.raises(ValueError):
        # Truth table for 3 vars with 2-var inputs
        tt_rep.evaluate(np.array([0, 0, 1]), np.array([0]*8))
    
    with pytest.raises(ValueError):
        # Fourier coeffs for 3 vars with 2-var inputs
        fourier_rep.evaluate(np.array([0, 1]), np.ones(8))

## Property-Based Tests ##

@pytest.mark.parametrize("n_vars", [1, 2, 3])
def test_truth_table_roundtrip(n_vars, tt_rep):
    """Test truth table creation and evaluation roundtrip"""
    # Create random truth table
    size = 2**n_vars
    random_tt = np.random.choice([0, 1], size=size)
    
    # Create all possible inputs
    inputs = np.array(list(np.ndindex((2,)*n_vars)))
    
    # Evaluate all inputs
    results = tt_rep.evaluate(inputs, random_tt)
    
    # Verify results match truth table values
    for i, input_vec in enumerate(inputs):
        idx = tt_rep._compute_index(input_vec)
        assert results[i] == random_tt[idx]

@pytest.mark.parametrize("n_vars", [1, 2, 3])
def test_fourier_roundtrip(n_vars, fourier_rep):
    """Test Fourier coefficient creation and evaluation roundtrip"""
    # Create random Fourier coefficients
    size = 2**n_vars
    random_coeffs = np.random.uniform(-1, 1, size=size)
    
    # Create all possible inputs
    inputs = np.array(list(np.ndindex((2,)*n_vars)))
    
    # Evaluate all inputs
    results = fourier_rep.evaluate(inputs, random_coeffs)
    
    # Verify consistent results
    for i, input_vec in enumerate(inputs):
        # Convert to ±1 domain
        x = 1 - 2 * input_vec.astype(float)
        # Manual Fourier expansion calculation
        manual_result = 0
        for j in range(size):
            # Get subset mask
            mask = np.array([(j >> k) & 1 for k in range(n_vars)])
            # Compute character function
            char_val = np.prod(x[mask.astype(bool)])
            manual_result += random_coeffs[j] * char_val
        assert np.isclose(results[i], manual_result)
