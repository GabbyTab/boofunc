import pytest
import numpy as np
import boolfunc as bf
from boolfunc.core.spaces import Space


@pytest.fixture
def xor_tt():
    return np.array([0, 1, 1, 0], dtype=bool)



@pytest.fixture
def and_tt():
    return np.array([0, 0, 0, 1], dtype=bool)


### 1. Test creation + evaluation ###
def test_create_and_evaluate_xor(xor_tt):
    f = bf.create(xor_tt)
    inputs = np.arange(4)
    expected = np.array([0, 1, 1, 0])

    result = f.evaluate(inputs)
    assert np.array_equal(result, expected)


def test_create_and_evaluate_and(and_tt):
    f = bf.create(and_tt)
    inputs = np.arange(4)
    expected = np.array([0, 0, 0, 1])

    result = f.evaluate(inputs)
    assert np.array_equal(result, expected)


### 2. Conversion between representations ###
def test_conversion_truth_table_to_fourier(and_tt):
    f = bf.create(and_tt)
    # Trigger conversion
    coeffs = f.get_representation('fourier_expansion')

    expected = np.array([0.5, 0.5, 0.5, -0.5])

    assert np.allclose(coeffs, expected, atol=1e-5)

def test_conversion_fourier_to_truth_table():
    coeffs = np.array([0.5, 0.5, 0.5, -0.5])
    f = bf.create(coeffs)

    # Convert to truth table
    table = f.get_representation('truth_table')
    

    expected = np.array([0, 0, 0, 1])
    assert np.allclose(table, expected, atol=1e-5)


### 3. Operator Composition ###
def test_composition_with_addition(xor_tt, and_tt):
    xor_func = bf.create(xor_tt)
    and_func = bf.create(and_tt)

    composed = and_func + xor_func
    inputs = np.arange(16)
    # 0000 0001 0010 ...
    # and(00) + xor(00), and(00) + xor(01), ...

    # XOR + AND
    # note that since both functions are boolean and over the same space, the output is also boolean and over the same space so 1 + 1  = 0
     
    expected = np.array([0, 1, 1, 0,
                         0, 1, 1, 0,
                         0, 1, 1, 0,
                         1, 0, 0, 1], dtype=bool)    
    result = composed.evaluate(inputs)
    assert np.array_equal(result, expected)


def test_composition_with_multiplication(xor_tt, and_tt):
    xor_func = bf.create(xor_tt)
    and_func = bf.create(and_tt)

    composed = xor_func * and_func
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # XOR * AND
    # [0*0, 1*0, 1*0, 0*1] → [0, 0, 0, 0]
    expected = np.array([0, 0, 0, 0])
    result = composed.evaluate(inputs)
    assert np.array_equal(result, expected)


def test_operator_inversion(and_tt):
    and_func = bf.create(and_tt)
    inverted = ~and_func

    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    expected = np.array([1, 1, 1, 0])  # Logical NOT of [0, 0, 0, 1]
    result = inverted.evaluate(inputs)
    assert np.array_equal(result, expected)


### 4. Space-preserving composition (advanced) ###
def test_composition_space_preservation():
    f1 = bf.create([0, 1, 1, 0], space="boolean_cube")
    f2 = bf.create([0, 0, 0, 1], space="boolean_cube")

    composed = f1 ^ f2
    assert composed.space == f1.space
    assert composed.space == f2.space


### 5. Symbolic Representation End-to-End ###
def test_symbolic_representation_and_evaluation():
    f = bf.create("x0 and x1", variables=["x0", "x1"])
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    expected = np.array([0, 0, 0, 1])

    result = f.evaluate(inputs)
    assert np.array_equal(result, expected)

    # Check internal representation
    sym = f.get_representation("symbolic")
    assert isinstance(sym, tuple)
    assert sym[0] == "x0 and x1"


### 6. Conversion chaining ###
def test_chain_conversion_xor():
    xor = np.array([0, 1, 1, 0])
    f = bf.create(xor)

    f.to('fourier_expansion')
    f.to('truth_table')
    assert np.array_equal(f.get_representation('truth_table'), xor)

# 9. Integration Tests
@pytest.mark.parametrize("input_data,expected_output", [
    (np.array([0,0]), False),
    (np.array([0,1]), True),
    (np.array([1,0]), True),
    (np.array([1,1]), False)
])
def test_xor_integration(input_data, expected_output):
    bf_instance = bf.create([0, 1, 1, 0], rep_type="truth_table")
    assert bf_instance.evaluate(input_data, bit_strings=True) == expected_output

