import pytest
import numpy as np
import operator
from unittest.mock import MagicMock, patch
import boolfunc as bf
from boolfunc.core.representations.truth_table import TruthTableRepresentation


# Fixtures for reusable test objects

# Fixture for a BooleanFunction instance
@pytest.fixture
def xor_function():
    """XOR function with truth table representation"""
    func = bf.create(n=2)
    func.representations = {
        'truth_table': np.array([0, 1, 1, 0])
    }
    return func

@pytest.fixture
def boolean_function():
    """XOR function with truth table representation"""
    func = bf.create(n=2)
    func.representations = {
        'truth_table': np.array([0, 1, 1, 0])
    }
    return func

# Fixture for a scalar value
@pytest.fixture
def scalar_value():
    return 3.5

# Fixture for another BooleanFunction
@pytest.fixture
def and_function():
    func = bf.create(n=2)
    func.representations = {
        'truth_table': np.array([0, 0, 0, 1])
    }
    return func


@pytest.fixture
def mock_strategy():
    """
    Provide a mock strategy whose evaluate() returns a fixed boolean array.
    """
    strategy = MagicMock(spec=TruthTableRepresentation)
    # Configure the mock to return [True, False] when evaluate() is called
    strategy.evaluate.return_value = np.array([0, 1, 1, 0], dtype=bool)
    return strategy


@pytest.fixture
def boolean_function():
    bf_instance = bf.BooleanFunction(space='plus_minus_cube', n=2)
    bf_instance.representations = {
        'truth_table': np.array([0, 1, 1, 0]),
        'function': lambda x: x[0] ^ x[1]
    }
    return bf_instance

# 1. Initialization and Space Handling
class TestBooleanFunctionInit:
    def test_default_initialization(self):
        bf_instance = bf.BooleanFunction()
        assert bf_instance.space == bf.Space.PLUS_MINUS_CUBE
        assert bf_instance.representations == {}
        assert isinstance(bf_instance.error_model, bf.ExactErrorModel)
        assert bf_instance.n_vars is None

    @pytest.mark.parametrize("space_str,expected_space", [
        ('boolean_cube', bf.Space.BOOLEAN_CUBE),
        ('plus_minus_cube', bf.Space.PLUS_MINUS_CUBE),
        ('real', bf.Space.REAL),
        ('log', bf.Space.LOG),
        ('gaussian', bf.Space.GAUSSIAN)
    ])
    def test_space_creation(self, space_str, expected_space):
        bf_instance = bf.BooleanFunction(space=space_str)
        assert bf_instance.space == expected_space

    def test_invalid_space_raises(self):
        with pytest.raises(ValueError, match="Unknown space type"):
            bf.BooleanFunction(space='invalid_space')

# 2. Factory Methods
class TestFactoryMethods:
    @pytest.mark.parametrize("input_data,expected_method", [
        ([0, 1, 1, 0], "from_polynomial"),
        (lambda x: x[0] & x[1], "from_function"),
        ({"x0": 1, "x1": 0}, "from_polynomial"),
        ("x0 and x1", "from_symbolic"),
        ({(0,1), (1,0)}, "from_input_invariant_truth_table"),
        (np.array([0, 1, 1, 0]), "from_polynomial"),
        (np.array([False, True, True, False]), "from_truth_table"),
        (np.array([1.0, 0.0, 0.0, 1.0]), "from_multilinear")
    ])
    def test_create_dispatch(self, input_data, expected_method):
        with patch(f'boolfunc.core.BooleanFunctionFactory.{expected_method}') as mock_method:
            mock_method.return_value = "created_instance"
            result = bf.create(input_data)
            assert result == "created_instance"

    def test_truth_table_creation(self):
        tt = [False, True, False, True]
        bf_instance = bf.create(tt)
        assert np.array_equal(bf_instance.representations['truth_table'], tt)
        assert bf_instance.n_vars == 2

    def test_symbolic_creation(self):
        expr = "x0 and not x1"
        bf_instance = bf.create(expr, variables=['x0', 'x1'])
        assert bf_instance.representations['symbolic'] == (expr, ['x0', 'x1'])

# 3. Representation Management
class TestRepresentations:
    def test_add_representation(self, boolean_function):
        new_rep = np.array([1, 0, 0, 1])
        boolean_function.add_representation(new_rep, 'polynomial')
        assert 'polynomial' in boolean_function.representations
        assert np.array_equal(boolean_function.representations['polynomial'], new_rep)

    def test_get_representation(self, boolean_function):
        tt = boolean_function.get_representation('truth_table')
        assert np.array_equal(tt, np.array([0, 1, 1, 0]))

    def test_missing_representation(self, boolean_function):
        with pytest.raises(KeyError):
            boolean_function.get_representation('bdd')

class TestEvaluation:
    @patch('boolfunc.core.base.get_strategy')
    def test_deterministic_evaluation(self, mock_get_strategy,
                                      boolean_function, mock_strategy):
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        # Act: call the method under test

        mock_get_strategy.return_value = mock_strategy

        result = boolean_function._evaluate_deterministic(inputs, rep_type='truth_table')

        # Assert: ensure evaluate() was invoked with correct arguments
        mock_get_strategy.assert_called_once_with('truth_table')

        #mock_strategy.evaluate.assert_called_once_with(
        #    inputs, boolean_function.representations['truth_table']
        #)
        # Assert: the return value matches the mock’s configured return_value
        np.testing.assert_array_equal(result, np.array([0, 1, 1, 0], dtype=bool))

    @patch('boolfunc.core.BooleanFunction._evaluate_stochastic')
    def test_stochastic_evaluation(self, mock_stochastic, boolean_function):
        mock_rv = MagicMock()
        mock_stochastic.return_value = "dist_result"
        result = boolean_function.evaluate(mock_rv, n_samples=500)
        assert result == "dist_result"
        mock_stochastic.assert_called_once_with(mock_rv, n_samples=500)

    def test_auto_representation_selection(self, boolean_function):
        with patch('boolfunc.core.BooleanFunction._evaluate_deterministic') as mock_eval:
            mock_eval.return_value = [False, True, True, False]
            inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
            results = boolean_function.evaluate(inputs)
            mock_eval.assert_called_once_with(inputs, rep_type=None)

# 5. Operator Overloading
class TestOperators:
    def test_and_operator(self, boolean_function):
        with patch('boolfunc.core.factory.BooleanFunctionFactory.create_composite') as mock_factory:
            other = bf.create([0,0,0,1])
            _ = boolean_function & other
            mock_factory.assert_called_once_with(operator.and_, boolean_function, other)

    def test_invert_operator(self, boolean_function):
        with patch('boolfunc.core.factory.BooleanFunctionFactory.create_composite') as mock_factory:
            _ = ~boolean_function
            mock_factory.assert_called_once_with(operator.invert, boolean_function, None)

# 6. Property System
class TestProperties:
    def test_property_management(self):
        bf_instance = bf.BooleanFunction()
        prop = bf.Property("linear", test_func=lambda f: True)
        bf_instance.properties.add(prop, status="verified")
        assert "linear" in bf_instance.properties._properties
        assert bf_instance.properties._properties["linear"]["status"] == "verified"

# 7. String Representations
class TestStringRepresentations:
    def test_str_representation(self, boolean_function):
        rep = str(boolean_function)
        assert "BooleanFunction" in rep
        assert "vars=2" in rep
        assert "space=Space.PLUS_MINUS_CUBE" in rep

    def test_repr_representation(self, boolean_function):
        rep = repr(boolean_function)
        assert "BooleanFunction" in rep
        assert "space=Space.PLUS_MINUS_CUBE" in rep
        assert "n_vars=2" in rep

# 8. Probabilistic Interface
class TestProbabilisticInterface:
    @patch('boolfunc.core.BooleanFunction._uniform_sample')
    def test_rvs_without_distribution(self, mock_sample, boolean_function):
        mock_sample.return_value = [0, 1, 0]
        samples = boolean_function.rvs(size=3)
        assert samples == [0, 1, 0]
        mock_sample.assert_called_once_with(3, None)

    def test_pmf_with_cache(self, boolean_function):
        boolean_function._pmf_cache = {(1, 0): 0.3}
        assert boolean_function.pmf([1, 0]) == 0.3
        assert boolean_function.pmf([0, 0]) == 0.0

# 9. Integration Tests
@pytest.mark.parametrize("input_data,expected_output", [
    (np.array([0,0]), False),
    (np.array([0,1]), True),
    (np.array([1,0]), True),
    (np.array([1,1]), False)
])
def test_xor_integration(input_data, expected_output):
    bf_instance = bf.create([0, 1, 1, 0], rep_type="truth_table")
    assert bf_instance.evaluate(input_data) == expected_output



## 1. __array__ method
def test_array_conversion(xor_function):
    """Test conversion to NumPy array"""
    arr = np.array(xor_function)
    assert isinstance(arr, np.ndarray)
    assert np.array_equal(arr, [0, 1, 1, 0])
    
    # Test with dtype conversion
    bool_arr = np.array(xor_function, dtype=bool)
    assert np.array_equal(bool_arr, [False, True, True, False])

## 2. Operator methods
@patch('boolfunc.core.factory.BooleanFunctionFactory.create_composite')
def test_binary_operators(mock_factory, xor_function, and_function):
    """Test +, *, &, |, ^ operators"""
    # Test addition
    _ = xor_function + and_function
    mock_factory.assert_called_with(operator.add, xor_function, and_function)
    
    # Test multiplication with function
    _ = xor_function * and_function
    mock_factory.assert_called_with(operator.mul, xor_function, and_function)
    
    # Test AND
    _ = xor_function & and_function
    mock_factory.assert_called_with(operator.and_, xor_function, and_function)
    
    # Test OR
    _ = xor_function | and_function
    mock_factory.assert_called_with(operator.or_, xor_function, and_function)
    
    # Test XOR
    _ = xor_function ^ and_function
    mock_factory.assert_called_with(operator.xor, xor_function, and_function)

def test_scalar_multiplication(xor_function, scalar_value):
    """Test multiplication with scalar"""
    with patch('boolfunc.core.factory.BooleanFunctionFactory.create_composite') as mock_scalar:
        _ = xor_function * scalar_value
        mock_scalar.assert_called_with(operator.mul, xor_function, scalar_value)

## 3. Unary operators
@patch('boolfunc.core.factory.BooleanFunctionFactory.create_composite')
def test_unary_operators(mock_factory, xor_function):
    """Test ~ (invert) and ** operators"""
    # Test inversion
    _ = ~xor_function
    mock_factory.assert_called_with(operator.invert, xor_function, None)
    
    # Test exponentiation
    _ = xor_function ** 2
    mock_factory.assert_called_with(operator.pow, xor_function, None)

## 4. __call__ method
def test_call_method(xor_function):
    """Test function call syntax"""
    with patch.object(xor_function, 'evaluate') as mock_eval:
        inputs = [0, 1]
        _ = xor_function(inputs)
        mock_eval.assert_called_once_with(inputs)

## 5. String representations
def test_string_representations(xor_function):
    """Test __str__ and __repr__ methods"""
    # Test __str__
    assert "BooleanFunction" in str(xor_function)
    assert "vars=2" in str(xor_function)
    assert "space=Space.PLUS_MINUS_CUBE" in str(xor_function)
    
    # Test __repr__
    repr_str = repr(xor_function)
    assert "BooleanFunction" in repr_str
    assert "space=Space.PLUS_MINUS_CUBE" in repr_str
    assert "n_vars=2" in repr_str

## 6. Edge cases
def test_missing_truth_table():
    """Test __array__ without truth table representation"""
    func = bf.BooleanFunction()
    with pytest.raises(KeyError):
        np.array(func)
        
def test_invalid_operand_type(xor_function):
    """Test operators with invalid types"""
    with pytest.raises(TypeError):
        _ = xor_function + "invalid"
