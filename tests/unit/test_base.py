# tests/test_boolean_function.py
import pytest
import numpy as np
import warnings
from unittest.mock import Mock, patch
from typing import Dict, Any

from boolfunc.core.base import BooleanFunction, Evaluable, Representable


class TestBooleanFunctionInit:
    """Test BooleanFunction initialization and basic properties."""
    
    def test_default_initialization(self):
        """Test default initialization parameters."""
        bf = BooleanFunction()
        assert bf.space is not None
        assert isinstance(bf.representations, dict)
        assert bf.representations == {}
        assert bf.error_model is not None
        assert bf.tracking is None
        assert bf.restrictions is None
        assert bf.n_vars is None
        assert bf._metadata == {}

    def test_initialization_with_kwargs(self):
        """Test initialization with custom parameters."""
        metadata = {"author": "test"}
        bf = BooleanFunction(
            space='boolean_cube',
            tracking=True,
            restrictions=['test'],
            n=3,
            metadata=metadata
        )
        assert bf.tracking is True
        assert bf.restrictions == ['test']
        assert bf.n_vars == 3
        assert bf._metadata == metadata

    @patch('boolfunc.core.base.BooleanFunction._create_space')
    def test_space_creation_called(self, mock_create_space):
        """Test that space creation is called during initialization."""
        mock_space = Mock()
        mock_create_space.return_value = mock_space
        
        bf = BooleanFunction(space='test_space')
        mock_create_space.assert_called_once_with('test_space')
        assert bf.space == mock_space


class TestBooleanFunctionMagicMethods:
    """Test magic methods (dunder methods) of BooleanFunction."""
    
    @pytest.fixture
    def bf(self):
        """Create a BooleanFunction instance for testing."""
        return BooleanFunction(n=2)
    
    def test_str_representation(self, bf):
        """Test string representation."""
        result = str(bf)
        assert "BooleanFunction" in result
        assert "vars=2" in result
        assert "space=" in result

    def test_repr_representation(self, bf):
        """Test repr representation."""
        result = repr(bf)
        assert "BooleanFunction" in result
        assert "space=" in result
        assert "n_vars=2" in result

    def test_call_method(self, bf):
        """Test that __call__ delegates to evaluate."""
        with patch.object(bf, 'evaluate') as mock_evaluate:
            mock_evaluate.return_value = True
            inputs = [1, 0]
            result = bf(inputs)
            mock_evaluate.assert_called_once_with(inputs)
            assert result is True

    def test_pow_valid_exponent(self, bf):
        """Test __pow__ with valid exponent."""
        with patch('boolfunc.core.base.Compose') as mock_compose:
            mock_compose.return_value = "composed_function"
            result = bf ** 2
            mock_compose.assert_called_once_with([bf, bf])
            assert result == "composed_function"

    def test_pow_invalid_exponent_negative(self, bf):
        """Test __pow__ with negative exponent raises ValueError."""
        with pytest.raises(ValueError, match="Exponent must be a non-negative integer"):
            bf ** -1

    def test_pow_invalid_exponent_float(self, bf):
        """Test __pow__ with float exponent raises ValueError."""
        with pytest.raises(ValueError, match="Exponent must be a non-negative integer"):
            bf ** 2.5

    def test_mul_with_scalar(self, bf):
        """Test multiplication with scalar values."""
        with patch('boolfunc.core.base.ScalarMultiple') as mock_scalar:
            mock_scalar.return_value = "scalar_multiple"
            
            result_int = bf * 5
            result_float = bf * 2.5
            
            assert mock_scalar.call_count == 2
            assert result_int == "scalar_multiple"
            assert result_float == "scalar_multiple"

    @pytest.mark.parametrize("op_method,op_func", [
        ("__and__", "operator.and_"),
        ("__or__", "operator.or_"),
        ("__xor__", "operator.xor"),
    ])
    def test_binary_operators(self, bf, op_method, op_func):
        """Test binary operators (and, or, xor)."""
        other_bf = BooleanFunction()
        with patch('boolfunc.core.base.CompositeBooleanFunction') as mock_composite:
            mock_composite.return_value = "composite_function"
            
            method = getattr(bf, op_method)
            result = method(other_bf)
            
            assert result == "composite_function"
            mock_composite.assert_called_once()


class TestBooleanFunctionClassMethods:
    """Test class methods for creating BooleanFunction instances."""
    
    def test_create_with_none(self):
        """Test create method with None data."""
        bf = BooleanFunction.create(data=None, n=3)
        assert bf.n_vars == 3

    def test_create_with_callable(self):
        """Test create method with callable data."""
        def test_func(x):
            return x > 0
        
        with patch.object(BooleanFunction, 'from_function') as mock_from_func:
            mock_from_func.return_value = "function_bf"
            result = BooleanFunction.create(data=test_func, n=2)
            mock_from_func.assert_called_once_with(test_func, n=2)
            assert result == "function_bf"

    def test_create_with_scipy_distribution(self):
        """Test create method with scipy distribution."""
        mock_dist = Mock()
        mock_dist.rvs = Mock()
        
        with patch.object(BooleanFunction, 'from_scipy_distribution') as mock_from_dist:
            mock_from_dist.return_value = "dist_bf"
            result = BooleanFunction.create(data=mock_dist)
            mock_from_dist.assert_called_once_with(mock_dist)
            assert result == "dist_bf"

    def test_create_with_dict(self):
        """Test create method with dictionary data."""
        poly_dict = {"x1": 1, "x2": 0}
        
        with patch.object(BooleanFunction, 'from_polynomial') as mock_from_poly:
            mock_from_poly.return_value = "poly_bf"
            result = BooleanFunction.create(data=poly_dict)
            mock_from_poly.assert_called_once_with(poly_dict)
            assert result == "poly_bf"

    def test_create_with_iterable(self):
        """Test create method with iterable data."""
        truth_table = [0, 1, 1, 0]
        
        with patch.object(BooleanFunction, 'from_truth_table') as mock_from_tt:
            mock_from_tt.return_value = "tt_bf"
            result = BooleanFunction.create(data=truth_table)
            mock_from_tt.assert_called_once_with(truth_table)
            assert result == "tt_bf"

    def test_create_with_invalid_type(self):
        """Test create method with invalid data type."""
        with pytest.raises(TypeError, match="Cannot create BooleanFunction from"):
            BooleanFunction.create(data=42)

    def test_from_truth_table(self):
        """Test from_truth_table class method."""
        truth_table = [0, 1, 1, 0]
        bf = BooleanFunction.from_truth_table(truth_table, n=2)
        
        assert 'truth_table' in bf.representations
        assert bf.representations['truth_table'] == truth_table
        assert bf.n_vars == 2

    def test_from_function(self):
        """Test from_function class method."""
        def test_func(x):
            return x[0] and x[1]
        
        bf = BooleanFunction.from_function(test_func, domain_size=4)
        
        assert 'function' in bf.representations
        assert bf.representations['function'] == test_func
        assert bf.n_vars == 2


class TestBooleanFunctionEvaluation:
    """Test evaluation methods of BooleanFunction."""
    
    @pytest.fixture
    def bf(self):
        return BooleanFunction()

    def test_evaluate_with_scipy_distribution(self, bf):
        """Test evaluate with scipy.stats random variable."""
        mock_rv = Mock()
        mock_rv.rvs = Mock()
        
        with patch.object(bf, '_evaluate_stochastic') as mock_eval_stoch:
            mock_eval_stoch.return_value = "stochastic_result"
            result = bf.evaluate(mock_rv, n_samples=500)
            mock_eval_stoch.assert_called_once_with(mock_rv, n_samples=500)
            assert result == "stochastic_result"

    def test_evaluate_with_list(self, bf):
        """Test evaluate with list input."""
        inputs = [1, 0, 1]
        
        with patch.object(bf, '_evaluate_deterministic') as mock_eval_det:
            mock_eval_det.return_value = True
            result = bf.evaluate(inputs)
            mock_eval_det.assert_called_once_with(inputs)
            assert result is True

    def test_evaluate_with_numpy_array(self, bf):
        """Test evaluate with numpy array input."""
        inputs = np.array([1, 0, 1])
        
        with patch.object(bf, '_evaluate_deterministic') as mock_eval_det:
            mock_eval_det.return_value = False
            result = bf.evaluate(inputs)
            mock_eval_det.assert_called_once_with(inputs)
            assert result is False

    def test_evaluate_with_invalid_type(self, bf):
        """Test evaluate with unsupported input type."""
        with pytest.raises(TypeError, match="Unsupported input type"):
            bf.evaluate("invalid_input")


class TestBooleanFunctionRepresentations:
    """Test representation management methods."""
    
    @pytest.fixture
    def bf(self):
        return BooleanFunction()

    def test_add_representation_truth_table(self, bf):
        """Test adding truth table representation."""
        truth_table = [0, 1, 1, 0]
        bf._add_representation('truth_table', truth_table)
        
        assert bf.representations['truth_table'] == truth_table
        assert bf.n_vars == 2  # log2(4) = 2

    def test_add_representation_other_type(self, bf):
        """Test adding non-truth-table representation."""
        bf.n_vars = 3
        mock_func = Mock()
        bf._add_representation('function', mock_func)
        
        assert bf.representations['function'] == mock_func
        assert bf.n_vars == 3  # Should not change

    def test_get_representation_existing(self, bf):
        """Test getting existing representation."""
        bf.representations['test'] = "test_data"
        result = bf.get_representation('test')
        assert result == "test_data"

    def test_get_representation_compute(self, bf):
        """Test getting representation that needs computation."""
        bf.representations['test'] = None
        
        with patch.object(bf, '_compute_representation') as mock_compute:
            mock_compute.return_value = "computed_data"
            result = bf.get_representation('test')
            mock_compute.assert_called_once_with('test')
            assert result == "computed_data"
            assert bf.representations['test'] == "computed_data"


class TestBooleanFunctionSpaces:
    """Test space-related functionality."""
    
    def test_create_space_boolean_cube(self):
        """Test creating boolean cube space."""
        with patch('boolfunc.core.base.BooleanCube') as mock_cube:
            mock_cube.return_value = "cube_space"
            bf = BooleanFunction()
            result = bf._create_space('boolean_cube')
            mock_cube.assert_called_once()
            assert result == "cube_space"

    def test_create_space_invalid(self):
        """Test creating invalid space type."""
        bf = BooleanFunction.__new__(BooleanFunction)  # Skip __init__
        with pytest.raises(ValueError, match="Unknown space type"):
            bf._create_space('invalid_space')


class TestBooleanFunctionProbabilistic:
    """Test probabilistic interface methods."""
    
    @pytest.fixture
    def bf(self):
        return BooleanFunction()

    def test_rvs_with_distribution(self, bf):
        """Test rvs method with distribution representation."""
        mock_dist = Mock()
        mock_dist.rvs = Mock(return_value=[1, 0, 1])
        bf.representations['distribution'] = mock_dist
        
        result = bf.rvs(size=3, rng=42)
        mock_dist.rvs.assert_called_once_with(size=3, random_state=42)
        assert result == [1, 0, 1]

    def test_rvs_without_distribution(self, bf):
        """Test rvs method without distribution representation."""
        with patch.object(bf, '_uniform_sample') as mock_uniform:
            mock_uniform.return_value = [0, 1]
            result = bf.rvs(size=2, rng=123)
            mock_uniform.assert_called_once_with(2, 123)
            assert result == [0, 1]

    def test_pmf_with_cache(self, bf):
        """Test pmf method with cached values."""
        bf._pmf_cache = {(1, 0): 0.3, (0, 1): 0.7}
        assert bf.pmf([1, 0]) == 0.3
        assert bf.pmf([0, 1]) == 0.7
        assert bf.pmf([1, 1]) == 0.0

    def test_pmf_without_cache(self, bf):
        """Test pmf method without cache."""
        with patch.object(bf, '_compute_pmf') as mock_compute_pmf:
            mock_compute_pmf.return_value = 0.5
            result = bf.pmf([1, 0])
            mock_compute_pmf.assert_called_once_with([1, 0])
            assert result == 0.5


@pytest.fixture(scope="session")
def sample_boolean_functions():
    """Session-scoped fixture providing sample boolean functions for testing."""
    return {
        'and_function': BooleanFunction.from_truth_table([0, 0, 0, 1]),
        'or_function': BooleanFunction.from_truth_table([0, 1, 1, 1]),
        'xor_function': BooleanFunction.from_truth_table([0, 1, 1, 0]),
    }


class TestIntegration:
    """Integration tests combining multiple features."""
    
    def test_complete_workflow(self, sample_boolean_functions):
        """Test complete workflow from creation to evaluation."""
        bf = sample_boolean_functions['and_function']
        
        # Test string representations
        assert "BooleanFunction" in str(bf)
        assert "BooleanFunction" in repr(bf)
        
        # Test representation access
        truth_table = bf.get_representation('truth_table')
        assert truth_table == [0, 0, 0, 1]
        
        # Test properties
        assert bf.n_vars == 2
        assert 'truth_table' in bf.representations


# Parametrized tests for comprehensive coverage
@pytest.mark.parametrize("exponent,expected_length", [
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
])
def test_pow_exponent_variations(exponent, expected_length):
    """Test __pow__ method with various valid exponents."""
    bf = BooleanFunction()
    with patch('boolfunc.core.base.Compose') as mock_compose:
        if exponent == 0:
            mock_compose.return_value = "empty_compose"
            result = bf ** exponent
            mock_compose.assert_called_once_with([])
        else:
            mock_compose.return_value = f"compose_{exponent}"
            result = bf ** exponent
            mock_compose.assert_called_once_with([bf] * exponent)


@pytest.mark.slow
def test_performance_large_truth_table():
    """Test performance with large truth tables."""
    large_table = [0, 1] * 512  # 1024 elements
    bf = BooleanFunction.from_truth_table(large_table)
    
    assert len(bf.representations['truth_table']) == 1024
    assert bf.n_vars == 10  # log2(1024) = 10
