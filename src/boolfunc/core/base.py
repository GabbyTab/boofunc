import warnings
import operator
from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any, Optional
import numpy as np
from ..spaces import BooleanCube
from .errormodels import ExactErrorModel
#The BooleanFunctionRepresentations and Spaces and ErrorModels are in separate files in the same directory, should I import them? 

try:
    from numba import jit
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False
    warnings.warn("Numba not installed - using pure Python mode")

class Evaluable(Protocol):
    def evaluate(self, inputs): ...

class Representable(Protocol):
    def to_representation(self, rep_type: str): ...

class BooleanFunction(Evaluable, Representable):
    def __init__(self, space: str = 'boolean_cube', error_model: Optional[Any] = None, **kwargs):
        self.space = self._create_space(space)
        self.representations: Dict[str, Any] = {}
        self.properties = PropertyStore()
        self.error_model = error_model or ExactErrorModel()
        self.tracking = kwargs.get('tracking')
        self.restrictions = kwargs.get('restrictions')
        self.n_vars = kwargs.get('n')
        self._metadata = kwargs.get('metadata', {})

    def __add__(self, other):
        return composite_boolean_function(operator.add, self, other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return ScalarMultiple(other, self)
        return CompositeBooleanFunction(operator.mul, self, other)

    def __and__(self, other):
        return CompositeBooleanFunction(operator.and_, self, other)

    def __or__(self, other):
        return CompositeBooleanFunction(operator.or_, self, other)

    def __xor__(self, other):
        return CompositeBooleanFunction(operator.xor, self, other)

    def __invert__(self):
        return CompositeBooleanFunction(operator.invert, self, None) # how is called imp

    def __pow__(self, exponent):
        if not isinstance(exponent, int) or exponent < 0:
            raise ValueError("Exponent must be a non-negative integer")
        return Compose([self] * exponent)

    def __call__(self, inputs):
        return self.evaluate(inputs)

    def __str__(self):
        return f"BooleanFunction(vars={self.n_vars}, space={self.space})"

    def __repr__(self):
        return f"BooleanFunction(space={self.space}, n_vars={self.n_vars})"

    @classmethod
    def create(cls, data=None, **kwargs):
        if data is None:
            return cls(**kwargs)
        if hasattr(data, '__call__'):
            return cls.from_function(data, **kwargs)
        elif hasattr(data, 'rvs'):
            return cls.from_scipy_distribution(data, **kwargs)
        elif isinstance(data, dict):
            return cls.from_polynomial(data, **kwargs)
        elif hasattr(data, '__iter__'):
            return cls.from_truth_table(data, **kwargs)
        else:
            raise TypeError(f"Cannot create BooleanFunction from {type(data)}")


    @classmethod
    def composite_boolean_function(cls, operator, boolean_func, other, **kwargs):
        """Create from truth table data"""
        _symbolic = f"({left.get_representation("symbolic")} {op.__name__} {right.get_representation("symbolic")})"
        instance = cls(**kwargs)
        instance._add_representation('symbolic', _symbolic)
        return instance

    @classmethod
    def from_truth_table(cls, truth_table, **kwargs):
        """Create from truth table data"""
        instance = cls(**kwargs)
        instance._add_representation('truth_table', truth_table)
        return instance
    
    @classmethod
    def from_function(cls, func, domain_size=None, **kwargs):
        """Create from callable function"""
        instance = cls(**kwargs)
        instance._add_representation('function', func)
        if domain_size:
            instance.n_vars = int(np.log2(domain_size))
        return instance
    
    @classmethod
    def from_scipy_distribution(cls, distribution, **kwargs):
        """Create from scipy.stats distribution"""
        instance = cls(**kwargs)
        instance._add_representation('distribution', distribution)
        instance._setup_probabilistic_interface()
        return instance
    
    def _add_representation(self, rep_type, data):
        """Add a representation with validation"""
        self.representations[rep_type] = data
        if rep_type == 'truth_table' and self.n_vars is None:
            self.n_vars = int(np.log2(len(data)))
    
    def _setup_probabilistic_interface(self):
        """Configure as scipy.stats-like random variable"""
        # Add methods that make this behave like rv_discrete/rv_continuous
        self._configure_sampling_methods()

    def _create_space(self, space_type: str):
        if space_type == 'boolean_cube':
            return None #BooleanCube()
        # Add other space types here
        raise ValueError(f"Unknown space type: {space_type}")

    def get_representation(self, rep_type: str):
        """Retrieve or compute representation"""
        if self.representations[rep_type] is None:
            self.representations[rep_type] = self._compute_representation(rep_type)
        return self.representations[rep_type]

    def _compute_representation(self, rep_type: str):
        # Implement conversion logic here
        pass

    def evaluate(self, inputs, **kwargs):
        """Evaluate function with automatic input type detection"""
        if hasattr(inputs, 'rvs'):  # scipy.stats random variable
            return self._evaluate_stochastic(inputs, **kwargs)
        elif isinstance(inputs, (list, np.ndarray)):
            return self._evaluate_deterministic(inputs)
        else:
            raise TypeError(f"Unsupported input type: {type(inputs)}")

    def _evaluate_stochastic(self, rv_inputs, n_samples=1000):
        """Handle random variable inputs using Monte Carlo"""
        samples = rv_inputs.rvs(size=n_samples)
        results = [self._evaluate_deterministic(sample) for sample in samples]
        return self._create_result_distribution(results)

    def evaluate_range(self, inputs):
        pass

    def rvs(self, size=1, rng=None):
        """Generate random samples (like scipy.stats)"""
        if 'distribution' in self.representations:
            return self.representations['distribution'].rvs(size=size, random_state=rng)
        # Fallback: uniform sampling from truth table
        return self._uniform_sample(size, rng)
    
    def pmf(self, x):
        """Probability mass function"""
        if hasattr(self, '_pmf_cache'):
            return self._pmf_cache.get(tuple(x), 0.0)
        return self._compute_pmf(x)
    
    def cdf(self, x):
        """Cumulative distribution function"""
        return self._compute_cdf(x)



class Property:
    def __init__(self, name, test_func=None, doc=None, closed_under=None):
        self.name = name
        self.test_func = test_func
        self.doc = doc
        self.closed_under = closed_under or set()

class PropertyStore:
    def __init__(self):
        self._properties = {}

    def add(self, prop: Property, status="user"):
        self._properties[prop.name] = {"property": prop, "status": status}

    def has(self, name):
        return name in self._properties
