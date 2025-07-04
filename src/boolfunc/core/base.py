import warnings
import operator
from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any, Optional
import numpy as np
from .errormodels import ExactErrorModel
from collections.abc import Iterable
from .spaces import Space
from .representations.registry import get_strategy
from .factory import BooleanFunctionFactory

#The BooleanFunctionRepresentations and Spaces and ErrorModels are in separate files in the same directory, should I import them? 

try:
    from numba import jit
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False
    warnings.warn("Numba not installed - using pure Python mode")


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


class Evaluable(Protocol):
    def evaluate(self, inputs): ...

class Representable(Protocol):
    def to_representation(self, rep_type: str): ...



class BooleanFunction(Evaluable, Representable):
    def __new__(cls, *args, **kwargs):
        # Allocate without calling __init__
        self = super().__new__(cls)
        # Delegate actual setup to a private initializer
        self._init(*args, **kwargs)
        return self

    def _init(self, space: str = 'plus_minus_cube', error_model: Optional[Any] = None, storage_manager=None, **kwargs):
        # Original __init__ logic moved here
        self.space = self._create_space(space)
        self.representations: Dict[str, Any] = {}
        self.properties = PropertyStore()
        self.error_model = error_model or ExactErrorModel()
        self.tracking = kwargs.get('tracking')
        self.restrictions = kwargs.get('restrictions')
        self.n_vars = kwargs.get('n') or kwargs.get('n_vars')
        self._metadata = kwargs.get('metadata', {})
        self.nickname = kwargs.get('nickname') or "x_0"



    def __array__(self, dtype=None) -> np.ndarray:
        """Return the truth table as a NumPy array for NumPy compatibility."""
        truth_table = self.get_representation('truth_table')
        return np.asarray(truth_table, dtype=dtype)

    def __add__(self, other):
        return BooleanFunctionFactory.create_composite(
            boolean_function_cls=type(self),  # or BooleanFunction if that's the base class
            operator="+",
            left_func=self,
            right_func=other
        )
   
    def __sub__(self, other):
        return BooleanFunctionFactory.create_composite(
            boolean_function_cls=type(self),  # or BooleanFunction if that's the base class
            operator="-",
            left_func=self,
            right_func=other
        )


    def __mul__(self, other):
        return BooleanFunctionFactory.create_composite(
            boolean_function_cls=type(self),  # or BooleanFunction if that's the base class
            operator="*",
            left_func=self,
            right_func=other
        )


    def __and__(self, other):
        return BooleanFunctionFactory.create_composite(
            boolean_function_cls=type(self),  # or BooleanFunction if that's the base class
            operator="&",
            left_func=self,
            right_func=other
        )
    def __or__(self, other):
        return BooleanFunctionFactory.create_composite(
            boolean_function_cls=type(self),  # or BooleanFunction if that's the base class
            operator="|",
            left_func=self,
            right_func=other
        )
    def __xor__(self, other):
        return BooleanFunctionFactory.create_composite(
            boolean_function_cls=type(self),  # or BooleanFunction if that's the base class
            operator="^",
            left_func=self,
            right_func=other
        )
    def __invert__(self):
        return BooleanFunctionFactory.create_composite(
            boolean_function_cls=type(self),  # or BooleanFunction if that's the base class
            operator= "~",
            left_func=self,
            right_func=None
        )
    def __pow__(self, exponent):
        return BooleanFunctionFactory.create_composite(
            boolean_function_cls=type(self),  # or BooleanFunction if that's the base class
            operator="**",
            left_func=self,
            right_func=None
        )
    def __call__(self, inputs):
        return self.evaluate(inputs)

    def __str__(self):
        return f"BooleanFunction(vars={self.n_vars}, space={self.space})" #TODO figure out what should be outputed here

    def __repr__(self):
        return f"BooleanFunction(space={self.space}, n_vars={self.n_vars})" #TODO figure out what should be outputed here

    def _create_space(self, space_type: str):
        if space_type == 'boolean_cube':
            return Space.BOOLEAN_CUBE
        elif space_type == 'plus_minus_cube':
            return Space.PLUS_MINUS_CUBE
        elif space_type == 'real':
            return Space.REAL
        elif space_type == 'log':
            return Space.LOG
        elif space_type == 'gaussian':
            return Space.GAUSSIAN
        else:
            raise ValueError(f"Unknown space type: {space_type}")
    
    def _compute_representation(self, rep_type: str):
        # Implement conversion logic here - Compute from nearest representation or run Dijkstra's 
        # if no representations, the function should error
        if rep_type in self.representations:
            return None

        source_rep_type = next(iter(self.representations))

        if source_rep_type is None:
            raise KeyError("Boolean Function is Empty (no representations)")
        
        data = self.representations[source_rep_type] 
        source_strategy = get_strategy(source_rep_type)
        target_strategy = get_strategy(rep_type)

        result = source_strategy.convert_to(target_strategy, data, self.space, self.n_vars)

        self.add_representation(result, rep_type)
        return None
      
    def get_representation(self, rep_type: str):
        """Retrieve or compute representation"""
        self._compute_representation(rep_type)
        rep_data = self.representations[rep_type]

        return rep_data

    def add_representation(self, data, rep_type = None):
        """Add a representation to this boolean function"""
        if rep_type == None:
            factory = BooleanFunctionFactory()
            rep_type = factory._determine_rep_type(data)
        
        self.representations[rep_type] = data
        return self



    def evaluate(self, inputs, rep_type=None, **kwargs):
        """
        Evaluate function with automatic input type detection and representation selection.
        
        Args:
            inputs: Input data (array, list, or scipy random variable)
            representation: Optional specific representation to use
            **kwargs: Additional evaluation parameters
        
        Returns:
            Boolean result(s) or distribution
        """
        bit_strings = False or kwargs.get('bit_strings')
        if bit_strings:
            inputs = self._compute_index(inputs)

        if hasattr(inputs, 'rvs'):  # scipy.stats random variable
            return self._evaluate_stochastic(inputs, rep_type=rep_type, **kwargs)
        elif isinstance(inputs, (list, np.ndarray)):
            return self._evaluate_deterministic(inputs, rep_type=rep_type)
        else:
            raise TypeError(f"Unsupported input type: {type(inputs)}")
    
    
    def _compute_index(self, bits: np.ndarray) -> int:
        """Convert boolean vector to integer index using bit packing"""
        return np.array(int(np.packbits(bits.astype(np.uint8), bitorder='little')[0]))
   

    def _evaluate_deterministic(self, inputs, rep_type=None):
        """
        Evaluate using the specified or first available representation.
        """
        inputs = np.asarray(inputs)
        if rep_type == None:
            rep_type = next(iter(self.representations))
       
        data = self.representations[rep_type]     
        strategy = get_strategy(rep_type)
        result = strategy.evaluate(inputs, data, self.space, self.n_vars)
        return result

        
    def _setup_probabilistic_interface(self):
        """Configure as scipy.stats-like random variable"""
        # Add methods that make this behave like rv_discrete/rv_continuous
        #self._configure_sampling_methods()
        pass


    def _evaluate_stochastic(self, rv_inputs, n_samples=1000):
        """Handle random variable inputs using Monte Carlo"""
        pass
        samples = rv_inputs.rvs(size=n_samples)
        results = [self._evaluate_deterministic(sample) for sample in samples]
        return self._create_result_distribution(results)

    def evaluate_range(self, inputs):
        pass

    def rvs(self, size=1, rng=None):
        """Generate random samples (like scipy.stats)"""
        pass
        if 'distribution' in self.representations:
            return self.representations['distribution'].rvs(size=size, random_state=rng)
        # Fallback: uniform sampling from truth table
        return self._uniform_sample(size, rng)
    
    def pmf(self, x):
        pass
        """Probability mass function"""
        if hasattr(self, '_pmf_cache'):
            return self._pmf_cache.get(tuple(x), 0.0)
        return self._compute_pmf(x)
    
    def cdf(self, x):
        pass
        """Cumulative distribution function"""
        #return self._compute_cdf(x)


    #get methods
    def get_n_vars(self):
        return self.n_vars

    #get methods
    def has_rep(self, rep_type):
        if rep_type in self.representations:
            return True
        return False

