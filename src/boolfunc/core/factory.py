import numpy as np
from collections.abc import Iterable
from .spaces import Space
import numbers

class BooleanFunctionFactory:
    """Factory for creating BooleanFunction instances from various representations"""


    @classmethod
    def _determine_rep_type(cls, data):
        """Determine the representation type based on data type"""
        if callable(data):
            return 'function'
        if hasattr(data, 'rvs'):
            return 'distribution'
        if isinstance(data, np.ndarray):
            if data.dtype == bool or np.issubdtype(data.dtype, np.bool_):
                return 'truth_table'
            if np.issubdtype(data.dtype, np.integer):
                return 'polynomial'
            if np.issubdtype(data.dtype, np.floating):
                return 'multilinear'
            return 'polynomial'
        if isinstance(data, list):
            return cls._determine_rep_type(np.array(data))
        if isinstance(data, dict):
            return 'polynomial'
        if isinstance(data, str):
            return 'symbolic'
        if isinstance(data, set):
            return 'invariant_truth_table'  
        if isinstance(data, Iterable):
            return 'iterable_rep'
        
        raise TypeError(f"Cannot determine representation type for {type(data)}")

    
    @classmethod
    def create(cls, boolean_function_cls, data=None, **kwargs):
        """
        Main factory method that dispatches to specialized creators
        based on input data type
        """
        if data is None:
            return boolean_function_cls(**kwargs)

        # Determine representation type and dispatch accordingly
        rep_type = kwargs.get('rep_type')
        if rep_type is None:
            rep_type = cls._determine_rep_type(data)
        
        if rep_type == 'function':
            return cls.from_function(boolean_function_cls, data, **kwargs)
        elif rep_type == 'distribution':
            return cls.from_scipy_distribution(boolean_function_cls, data, **kwargs)
        elif rep_type == 'truth_table':
            return cls.from_truth_table(boolean_function_cls, data, **kwargs)
        elif rep_type == 'invariant_truth_table':
            return cls.from_input_invariant_truth_table(boolean_function_cls, data, **kwargs)
        elif rep_type == 'polynomial':
            return cls.from_polynomial(boolean_function_cls, data, **kwargs)
        elif rep_type == 'multilinear':
            return cls.from_multilinear(boolean_function_cls, data, **kwargs)
        elif rep_type == 'symbolic':
            return cls.from_symbolic(boolean_function_cls, data, **kwargs)
        elif rep_type == 'iterable_rep':
            return cls.from_iterable(boolean_function_cls, data, **kwargs)
        
        raise TypeError(f"Cannot create BooleanFunction from {type(data)}")


    @classmethod
    def from_truth_table(cls, boolean_function_cls, truth_table, rep_type='truth_table', **kwargs):
        """Create from truth table data"""

        n_vars = kwargs.get('n')
        if n_vars is None:
            n_vars = int(np.log2(len(truth_table)))
            kwargs['n'] = n_vars
            
        instance = boolean_function_cls(**kwargs)
        instance.add_representation(truth_table, rep_type)
        return instance

    @classmethod
    def from_function(cls, boolean_function_cls, func, rep_type='function', domain_size=None, **kwargs):
        """Create from callable function"""
        instance = boolean_function_cls(**kwargs)
        instance.add_representation(func, rep_type)
        if domain_size:
            instance.n_vars = int(np.log2(domain_size))
        return instance

    @classmethod
    def from_scipy_distribution(cls, boolean_function_cls, distribution, rep_type='distribution', **kwargs):
        """Create from scipy.stats distribution"""
        instance = boolean_function_cls(**kwargs)
        instance.add_representation(distribution, rep_type)
        instance._setup_probabilistic_interface()
        return instance

    @classmethod
    def from_polynomial(cls, boolean_function_cls, coeffs, rep_type='polynomial', **kwargs):
        """Create from polynomial coefficients"""
        instance = boolean_function_cls(**kwargs)
        instance.add_representation(coeffs, rep_type)
        return instance

    @classmethod
    def from_multilinear(cls, boolean_function_cls, coeffs, rep_type='multilinear', **kwargs):
        """Create from multilinear polynomial coefficients"""
        instance = boolean_function_cls(**kwargs)
        instance.add_representation(coeffs, rep_type)
        return instance

    @classmethod
    def from_iterable(cls, boolean_function_cls, data, rep_type='iterable_rep', **kwargs):
        """Create from streaming truth table"""
        instance = boolean_function_cls(**kwargs)
        instance.add_representation(list(data), rep_type)
        return instance

    @classmethod
    def from_symbolic(cls, boolean_function_cls, expression, rep_type='symbolic', **kwargs):
        """Create from symbolic expression string"""
        instance = boolean_function_cls(**kwargs)
        variables = kwargs.get('variables')
        #if variables is None:
        #    variables = [f'x{i}' for i in range(instance.n_vars)]
        #kwargs.get('variables', [f'x{i}' for i in range(instance.n_vars)])
        instance.add_representation((expression, variables), rep_type)
        return instance

    @classmethod
    def from_input_invariant_truth_table(cls, boolean_function_cls, true_inputs, rep_type='truth_table', **kwargs):
        """Create from set of true input vectors"""
        n_vars = len(next(iter(true_inputs))) if true_inputs else kwargs.get('n_vars', 0)
        size = 1 << n_vars
        truth_table = np.zeros(size, dtype=bool)
        
        for i in range(size):
            vec = tuple(int(b) for b in np.binary_repr(i, width=n_vars))
            truth_table[i] = vec in true_inputs
        
        instance = boolean_function_cls(**kwargs)
        instance.add_representation(truth_table, rep_type)
        return instance


    @classmethod
    def create_composite(cls, boolean_function_cls, operator, left_func, right_func, rep_type='symbolic', **kwargs):
        """Create composite function from two BooleanFunctions or numerical values"""
        
        # Handle numerical types for left function
        if isinstance(left_func, numbers.Number):
            left_sym = str(left_func)
        else:
            left_sym = left_func.get_representation('symbolic')
        
        # Handle numerical types for right function
        if isinstance(right_func, numbers.Number):
            right_sym = str(right_func)
        else:
            right_sym = right_func.get_representation('symbolic')
        
        # Create symbolic expression
        expression = f"({left_sym} {operator.__name__} {right_sym})"
        
        # Create and return composite instance
        instance = boolean_function_cls(**kwargs)
        instance.add_representation(expression, rep_type)
        return instance