from abc import ABC
from typing import Any, Dict, List, Tuple, Union
import numpy as np
from .registry import register_strategy
from .base import BooleanFunctionRepresentation
from ..spaces import Space

@register_strategy('symbolic')
class SymbolicRepresentation(BooleanFunctionRepresentation[Tuple[str, List[str]]]):
    """
    Symbolic representation storing an expression and variable order.
    
    Data format: (expression: str, variables: List[str])
    """
    def evaluate(self, inputs: np.ndarray, data: Tuple[str, List[Any]], space: Space, n_vars: int) -> np.ndarray:
        """
        Evaluate the symbolic Boolean expression composed of sub-BooleanFunctions using bit slicing.

        Inputs are integers representing bitstrings. Only the last `n_vars` bits are used.

        Each sub-function is evaluated on a slice of those bits, passed as an integer or array of integers.

        Args:
            inputs: np.ndarray of shape () or (batch,) — each integer encodes a bitstring.
            data: Tuple (expr: str, funcs: List[BooleanFunction]) — symbolic expression and sub-functions.
            space: evaluation space.
            kwargs: must include 'n_vars': total number of bits in each bitstring.

        Returns:
            Single bool or np.ndarray of bools (same shape as inputs).

        issues:
            is recomputing repeated variables
        """
        expr, funcs = data
        func_lengths = [f.get_n_vars() for f in funcs]

        # Precompute bitmasks for each subfunction
        bit_slices = []
        bit_index = 0
        for length in func_lengths:
            mask = (1 << length) - 1
            bit_slices.append((bit_index, mask))
            bit_index += length

        #swap to matrix operations in future
        def eval_point(val: int) -> bool:
            context = {}
            for j, (f, (offset, mask)) in enumerate(zip(funcs, bit_slices)):
                sub_input = (val >> (n_vars - offset - func_lengths[j])) & mask
                context[f"x{j}"] = f.evaluate(np.array(sub_input), None, space=space)
            return eval(expr, {}, context)

        if np.isscalar(inputs) or inputs.ndim == 0:
            return np.array(eval_point(int(inputs)))

        # Batch evaluation
        return np.array([eval_point(int(v)) for v in inputs])

    def dump(self, data: Tuple[str, List[str]], **kwargs) -> Dict[str, Any]:
        """
        Export the symbolic representation as a dictionary.
        
        Returns:
            {
                "expression": expr,
                "variables": vars
            }
        """
        expr, vars = data
        return {"expression": expr, "variables": vars}

    def convert_from(self, source_repr: BooleanFunctionRepresentation, source_data: Any, space: Space, n_vars: int, **kwargs) -> Tuple[str, List[Any]]:
        """
        Convert from another representation to symbolic form.

        Creates a symbolic expression that calls x0(...) etc. on the original function using wrapped variables.

        Args:
            source_repr: The original representation class.
            source_data: The original representation data.
            space: The evaluation space (e.g. Boolean cube).
            n_vars: Number of variables in the original function.
            kwargs: Must include 'functions' — the BooleanFunction constructor/class.

        Returns:
            Tuple[str, List[BooleanFunction]] — symbolic expression and subfunctions list.
        """
      
        # Wrap the original function as one symbolic call: x0
        expr = "x0"

        # Create a single BooleanFunction instance representing the whole function

        return ()

    def convert_to(self, target_repr: BooleanFunctionRepresentation, souce_data: Any, space: Space, n_vars: int, **kwargs) -> np.ndarray:
            """Convert to another representation from Fourier expansion"""
            # Placeholder: Actual conversion requires inverse transform
            return target_repr.convert_from(self, souce_data, space, n_vars, **kwargs)

    def create_empty(self, n_vars: int, **kwargs) -> Tuple[str, List[str]]:
        """
        Create an empty symbolic representation:
        - Empty expression string
        - Default variable symbols ["x0", "x1", ..., "x{n_vars-1}"]
        """
        vars = [f"x{i}" for i in range(n_vars)]
        return "", vars

    def is_complete(self, data: Tuple[str, List[str]]) -> bool:
        """
        A symbolic representation is complete if the expression is non-empty.
        """
        expr, _ = data
        return bool(expr)

    def get_storage_requirements(self, n_vars: int) -> Dict[str, int]:
        """
        Estimate minimal storage for the expression metadata.
        """
        return {
            "expression_chars": 0,    # dynamic, depends on expr length
            "variables": n_vars       # one pointer per variable symbol
        }

    def time_complexity_rank(self, n_vars: int) -> Dict[str, int]:
        """Return time_complexity for computing/evalutating n variables."""
        pass