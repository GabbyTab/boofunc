from abc import ABC
from typing import Any, Dict, List, Tuple, Union
import numpy as np
from .registry import register_strategy
from .base import BooleanFunctionRepresentation

@register_strategy('symbolic')
class SymbolicRepresentation(BooleanFunctionRepresentation[Tuple[str, List[str]]]):
    """
    Symbolic representation storing an expression and variable order.
    
    Data format: (expression: str, variables: List[str])
    """
    def evaluate(self, inputs: np.ndarray, data: DataType, **kwargs) -> Union[bool, np.ndarray]:

        """
        Evaluate the symbolic expression using Python eval.
        
        Args:
            inputs: 1D or 2D array of booleans.
            data: (expr, vars) where expr is a Python boolean expression
                  string and vars is the variable symbol list.
        
        Returns:
            Boolean result(s) by substituting inputs into expr.
        """
        expr, vars = data
        # Build a local namespace mapping each symbol to its input value
        def eval_point(x):
            local = {var: bool(x[i]) for i, var in enumerate(vars)}
            return bool(eval(expr, {}, local))  # safe eval with empty globals
        
        if inputs.ndim == 1:
            return eval_point(inputs)
        return np.array([eval_point(row) for row in inputs])

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

    def convert_from(
        self, source_repr: str, source_data: Any, **kwargs
    ) -> Tuple[str, List[str]]:
        """
        Convert another representation into symbolic form.
        
        Currently unsupported for non-symbolic sources.
        """
        raise NotImplementedError("Conversion to symbolic not implemented")
    
    def convert_to(
        self, target_repr: str, data: Tuple[str, List[str]], **kwargs
    ) -> Any:
        """
        Convert symbolic data to another representation.
        
        Delegates to the target’s convert_from if possible.
        """
        return target_repr.convert_from(self, data, **kwargs)

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
