import numpy as np
from typing import Any, Dict, Union
from .truth_table import TruthTableRepresentation
from .registry import register_strategy
from .base import BooleanFunctionRepresentation

@register_strategy('polynomial')
class PolynomialRepresentation(BooleanFunctionRepresentation[np.ndarray]):
    """Polynomial (ANF) representation using a flat coefficient array."""

    def evaluate(self, inputs: np.ndarray, data: np.ndarray) -> Union[bool, np.ndarray]:
        """
        Evaluate the polynomial on given inputs.
        
        Args:
            inputs: 1D array of length n or 2D array of shape (m, n)
            data: 1D coefficient array of length 2**n
        
        Returns:
            Boolean result or array of results
        """
        def eval_one(x):
            # Select coefficients for monomials where all variables in subset are 1
            mask = (data.reshape(-1, 1) * np.prod(
                x[np.where(((np.arange(data.size)[:, None] >> 
                              np.arange(x.size)) & 1)], axis=0), axis=1
            )).astype(int) % 2
            return bool(mask.sum() % 2)

        if inputs.ndim == 1:
            return eval_one(inputs)
        else:
            return np.array([eval_one(row) for row in inputs])

    def dump(self, data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Export the polynomial coefficients.
        
        Returns:
            Dict with 'n_vars' and list of coefficient values (0 or 1)
        """
        n_vars = int(np.log2(data.size))
        return {"n_vars": n_vars, "coefficients": data.astype(int).tolist()}

    def convert_from(self,
                     source_repr: BooleanFunctionRepresentation,
                     source_data: Any,
                     **kwargs) -> np.ndarray:
        """
        Convert from a truth table to ANF via Möbius inversion.
        """
        if isinstance(source_repr, TruthTableRepresentation):
            tt = source_data.astype(int)
            coeffs = tt.copy()
            n = int(np.log2(coeffs.size))
            # Möbius (subset) transform
            for i in range(n):
                step = 1 << i
                for j in range(coeffs.size):
                    if j & step:
                        coeffs[j] ^= coeffs[j ^ step]
            return coeffs
        raise NotImplementedError(f"Conversion from {type(source_repr)} not supported")

    def convert_to(self,
                   target_repr: BooleanFunctionRepresentation,
                   data: np.ndarray,
                   **kwargs) -> Any:
        """
        Convert ANF coefficients to another representation.
        """
        if isinstance(target_repr, TruthTableRepresentation):
            # Evaluate polynomial on all inputs to build truth table
            n = int(np.log2(data.size))
            tt = np.zeros(data.size, dtype=bool)
            for idx in range(tt.size):
                inp = np.array(list(map(int, np.binary_repr(idx, n))))
                tt[idx] = self.evaluate(inp, data)
            return tt
        return target_repr.convert_from(self, data, **kwargs)

    def create_empty(self, n_vars: int, **kwargs) -> np.ndarray:
        """
        Create an empty (zero) coefficient array for n variables.
        """
        return np.zeros(1 << n_vars, dtype=int)

    def is_complete(self, data: np.ndarray) -> bool:
        """
        Check if all coefficients are known (no NaNs).
        """
        return not np.any(np.isnan(data))

    def get_storage_requirements(self, n_vars: int) -> Dict[str, int]:
        """
        Return storage details for the coefficient array.
        """
        entries = 1 << n_vars
        return {
            "entries": entries,
            "memory_bytes": entries,            # 1 byte per int
            "time_complexity": f"O(2^{n_vars}·n_vars)",
            "space_complexity": f"O(2^{n_vars})"
        }
