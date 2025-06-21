import numpy as np
from typing import Any, Dict, Optional, Union, TypeVar, Generic
from .registry import register_strategy
from .base import BooleanFunctionRepresentation


@register_strategy('truth_table')
class TruthTableRepresentation(BooleanFunctionRepresentation[np.ndarray]):
    """Truth table representation using NumPy arrays."""

    def evaluate(self, inputs: np.ndarray, data: np.ndarray) -> Union[bool, np.ndarray]:
        """Evaluate using direct table lookup."""
        # Single input vector
        if inputs.ndim == 1:
            idx = self._compute_index(inputs)
            return bool(data[idx])
        # Batch of input vectors
        indices = [self._compute_index(row) for row in inputs]
        return data[np.array(indices)]

    def dump(self, data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Export the truth table.
        
        Returns a serializable dictionary containing:
        - 'table': list of booleans
        - 'n_vars': number of variables
        """
        return {
            'n_vars': int(np.log2(data.size)),
            'table': data.astype(bool).tolist()
        }

    def convert_from(self, source_repr: str,
                     source_data: Any, **kwargs) -> np.ndarray:
        """Convert from polynomial """

        raise NotImplementedError(f"Cannot convert from {type(source_repr)}")

    def convert_to(self, target_repr: str,
                   data: np.ndarray, **kwargs) -> Any:
        """Convert truth table to another representation."""
        return target_repr.convert_from(self, data, **kwargs)

    def create_empty(self, n_vars: int, **kwargs) -> np.ndarray:
        """Create an empty (all-False) truth table for n variables."""
        size = 1 << n_vars
        return np.zeros(size, dtype=bool)

    def get_storage_requirements(self, n_vars: int) -> Dict[str, int]:
        """Storage grows exponentially: 1 byte per entry (packed to bits)."""
        entries = 1 << n_vars
        return {
            'entries': entries,
            'memory_bytes': entries // 8,            # packed bits
            'time_complexity': 1,                   # O(1) per lookup
            'space_complexity': f'O(2^{n_vars})'
        }

    def _compute_index(self, bits: np.ndarray) -> int:
        """Convert a binary vector to its integer index."""
        # Ensure boolean dtype and flatten
        bits = bits.astype(bool).flatten()
        # Interpret MSB at index 0
        return int(np.dot(bits, 1 << np.arange(bits.size)[::-1]))

    def _from_polynomial(self, coeffs: Dict[tuple, float], **kwargs) -> np.ndarray:
        """Build a truth table from polynomial coefficients."""
        n_vars = kwargs.get('n_vars', int(max(idx for mono in coeffs for idx in mono) + 1))
        table = self.create_empty(n_vars)
        for idx in range(table.size):
            inp = np.array(list(map(int, np.binary_repr(idx, n_vars))))
            # Evaluate polynomial mod 2
            val = sum(coeffs.get(mono, 0.0) * np.prod(inp[list(mono)]) 
                      for mono in coeffs) % 2
            table[idx] = bool(val)
        return table

    def is_complete(self, data: np.ndarray) -> bool:
        """Check if the representation contains complete information."""
        pass
