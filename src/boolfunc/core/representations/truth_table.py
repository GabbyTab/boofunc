import numpy as np
from typing import Any, Dict, Optional, Union, TypeVar, Generic
from .registry import register_strategy
from .base import BooleanFunctionRepresentation


@register_strategy('truth_table')
class TruthTableRepresentation(BooleanFunctionRepresentation[np.ndarray]):
    """Truth table representation using NumPy arrays."""

    def evaluate(self, inputs: np.ndarray, data: np.ndarray, **kwargs) -> Union[bool, np.ndarray]:
        # Input validation
        if not isinstance(inputs, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(inputs)}")
        
        # Integer index processing
        elif np.issubdtype(inputs.dtype, np.integer):
        # Validate bounds
        if np.any((inputs < 0) | (inputs >= len(data))):
            raise ValueError(f"Index out of bounds for truth table of size {len(data)}")
        
        # Direct indexing handles both scalars and arrays
        return data[inputs]
   
    def _compute_index(self, bits: np.ndarray) -> int:
        """Optimized bit packing using NumPy"""
        return int(np.packbits(bits.astype(np.uint8), bitorder='big')[0])

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

    def time_complexity_rank(self, n_vars: int) -> Dict[str, int]:
        """Return time_complexity for computing/evalutating n variables."""
        pass

    def _compute_index(self, bits: np.ndarray) -> int:
        """Convert boolean vector to integer index using bit packing"""
        return int(np.packbits(bits.astype(np.uint8), bitorder='big')[0])


    def _from_polynomial(self, coeffs: Dict[tuple, float], **kwargs) -> np.ndarray:
        """Build a truth table from polynomial coefficients. Speedy version should use FFT depending on size"""
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
