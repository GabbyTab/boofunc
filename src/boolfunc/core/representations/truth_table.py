import numpy as np
from typing import Any, Dict, Optional, Union, TypeVar, Generic
from .registry import register_strategy
from .base import BooleanFunctionRepresentation
from ..spaces import Space


@register_strategy('truth_table')
class TruthTableRepresentation(BooleanFunctionRepresentation[np.ndarray]):
    """Truth table representation using NumPy arrays."""

    def evaluate(self, inputs: np.ndarray, data: np.ndarray, space: Space, n_vars: int) -> np.ndarray:
        # Input validation
        if not isinstance(inputs, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(inputs)}")
          
        output = data[inputs]

        if np.isscalar(output) or output.shape == ():
            return output  # unwrap np.bool_ or np.int_ to bool/int
    
        # Direct indexing handles both scalars and arrays
        return output
   
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
            'type': "truth_table",
            'n': int(np.log2(data.size)),
            'size': data.size,
            'values': data.astype(bool).tolist()
        }

    def convert_from(self, source_repr: BooleanFunctionRepresentation, source_data: Any, space: Space, n_vars: int, **kwargs) -> np.ndarray:
        """Convert from any representation by evaluating all possible inputs."""
        size = 1 << n_vars  # 2^n
        truth_table = np.zeros(size, dtype=bool)

        # Generate all possible input indices
        for idx in range(size):
            value = source_repr.evaluate(idx, source_data, space, n_vars)
            #should handle differentley depending on the space
            value = (1 - value)/2
            truth_table[idx] = value

        return truth_table

    def convert_to(self, target_repr: BooleanFunctionRepresentation, souce_data: Any, space: Space, n_vars: int, **kwargs) -> np.ndarray:
        """Convert truth table to another representation."""
        return target_repr.convert_from(self, souce_data, space, n_vars, **kwargs)

    def create_empty(self, n_vars: int, **kwargs) -> np.ndarray:
        """Create an empty (all-False) truth table for n variables."""
        size = 1 << n_vars
        return np.zeros(size, dtype=bool)

    def get_storage_requirements(self, n_vars: int) -> Dict[str, int]:
        """Storage grows exponentially: 1 byte per entry (packed to bits)."""
        entries = 1 << n_vars
        return {
            'entries': entries,
            'bytes': entries // 8,            # packed bits
            'space_complexity': 'O(2^n)'
        }


    def time_complexity_rank(self, n_vars: int) -> Dict[str, int]:
        """Return time_complexity for computing/evalutating n variables."""
        pass



    def is_complete(self, data: np.ndarray) -> bool:
        """Check if the representation contains complete information."""
        pass
