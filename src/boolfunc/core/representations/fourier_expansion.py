import numpy as np
from typing import Any, Dict, Optional, Union, TypeVar, Generic
from .registry import register_strategy
from .base import BooleanFunctionRepresentation
from ..spaces import Space

@register_strategy('fourier_expansion')
class FourierExpansionRepresentation(BooleanFunctionRepresentation[np.ndarray]):
    """Fourier expansion representation of Boolean functions"""
    
    def evaluate(self, inputs: np.ndarray, data: np.ndarray, space: Space, n_vars: int) -> np.ndarray:
        """
        Evaluate the Fourier expansion at given inputs.
        
        Args:
            inputs: Binary input vectors (shape: (m, n) or (n,))
            data: Fourier coefficients (1D array of length 2**n)
            
        Returns:
            Fourier expansion values (real numbers)
        """
    
        # Convert to ±1 domain

        if not isinstance(inputs, np.ndarray):
            return self._evaluate_single(inputs, data)
        return self._evaluate_batch(inputs, data)
    
    def _evaluate_single(self, x: int, coeffs: np.ndarray) -> float:
        """Evaluate f(x) given x as an integer bitstring, and coeffs of Fourier expansion over {0,1}^n"""
        result = 0.0
        for j in range(len(coeffs)):
            parity = bin(x & j).count("1") % 2
            char_val = (-1) ** parity
            result += coeffs[j] * char_val
        return result

    
    def _evaluate_batch(self, X: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """Evaluate a batch of inputs X, where each x is an integer bitstring"""
        results = np.zeros(len(X))
        for idx, x in enumerate(X):
            results[idx] = self._evaluate_single(x, coeffs)
        return results


    def dump(self, data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Export Fourier coefficients in serializable format"""
        return {
            'coefficients': data.tolist(),
            'type': 'fourier_expansion',
            'metadata': {
                'num_vars': int(np.log2(len(data))),
                'norm': float(np.linalg.norm(data))
            }
        }

    def convert_from(self, source_repr: BooleanFunctionRepresentation, source_data: Any, space: Space, n_vars: int, **kwargs) -> np.ndarray:
        """
        Compute Fourier coefficients from any evaluable Boolean representation.
        Fourier basis: {(-1)^{x·s}} for s ⊆ [n]
        """

        size = 1 << n_vars  # 2^n
        coeffs = []

        # Generate all input vectors x (as bits) and their integer indicesc
        x_bits = np.array([list(map(int, np.binary_repr(i, n_vars))) for i in range(size)])
        x_indices = np.arange(size, dtype=int)

        # Evaluate function f on all inputs
        f_vals = source_repr.evaluate(x_indices, source_data, space, n_vars)
        f_vals = np.asarray(f_vals, dtype=float)
        
        # Optionally map {0,1} to {1,-1} for Boolean-to-sign function conversion
        f_vals = 1 - 2 * f_vals  # Now f ∈ {1, -1}

        # For all subsets S ⊆ [n] (Fourier basis functions)
        for i in range(size):
            S = tuple(j for j in range(n_vars) if (i >> j) & 1)

            # Compute parity vector: (-1)^{x·S}
            signs = (-1) ** np.sum(x_bits[:, list(S)], axis=1)
            coeff = np.dot(f_vals, signs) / size

            if coeff != 0.0:
                coeffs.append(coeff)

        return np.array(coeffs)

    def convert_to(self, target_repr: BooleanFunctionRepresentation, souce_data: Any, space: Space, n_vars: int, **kwargs) -> np.ndarray:
        """Convert to another representation from Fourier expansion"""
        # Placeholder: Actual conversion requires inverse transform
        return target_repr.convert_from(self, souce_data, space, n_vars, **kwargs)

    def create_empty(self, n_vars: int, **kwargs) -> np.ndarray:
        """Create zero-initialized Fourier coefficients array"""
        return np.zeros(2**n_vars, dtype=float)

    def is_complete(self, data: np.ndarray) -> bool:
        """Check if representation contains non-zero coefficients"""
        return np.any(data != 0)

    def time_complexity_rank(self, n_vars: int) -> Dict[str, int]:
        """Return time_complexity for computing/evalutating n variables."""
        pass

    def get_storage_requirements(self, n_vars: int) -> Dict[str, int]:
        """Return memory requirements for n variables"""
        num_coeffs = 2**n_vars
        return {
            'dtype': 'float64',
            'elements': num_coeffs,
            'bytes': num_coeffs * 8,  # 8 bytes per float
            'human_readable': f"{num_coeffs * 8 / 1024:.2f} KB" if num_coeffs > 1024 
                             else f"{num_coeffs * 8} bytes"
        }

    def _compute_fourier_coeffs(self, source_repr: 'BooleanFunctionRepresentation',
                               source_data: Any, n_vars: int) -> np.ndarray:
        """Compute Fourier coefficients from source representation"""
        # This should implement the Fast Fourier Transform (FFT) for Boolean functions
        # Placeholder implementation - returns identity coefficients
        return np.random.uniform(-1, 1, size=2**n_vars)
