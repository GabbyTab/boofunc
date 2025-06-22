import numpy as np
from typing import Any, Dict, Optional, Union, List, Tuple
from .base import BooleanFunctionRepresentation, DataType
from ..spaces import Space

@register_strategy('fourier_expansion')
class FourierExpansionRepresentation(BooleanFunctionRepresentation[np.ndarray]):
    """Fourier expansion representation of Boolean functions"""
    
    def __init__(self):
        super().__init__('fourier_expansion')
    
    def evaluate(self, inputs: np.ndarray, space: Space, data: DataType, **kwargs) -> Union[bool, np.ndarray]:
        """
        Evaluate the Fourier expansion at given inputs.
        
        Args:
            inputs: Binary input vectors (shape: (m, n) or (n,))
            data: Fourier coefficients (1D array of length 2**n)
            
        Returns:
            Fourier expansion values (real numbers)
        """
        n_vars = int(np.log2(len(data)))
        
        # Convert to ±1 domain
        
        
        if inputs.ndim == 1:
            return self._evaluate_single(inputs, data, n_vars)
        return self._evaluate_batch(inputs, data, n_vars)
    
    def _evaluate_single(self, x: np.ndarray, coeffs: np.ndarray, n_vars: int) -> float:
        """Evaluate single input vector"""
        result = 0.0
        for j in range(len(coeffs)):
            # Get binary representation of subset index
            s = np.array([(j >> i) & 1 for i in range(n_vars)])
            # Compute character function χ_s(x) = ∏_{i in s} x_i
            char_val = np.prod(x[s.astype(bool)])
            result += coeffs[j] * char_val
        return result
    
    def _evaluate_batch(self, X: np.ndarray, coeffs: np.ndarray, n_vars: int) -> np.ndarray:
        """Evaluate batch of inputs"""
        results = np.zeros(X.shape[0])
        for idx, x in enumerate(X):
            results[idx] = self._evaluate_single(x, coeffs, n_vars)
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

    def convert_from(self, source_repr: 'BooleanFunctionRepresentation', 
                    source_data: Any, **kwargs) -> np.ndarray:
        """Convert from another representation to Fourier expansion"""
        # Placeholder: Actual conversion requires Fourier transform implementation
        n_vars = kwargs.get('n_vars')
        if n_vars is None:
            raise ValueError("n_vars must be provided for conversion")
        return self._compute_fourier_coeffs(source_repr, source_data, n_vars)

    def convert_to(self, target_repr: 'BooleanFunctionRepresentation',
                  data: np.ndarray, **kwargs) -> Any:
        """Convert to another representation from Fourier expansion"""
        # Placeholder: Actual conversion requires inverse transform
        n_vars = int(np.log2(len(data)))
        return target_repr.convert_from(self, data, n_vars=n_vars, **kwargs)

    def create_empty(self, n_vars: int, **kwargs) -> np.ndarray:
        """Create zero-initialized Fourier coefficients array"""
        return np.zeros(2**n_vars, dtype=float)

    def is_complete(self, data: np.ndarray) -> bool:
        """Check if representation contains non-zero coefficients"""
        return np.any(data != 0)

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
