import numpy as np
from typing import Dict, Any
from .polynomial import PolynomialRepresentation
from .registry import register_strategy

@register_strategy('multilinear_polynomial')
class MultilinearPolynomialRepresentation(PolynomialRepresentation):
    """
    Polynomial representation enforcing multilinearity.
    Ensures that the function can be extended as a real-valued multilinear polynomial.
    """

    def __init__(self, data: np.ndarray):
        # Validate that coefficients form a multilinear polynomial
        super().__init__()
        if not self._is_multilinear(data):
            raise ValueError("Data does not represent a multilinear polynomial")
        self.coefficients = data

    @staticmethod
    def _is_multilinear(coeffs: np.ndarray) -> bool:
        """
        Check multilinearity: since ANF coefficients are 0/1, 
        multilinearity holds by construction. In real extensions,
        ensure no squared terms exist (coeffs corresponding to repeated variables).
        """
        # In Boolean ANF, exponents >1 do not occur; return True for Boolean input
        # For real-valued polynomials, additional checks would be required.
        return coeffs.dtype in (np.int_, np.bool_) and np.all(np.logical_or(coeffs == 0, coeffs == 1))

    def dump(self, data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Export multilinear coefficients same as ANF."""
        n_vars = int(np.log2(data.size))
        return {"n_vars": n_vars, "multilinear_coeffs": data.astype(int).tolist()}

    def evaluate(self, inputs: np.ndarray, data: np.ndarray) -> np.ndarray:
        """
        Evaluate on real or Boolean inputs using multilinear extension:
        f(x) = sum_{S⊆[n]} a_S ∏_{i∈S} x_i
        """
        def eval_point(x):
            result = 0.0
            for idx, coeff in enumerate(data):
                if coeff:
                    # compute monomial product over real inputs
                    bits = np.array(list(map(int, np.binary_repr(idx, x.size))), dtype=bool)
                    result += np.prod(np.where(bits, x, 1.0))
            return result
        if inputs.ndim == 1:
            return eval_point(inputs)
        return np.array([eval_point(pt) for pt in inputs])

    def get_storage_requirements(self, n_vars: int) -> Dict[str, Any]:
        """Same storage as polynomial ANF but tagged as multilinear."""
        base = super().get_storage_requirements(n_vars)
        base["representation"] = "multilinear_polynomial"
        return base
