import uncertainties.unumpy as unp
from uncertainties import ufloat, umath


class ErrorModel:
    def __init__(self, model_type='exact'):
        self.model_type = model_type
    
    def propagate_binary_op(self, left_error, right_error, operation):
        """Automatic error propagation for binary operations"""
        if self.model_type == 'linear':
            # Use uncertainties library for linear propagation
            left_ufloat = ufloat(left_error.value, left_error.std_dev)
            right_ufloat = ufloat(right_error.value, right_error.std_dev)
            result = operation(left_ufloat, right_ufloat)
            return ErrorBound(result.nominal_value, result.std_dev)
        elif self.model_type == 'pac':
            return self._pac_error_propagation(left_error, right_error, operation)

class PACErrorModel(ErrorModel):
    def __init__(self, confidence=0.95, epsilon=0.1):
        self.delta = 1 - confidence  # Confidence parameter
        self.epsilon = epsilon       # Accuracy parameter
    
    def combine_pac_bounds(self, error1, error2, operation):
        """Combine PAC learning error bounds"""
        # Union bound for probability
        combined_delta = min(1.0, error1.delta + error2.delta)
        
        # Error combination depends on operation
        if operation == 'addition':
            combined_epsilon = error1.epsilon + error2.epsilon
        elif operation == 'multiplication':
            combined_epsilon = error1.epsilon * error2.epsilon + error1.epsilon + error2.epsilon
        
        return PACErrorBound(combined_epsilon, combined_delta)

class ExactErrorModel(ErrorModel):
    def __init__(self):
        pass
    def combine_pac_bounds(self, error1, error2, operation):
        pass
