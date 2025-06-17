class BooleanFunctionError(Exception):
    """Base exception for all library errors"""

class InvalidRepresentationError(BooleanFunctionError):
    """Raised when requesting unsupported representation"""

