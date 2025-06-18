from boolfunc.core import BooleanFunction
from boolfunc.core.factory import BooleanFunctionFactory

def create(data=None, **kwargs):
    """User-friendly entry point for creating BooleanFunction instances"""
    return BooleanFunctionFactory.create(BooleanFunction, data, **kwargs)
