# representations/registry.py

from typing import Callable, Dict, Type
from .base import BooleanFunctionRepresentation

STRATEGY_REGISTRY: Dict[str, Type[BooleanFunctionRepresentation]] = {}

def register_strategy(key: str, cls: Type[BooleanFunctionRepresentation]):
    """Register a full-featured strategy class."""
    STRATEGY_REGISTRY[key] = cls

def register_partial_strategy(
    key: str,
    *,
    evaluate: Callable,
    dump: Callable = None,
    convert_from: Callable = None,
    convert_to: Callable = None,
    create_empty: Callable = None,
    is_complete: Callable = None,
    get_storage_requirements: Callable = None
):
    """
    Register a strategy by supplying only the key methods.
    Missing methods raise NotImplementedError by default.
    """
    # Dynamically build subclass
    methods = {
        'evaluate': evaluate,
        'dump': dump or (lambda self, data, **kw: {'data': data}),
        'convert_from': convert_from or (lambda self, src, data, **kw: NotImplementedError()),
        'convert_to': convert_to or (lambda self, tgt, data, **kw: NotImplementedError()),
        'create_empty': create_empty or (lambda self, n, **kw: NotImplementedError()),
        'is_complete': is_complete or (lambda self, data: True),
        'get_storage_requirements': get_storage_requirements or (lambda self, n: {})
    }
    # Create new class
    NewStrategy = type(f"{key.title()}Strategy", (BooleanFunctionRepresentation,), methods)
    # Register it
    register_strategy(key, NewStrategy)
