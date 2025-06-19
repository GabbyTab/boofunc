# representations/registry.py
from typing import Callable, Dict, Type, Any
from .base import BooleanFunctionRepresentation

STRATEGY_REGISTRY: Dict[str, Type[BooleanFunctionRepresentation]] = {}

def get_strategy(rep_key: str) -> BooleanFunctionRepresentation:
    """
    Retrieve and instantiate the strategy class for the given representation key.
    
    Args:
        rep_key: Representation key (e.g., 'truth_table')
    
    Returns:
        Instance of the strategy class
        
    Raises:
        KeyError: If no strategy is registered for the key
    """
    if rep_key not in STRATEGY_REGISTRY:
        raise KeyError(f"No strategy registered for '{rep_key}'")
    strategy_cls = STRATEGY_REGISTRY[rep_key]
    return strategy_cls()

def register_strategy(key: str):
    """Decorator to register representation classes"""
    def decorator(cls: Type[BooleanFunctionRepresentation]):
        STRATEGY_REGISTRY[key] = cls
        return cls
    return decorator
    
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
