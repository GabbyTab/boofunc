from abc import ABC, abstractmethod

class BooleanFunctionRepresentation(ABC):
    """Abstract base class for all Boolean function representations"""
    
    @abstractmethod
    def evaluate(self, inputs):
        """Evaluate the function on given inputs"""
        pass

    @abstractmethod
    def dump(self, inputs):
        """returns the representation"""
        pass

    # convert from
    # convert to 
    # str
    # repr
    
    # ... other abstract methods

