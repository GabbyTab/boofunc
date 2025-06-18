from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, TypeVar, Generic
import numpy as np

DataType = TypeVar('DataType')

class BooleanFunctionRepresentation(ABC, Generic[DataType]):
    """Abstract base class for all Boolean function representations"""
    
    @abstractmethod
    def evaluate(self, inputs: np.ndarray, data: DataType) -> Union[bool, np.ndarray]:
        """
        Evaluate the function on given inputs using the provided data.
        
        Args:
            inputs: Input values (binary array or boolean values)
            data: Representation-specific data (coefficients, truth table, etc.)
        
        Returns:
            Boolean result or array of results
        """
        pass

    @abstractmethod
    def dump(self, data: DataType, **kwargs) -> Dict[str, Any]:
        """
        Export the representation data in a serializable format.
        
        Args:
            data: The representation data to export
            **kwargs: Representation-specific options
        
        Returns:
            Dictionary containing the exported representation
        """
        pass

    @abstractmethod
    def convert_from(self, source_repr: 'BooleanFunctionRepresentation', 
                    source_data: Any, **kwargs) -> DataType:
        """
        Convert from another representation to this representation.
        
        Args:
            source_repr: Source representation strategy
            source_data: Data in source format
            **kwargs: Conversion options
        
        Returns:
            Data in this representation's format
        """
        pass

    @abstractmethod
    def convert_to(self, target_repr: 'BooleanFunctionRepresentation',
                  data: DataType, **kwargs) -> Any:
        """
        Convert from this representation to target representation.
        
        Args:
            target_repr: Target representation strategy
            data: Data in this representation's format
            **kwargs: Conversion options
        
        Returns:
            Data in target representation's format
        """
        pass

    @abstractmethod
    def create_empty(self, n_vars: int, **kwargs) -> DataType:
        """Create empty representation data structure for n variables."""
        pass

    @abstractmethod
    def is_complete(self, data: DataType) -> bool:
        """Check if the representation contains complete information."""
        pass

    @abstractmethod
    def get_storage_requirements(self, n_vars: int) -> Dict[str, int]:
        """Return memory/storage requirements for n variables."""
        pass

    def __str__(self) -> str:
        """String representation for user display."""
        return f"{self.__class__.__name__}()"

    def __repr__(self) -> str:
        """Developer-friendly string representation."""
        return f"{self.__class__.__name__}()"


class PartialRepresentation(Generic[DataType]):
    """Wrapper for handling partial representation data."""
    
    def __init__(self, strategy: BooleanFunctionRepresentation[DataType], 
                 data: DataType, known_mask: Optional[np.ndarray] = None):
        self.strategy = strategy
        self.data = data
        self.known_mask = known_mask  # Boolean mask indicating known values
        self._confidence_cache = {}
    
    def evaluate_with_confidence(self, inputs: np.ndarray) -> tuple[Any, float]:
        """Evaluate with confidence measure for partial data."""
        if self.strategy.is_complete(self.data):
            return self.strategy.evaluate(inputs, self.data), 1.0
        
        # Implement uncertainty propagation for partial data
        return self._estimate_with_uncertainty(inputs)
    
    def _estimate_with_uncertainty(self, inputs: np.ndarray) -> tuple[Any, float]:
        """Estimate result and confidence for incomplete data."""
        # Implementation depends on specific representation
        pass
