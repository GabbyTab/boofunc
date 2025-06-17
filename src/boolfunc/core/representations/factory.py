class RepresentationFactory:
    """Factory for creating custom representations"""
    _representations = {}
    
    @classmethod
    def register(cls, name: str, representation_class):
        """Register a new representation type"""
        cls._representations[name] = representation_class
    
    @classmethod
    def create(cls, name: str, *args, **kwargs):
        """Create a representation instance"""
        if name not in cls._representations:
            raise InvalidRepresentationError(f"Unknown representation: {name}")
        return cls._representations[name](*args, **kwargs)
