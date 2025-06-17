# There should be some sort of factory class that the user has acces to that can create custum BooleanFunctionRepresentations, Where the defualt 

class LegacyBooleanFunctionRepresentation:
    """Existing boolean function implementation"""
    def legacy_evaluate(self, x):
        pass

class LegacyAdapter(LegacyBooleanFunctionRepresentation):
    """Adapter to make legacy functions compatible"""
    
    def __init__(self, legacy_function):
        self.legacy_function = legacy_function
    
    def evaluate(self, inputs):
        return self.legacy_function.legacy_evaluate(inputs)

