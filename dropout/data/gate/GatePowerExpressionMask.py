class GatePowerExpressionMask:
    """
    The `GatePowerExpressionMask` class manages a mask (dictionary) that allows custom power expressions
    to be applied to specific gates during exponentiation operations. 

    In quantum circuits, when a gate R(a) --- where 'a' is the angle of rotation --- is exponentiated 
    by a factor 'c' to form R^c(a), this class provides a mechanism to override the default power 
    behavior with custom expressions defined by the user. The custom expressions are particularly 
    useful for gates that require non-standard behavior when raising their parameters to a power.

    Attributes:
    -----------
    gate_power_expression_mask : dict
        A dictionary that maps gate names (str) to lambda expressions. Each lambda expression defines
        a custom way to update a gate's parameter(s) when raising it to a power. The lambda function
        takes two arguments: the current parameter of the gate and the power factor, and returns the 
        updated parameter.
    
    Methods:
    --------
    __init__(mask={})
        Initializes the `GatePowerExpressionMask` with an optional custom mask. This mask maps gate 
        names to lambda expressions that override the default exponentiation behavior.

    has_power_expression(name)
        Checks if a custom power expression exists for a given gate name.

    apply_power_expression(op, factor)
        Applies the custom power expression to a gate's parameters if defined. Otherwise, the default 
        power operation for the gate is applied.
    """

    def __init__(self, mask={}):
        """
        Initializes the `GatePowerExpressionMask` object.

        If a custom mask is provided, it will be used to define the custom power expressions for gates.
        The mask is a dictionary where the keys are gate names, and the values are lambda expressions 
        that define how to update the gate's parameter(s) when computing powers of the gate.

        Args:
            mask (dict, optional): A dictionary mapping gate names (str) to lambda expressions (function).
                                   The lambda expression should take two arguments: the current parameter 
                                   of the gate and the power factor, and return the updated parameter. 
                                   Defaults to an empty dictionary if no mask is provided.
        """
        self.gate_power_expression_mask = mask

    def has_power_expression(self, name):
        """
        Checks if a custom power expression is defined for a specific gate.

        This method looks for the gate's name in the `gate_power_expression_mask` dictionary. If the gate
        is found in the dictionary, it indicates that a custom power expression exists for that gate.

        Args:
            name (str): The name of the gate to check.

        Returns:
            bool: Returns `True` if a custom power expression exists for the gate, `False` otherwise.
        """
        return name in self.gate_power_expression_mask

    def apply_power_expression(self, op, factor):
        """
        Applies a custom power expression to a gate's parameters if one is defined in the mask.

        If the gate's name is found in the `gate_power_expression_mask`, the corresponding lambda 
        expression will be applied to modify the gate's parameter(s) based on the specified power factor.
        If no custom expression is defined for the gate, the method falls back to the default power 
        operation using the gate's `.power()` method.

        Args:
            op (Gate): The gate object whose parameter(s) need to be updated based on the power factor.
            factor (float): The exponent to apply to the gate's parameter(s).

        Returns:
            Gate: The updated gate object with modified parameters after applying the custom power 
            expression or default power method.
        """
        if self.has_power_expression(op.name):
            # Apply custom power expression from the mask
            op.params[0] = self.gate_power_expression_mask[op.name](op.params[0], factor)
        else:
            # Use the default power operation of the gate
            return op.power(factor)
        
        return op
