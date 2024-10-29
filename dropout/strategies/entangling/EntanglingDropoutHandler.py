import dropout.strategies.stencil.DropoutHandler as DH
import dropout.data.dictionary.DropoutDataField as DDF
from qiskit.circuit import Parameter, ParameterExpression

class EntanglingDropoutHandler(DH.DropoutHandler):
    """
    This class is a specialized instance of `DropoutHandler` that defines the behavior of an Entangling Dropout.

    In this regime, only ENTANGLING gates (as defined in the `CircuitMask`) are eligible for dropout. 
    When an ENTANGLING gate is dropouted, it undergoes a transformation where the gate is "powered" based on a scaling factor specified in the 
    dropout data. There are no collateral dropout effects in this scenario, meaning that no other gates in the circuit are affected 
    when an ENTANGLING gate is dropouted.

    Methods:
    --------
    f_candidate_to_dropout(idx)
        Check whether the gate at the given index is an ENTANGLING gate, returning True if it is, and False otherwise.
    f_dropout_edit(idx)
        Apply dropout to the ENTANGLING gate at the given index, transforming it into its "powered" version 
        if it is parametric, or simply exponentiating it if non-parametric.
    """
    
    def __init__(self, dropout_data):
        super().__init__(dropout_data)

    def f_candidate_to_dropout(self, idx): 
        """
        Check whether the gate at the given index is an ENTANGLING gate.

        Only ENTANGLING gates are eligible for dropout in this regime. This method checks the gate type using the `CircuitMask`.

        Parameters:
        -----------
        idx : int
            The index of the gate within the circuit.

        Returns:
        --------
        bool
            Returns True if the gate is an ENTANGLING gate, False otherwise.
        """
        return self.get_dropout_data_field(DDF.DropoutDataField.CIRC_MASK).is_an_entangling(idx)

    def f_dropout_edit(self, idx):
        """
        Apply dropout to the ENTANGLING gate at the given index.

        If the gate is parametric (i.e., it has parameters like angles), it is transformed into its "powered" version 
        based on the scaling factor from the dropout data. The transformation is handled by the `gate_power_expression_mask`. 
        For non-parametric gates, they are simply exponentiated.

        Parameters:
        -----------
        idx : int
            The index of the gate to apply dropout to.

        Returns:
        --------
        Operation
            The modified gate after dropout, which is a powered version of the original ENTANGLING gate.
        """
        # operation = ansatz[idx].operation
        operation = self.get_dropout_data_field(DDF.DropoutDataField.ANSATZ)[idx].operation
        softness = self.get_dropout_data_field(DDF.DropoutDataField.SOFTNESS)
        is_parametric = any(isinstance(param, (Parameter, ParameterExpression)) for param in operation.params)
        
        if is_parametric:
            return self.get_dropout_data_field(DDF.DropoutDataField.GATE_POWERS_MASK).apply_power_expression(operation, softness)

        # return operation.power(self.softness)
        return operation.power(softness)

