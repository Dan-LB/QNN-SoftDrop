import dropout.strategies.stencil.DropoutHandler as DH
import dropout.data.dictionary.DropoutDataField as DDF
from qiskit.circuit import Parameter, ParameterExpression

class RotationDropoutHandler(DH.DropoutHandler):
    """
    This class is a specific instance of `DropoutHandler` designed to manage the behavior of a Rotation Dropout.

    In this regime, only BASIC gates (as classified by the CircuitMask) are eligible for dropout. 
    When a BASIC gate is dropouted, it undergoes a "powering" transformation based on a scaling factor specified in the 
    dropout data. There are no collateral effects associated with the dropout, meaning no other gates are affected 
    when a BASIC gate is dropouted.

    Methods:
    --------
    f_candidate_to_dropout(idx)
        Check whether a gate at the given index is a BASIC gate, returning True if it is, and False otherwise.
    f_dropout_edit(idx)
        Apply dropout to the BASIC gate at the given index, transforming it into its "powered" version 
        if it is parametric, or simply exponentiating it if non-parametric.
    """

    # ----------------------------------------------------------------------------------------- #

    # Methods overriding DropoutHandler functionality
    def __init__(self, dropout_data):
        super().__init__(dropout_data)

    def f_candidate_to_dropout(self, idx): 
        """
        Check whether a gate at the given index is a BASIC gate.

        Only BASIC gates are candidates for dropout in the RotationDropout regime.

        Parameters:
        -----------
        idx : int
            The index of the gate within the circuit.

        Returns:
        --------
        bool
            Returns True if the gate is a BASIC gate, False otherwise.
        """

        return self.get_dropout_data_field(DDF.DropoutDataField.CIRC_MASK).is_a_basic(idx)

    def f_dropout_edit(self, idx):
        """
        Apply dropout to the BASIC gate at the given index.

        If the gate is parametric (i.e., it has parameters like angles), it is transformed into its "powered" version 
        using the power coefficient from the dropout data. This is done by applying the power expression mask. 
        Non-parametric gates are simply exponentiated.

        Parameters:
        -----------
        idx : int
            The index of the gate to apply dropout to.

        Returns:
        --------
        Operation
            The modified gate after the dropout (powered version of the original gate).
        """
        operation = self.get_dropout_data_field(DDF.DropoutDataField.ANSATZ)[idx].operation
        softness = self.get_dropout_data_field(DDF.DropoutDataField.SOFTNESS)
        is_parametric = any(isinstance(param, (Parameter, ParameterExpression)) for param in operation.params)
        
        if is_parametric:
            return self.get_dropout_data_field(DDF.DropoutDataField.GATE_POWERS_MASK).apply_power_expression(operation, softness)

        return operation.power(softness)


