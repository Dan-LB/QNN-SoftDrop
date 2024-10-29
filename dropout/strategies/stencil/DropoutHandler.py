from dropout.data.gate.CircuitMask import CircuitMask
import dropout.data.dictionary.DropoutDataField as DDF
import dropout.data.gate.GatePowerExpressionMask as GPEM

class DropoutHandler:
    """
    A base class for creating and managing dropout strategies for quantum circuits.

    This class provides the foundational framework for implementing different types of dropout mechanisms
    in quantum circuits. Each subclass of `DropoutHandler` implements a specific dropout strategy 
    by customizing abstract methods, while general utility methods for managing gates, applying power transformations, 
    and handling circuit masks are shared across all strategies.

    Dropout strategies remove gates probabilistically, based on the probability values set during initialization, 
    to reduce complexity or induce regularization in the quantum model.

    Once initialized, a `DropoutHandler` instance must be populated with gate-specific logic 
    through subclass implementations. The dropout strategy is then applied to a quantum circuit via a `Dropouter` object.

    Attributes:
    -----------
    computed_powers : dict
        A cache to store previously computed gate powers, avoiding redundant recalculations.
    circ_mask : CircuitMask
        The circuit mask representing the structure and features of the quantum circuit at a specific layer.
        This mask is initialized during the creation of the `DropoutHandler` object and can be updated later.
    gate_power_expression_mask : GatePowerExpressionMask
        The mask that encapsulates gate-specific power transformations based on the provided gate power expression dictionary.
    additional_data : dict
        Any extra information that might be required by specific dropout strategies (e.g., feature maps, reupload flags).

    Methods:
    --------
    __init__(dropout_data)
        Initializes the DropoutHandler with the specified dropout data and prepares the circuit mask and gate power expression mask.
    get_dropout_data_field(dropout_data_field)
        Retrieve the specified field from the dropout data.
    update_dropout_data_field(dropout_data_field, new_value)
        Update a specified field in the dropout data, recalculating necessary masks as needed.
    check_already_computed_powers(idx)
        Check if the power of the gate at a given index has already been computed and stored, applying transformations if necessary.
    is_controlled_gate(gate)
        Determine if the given gate is a controlled gate with control qubits.
    f_candidate_to_dropout(idx)
        Abstract method: Determine if the gate at the specified index is a candidate for dropout, to be implemented by subclasses.
    f_find_collateral_dropout(idx)
        Abstract method: Identify other gates affected by the dropout of a gate, to be implemented by subclasses.
    f_dropout_edit(idx)
        Abstract method: Apply the dropout transformation to the gate at the given index, to be implemented by subclasses.
    f_collateral_edit(idx)
        Abstract method: Apply the collateral dropout transformation to the gate at the given index, to be implemented by subclasses.
    """

    def __init__(self, dropout_data):
        """
        Initializes the DropoutHandler with the specified dropout data.

        The initializer creates a CircuitMask using data from the DropoutData object and encapsulates
        the gate power expression dictionary within a GatePowerExpressionMask.

        Parameters:
        -----------
        dropout_data : DropoutData
            An object containing various parameters and settings related to the dropout strategy.
        """
        self.computed_powers = {}
        self.dropout_data = dropout_data
        
        # Initializes the CircuitMask using relevant fields from the DropoutData object
        circ_mask = CircuitMask(
            self.get_dropout_data_field(DDF.DropoutDataField.LAYER),
            self.get_dropout_data_field(DDF.DropoutDataField.FEATURE_MAP_SIZE),
            self.get_dropout_data_field(DDF.DropoutDataField.DATA_REUPLOADING)
        )
        self.dropout_data.set_field(DDF.DropoutDataField.CIRC_MASK, circ_mask)

        # Encapsulates the gate power expression dictionary in a GatePowerExpressionMask
        gate_power_expression_mask = GPEM.GatePowerExpressionMask(
            self.get_dropout_data_field(DDF.DropoutDataField.GATE_POWERS_DICT)
        )
        self.dropout_data.set_field(DDF.DropoutDataField.GATE_POWERS_MASK, gate_power_expression_mask)

    def get_dropout_data_field(self, dropout_data_field):
        """
        Retrieve the specified field from the dropout data.

        Parameters:
        -----------
        dropout_data_field : DropoutDataField
            The field of the dropout data to retrieve.

        Returns:
        --------
        The value of the specified field from the dropout data.
        """
        return self.dropout_data.get_field(dropout_data_field)

    def update_dropout_data_field(self, dropout_data_field, new_value):
        """
        Update a specified field in the dropout data.

        This method allows updating certain fields in the dropout data, such as the circuit mask.
        After the update, the method recalculates the CircuitMask and the GatePowerExpressionMask as necessary.

        Parameters:
        -----------
        dropout_data_field : DropoutDataField
            The field of the dropout data to update.
        new_value : Any
            The new value to set for the specified field.
        
        Raises:
        -------
        RuntimeError
            If attempting to update certain fields after instantiation.
        """
        match dropout_data_field: 
            case DDF.DropoutDataField.LAYER | DDF.DropoutDataField.FEATURE_MAP_SIZE | DDF.DropoutDataField.DATA_REUPLOADING: 
                circ_mask = CircuitMask(
                    self.dropout_data.get_field(DDF.DropoutDataField.LAYER),
                    self.dropout_data.get_field(DDF.DropoutDataField.FEATURE_MAP_SIZE),
                    self.dropout_data.get_field(DDF.DropoutDataField.DATA_REUPLOADING)
                )
                self.dropout_data.set_field(DDF.DropoutDataField.CIRC_MASK, circ_mask)
            
            case DDF.DropoutDataField.GATE_POWERS_DICT:
                gate_power_expression_mask = GPEM.GatePowerExpressionMask(
                    self.dropout_data.get_field(DDF.DropoutDataField.GATE_POWERS_DICT)
                )
                self.dropout_data.set_field(DDF.DropoutDataField.GATE_POWERS_MASK, gate_power_expression_mask)
            
            case DDF.DropoutDataField.TYPE | DDF.DropoutDataField.ANSATZ:
                raise RuntimeError(f"You must not update {dropout_data_field} after instantiating a DropoutHandler!")

        self.dropout_data.set_field(dropout_data_field, new_value)

    # Abstract methods to be implemented in each specific dropout strategy

    def check_already_computed_powers(self, idx):
        """
        Check if the power of the gate at a given index has already been computed and stored.

        This method checks if the gate's power expression at the given index has already been computed and cached.
        If the power has not been computed, it applies the power transformation and stores the result in the cache 
        for future access, avoiding redundant computations.

        Parameters:
        -----------
        idx : int
            The index of the gate in the circuit for which the power computation is checked.

        Returns:
        --------
        Operation
            The edited gate with the computed power expression applied.
        """
        op = self.get_dropout_data_field(DDF.DropoutDataField.ANSATZ).data[idx].operation
        if op.name in self.computed_powers:
            edited_gate = self.computed_powers[op.name]
        else:
            # Apply the gate power expression and cache the result
            edited_gate = self.get_dropout_data_field(DDF.DropoutDataField.GATE_POWERS_MASK).apply_power_expression(
                op, self.get_dropout_data_field(DDF.DropoutDataField.SOFTNESS)
            )
            self.computed_powers[op.name] = edited_gate

        return edited_gate

    @staticmethod
    def is_controlled_gate(gate):
        """
        Determine if the given gate is a controlled gate.

        Controlled gates typically have control qubits (e.g., CNOT gates), which are handled differently 
        in dropout operations due to their multi-qubit structure.

        Parameters:
        -----------
        gate : Gate
            The quantum gate to be checked.

        Returns:
        --------
        bool
            Returns True if the gate has control qubits, otherwise False.
        """
        return hasattr(gate, 'num_ctrl_qubits') and gate.num_ctrl_qubits > 0

    # ----------------------------------------------------------------------------------------- #

    # Abstract methods to be implemented in each specific dropout strategy

    def f_candidate_to_dropout(self, idx):
        """
        Abstract method: Determine if the gate at the specified index is a candidate for dropout.

        This method should be implemented by subclasses to define the specific logic that identifies 
        whether a gate should be dropouted based on the dropout strategy.

        Parameters:
        -----------
        idx : int
            The index of the gate in the circuit.

        Returns:
        --------
        bool
            Returns True if the gate is eligible for dropout, otherwise False.
        """
        return True

    def f_find_collateral_dropout(self, idx):
        """
        Abstract method: Identify other gates affected by the dropout of a gate.

        When a gate at a particular index is dropped, this method identifies other gates (collaterals) that 
        might be indirectly affected by this dropout. For example, removing a rotation gate might 
        impact entangling gates in a quantum circuit, requiring additional dropouts.

        Parameters:
        -----------
        idx : int
            The index of the gate that was dropped.

        Returns:
        --------
        set
            A set of indices representing gates that are collateral dropouts because of the main gate's dropout.
        """
        return set()

    def f_dropout_edit(self, idx):
        """
        Abstract method: Apply the dropout transformation to the gate at the given index.

        This method is responsible for transforming the gate at the specified index according to the 
        dropout strategy, replacing it with its "dropouted" version.

        Parameters:
        -----------
        idx : int
            The index of the gate to be transformed.

        Returns:
        --------
        Operation
            The transformed (dropouted) gate.
        """
        return self.get_dropout_data_field(DDF.DropoutDataField.ANSATZ).data[idx].operation

    def f_collateral_edit(self, idx):
        """
        Abstract method: Apply the collateral dropout transformation to the gate at the given index.

        This method transforms the gate at the specified index into its collateral "dropouted" version.
        Collateral dropouts occur when other gates in the circuit are indirectly impacted by the dropout 
        of a particular gate.

        Parameters:
        -----------
        idx : int
            The index of the gate to be transformed.

        Returns:
        --------
        Operation
            The transformed (collateral dropouted) gate.
        """
        return self.get_dropout_data_field(DDF.DropoutDataField.ANSATZ).data[idx].operation
