import numpy as np
from qiskit import QuantumCircuit
import dropout.main.DropoutType as DT
import dropout.data.dictionary.DropoutDataField as DDF

class Dropouter:
    """
    The Dropouter class manages the application of dropout mechanisms 
    to a quantum circuit. It utilizes various dropout handlers to determine 
    which gates should be dropouted and how collateral gates are affected.

    Attributes:
    -----------
    __ansatz: QuantumCircuit
        The quantum circuit (ansatz) on which the dropout operations will be applied.
    __dropout_handlers: list
        A list of dropout handlers used to manage different dropout types and their behaviors.
    __current_handler: DropoutHandler
        The currently active dropout handler for processing dropout operations.

    Methods:
    --------
    __init__(dropout_data)
        Initializes the Dropouter with the specified dropout data.
    
    update_handlers_data(handler_index, field, new_value)
        Updates the specified field in the dropout handler at the given index with a new value.

    apply()
        Applies dropout to the ansatz using all registered dropout handlers.

    apply_dropout()
        Determines which gates to drop out and applies the necessary transformations 
        to the ansatz based on the currently active dropout handler.
    """

    def __init__(self, dropout_data):
        """
        Initializes the Dropouter with dropout data.

        Args:
            dropout_data (DropoutData): Data related to dropout mechanisms, including 
            the quantum circuit (ansatz) and other parameters.
        """
        self.ansatz = dropout_data.get_field(DDF.DropoutDataField.ANSATZ)
        self.dropout_handlers = DT.DropoutType.type_to_handlers(dropout_data)

    def update_handlers_data(self, handler_index, field, new_value):
        """
        Updates the dropout data for a specified dropout handler.

        Args:
            handler_index (int): Index of the handler to update.
            field (str): The field in the handler to update.
            new_value: The new value to set for the specified field.
        """
        self.dropout_handlers[handler_index].update_dropout_data(field, new_value)

    def apply(self):
        """
        Applies dropout to the ansatz using all registered dropout handlers.

        Returns:
            QuantumCircuit: The modified quantum circuit with dropout applied.
        """
        old_ansatz = self.ansatz
        for handler in self.dropout_handlers:
            self.current_handler = handler
            self.ansatz = self.apply_dropout()
        
        ansatz = self.ansatz
        self.ansatz = old_ansatz

        return ansatz
    
    def apply_dropout(self):
        """
        Identifies which gates in the ansatz should be dropouted and applies 
        the corresponding transformations to the circuit.

        Returns:
            QuantumCircuit: A new quantum circuit with specified gates dropouted 
            and collateral effects applied.
        """
        gates_to_dropout = set()  # Gates to be dropouted
        collateral_edited_gates = set()  # Collateral gates to be edited

        # Determine gates to be dropouted and collateral gates to be affected
        for idx, (op, qubits, clbits) in enumerate(self.ansatz.data):
            # Skip gates used for the feature map
            if self.current_handler.get_dropout_data_field(DDF.DropoutDataField.CIRC_MASK).is_static(idx):
                continue

            if self.current_handler.f_candidate_to_dropout(idx):
                if np.random.rand() <= self.current_handler.get_dropout_data_field(DDF.DropoutDataField.DROPOUT_PROB):
                    gates_to_dropout.add(idx)
                    additional = self.current_handler.f_find_collateral_dropout(idx)
                    collateral_edited_gates |= additional

        gates_to_dropout = sorted(gates_to_dropout, reverse=True)
        collateral_edited_gates = sorted(collateral_edited_gates, reverse=True)

        dropouted_ansatz = QuantumCircuit(self.ansatz.num_qubits)

        # Apply dropout and update collateral gates
        for idx, (op, qubits, clbits) in enumerate(self.ansatz.data):
            if idx in collateral_edited_gates:
                edited_gate = self.current_handler.f_collateral_edit(idx)
                dropouted_ansatz.append(edited_gate, qubits, clbits)
                continue

            if idx in gates_to_dropout:
                edited_gate = self.current_handler.f_dropout_edit(idx)
                dropouted_ansatz.append(edited_gate, qubits, clbits)
                continue

            dropouted_ansatz.append(op, qubits, clbits)

        return dropouted_ansatz
