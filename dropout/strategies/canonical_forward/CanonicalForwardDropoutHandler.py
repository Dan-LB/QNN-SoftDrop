import dropout.strategies.stencil.DropoutHandler as DH
import dropout.data.dictionary.DropoutDataField as DDF
from qiskit.circuit import Parameter, ParameterExpression

class CanonicalForwardDropoutHandler(DH.DropoutHandler):
    """
    A specialized implementation of `DropoutHandler` that applies Canonical Forward Dropout to quantum circuits.

    This dropout strategy targets BASIC gates in the circuit and removes ENTANGLING gates as collateral, based on their position and qubit interaction. 
    The canonical forward dropout defines:
    - BASIC gates as candidates for dropout.
    - ENTANGLING gates on the same qubit as collateral if the qubit is a control qubit in an ENTANGLING gate after the BASIC gate.

    Methods:
    --------
    f_candidate_to_dropout(idx)
        Determines if a gate at the given index is a candidate for dropout.
    f_find_collateral_dropout(idx)
        Finds the collateral gates that will also be affected by dropping the gate at the given index.
    f_dropout_edit(idx)
        Applies dropout to the gate at the given index.
    f_collateral_edit(idx)
        Applies collateral dropout to the gate at the given index.
    """
    
    def __init__(self, dropout_data):
        super().__init__(dropout_data)
        self.populate_canonical_forward_dropout_ds()

    def populate_canonical_forward_dropout_ds(self):
        """
        Populate the dropout dictionary with the mapping of BASIC gates and their corresponding right-side closer ENTANGLING gates acting on the same qubit as a control.
        """
        self.closer_right_entangling = {}
        layer = self.get_dropout_data_field(DDF.DropoutDataField.LAYER)

        self.layer_length = len(layer)
        wired_view_layer_control = [[] for _ in range(layer.num_qubits)]

        for gate_idx, (op, qubits, cbits) in enumerate(layer.data):
            if self.get_dropout_data_field(DDF.DropoutDataField.CIRC_MASK).is_a_basic(gate_idx):
                qubit_idx = layer.find_bit(qubits[0])
                wired_view_layer_control[qubit_idx.index].append(gate_idx)
                continue

            if self.get_dropout_data_field(DDF.DropoutDataField.CIRC_MASK).is_an_entangling(gate_idx):
                last_qubit_checked = len(qubits)
                if DH.DropoutHandler.is_controlled_gate(op):
                    last_qubit_checked = op.num_ctrl_qubits
                affected_qubits_controls = [layer.find_bit(qubits[i]) for i in range(0, last_qubit_checked)]
                for qubit in affected_qubits_controls:
                    wired_view_layer_control[qubit.index].append(gate_idx)
                continue

        right_closer_control_gate_idx = [self.compute_right_closer_controlled_gate(wire) for wire in wired_view_layer_control]

        for i in range(len(wired_view_layer_control)):
            for j in range(len(wired_view_layer_control[i])):
                x = wired_view_layer_control[i][j]
                k = right_closer_control_gate_idx[i][j]
                if self.get_dropout_data_field(DDF.DropoutDataField.CIRC_MASK).is_a_basic(x) and k != -1:
                    first_controlled = wired_view_layer_control[i][k]
                    self.closer_right_entangling[x] = first_controlled

    def compute_right_closer_controlled_gate(self, wire):
        """
        Computes the index of the closest right controlled gate for each gate in the wire.

        Parameters:
        -----------
        wire : list
            List of gate indices for a specific qubit.

        Returns:
        --------
        list
            List of indices for the closest right controlled gates.
        """
        return self.compute_right_closer_control(wire, 0, len(wire), [-1 for _ in wire])[0]

    def compute_right_closer_control(self, wired_view_ansatz_control, curr, end, res):
        """
        Recursively computes the closest right controlled gate for each gate in the wire.

        Parameters:
        -----------
        wired_view_ansatz_control : list
            List of gate indices for a specific qubit.
        curr : int
            Current index being processed.
        end : int
            End index for the list.
        res : list
            Result list to store the closest right controlled gate indices.

        Returns:
        --------
        tuple
            Updated result list and the index of the closest controlled gate.
        """
        if curr < end - 1:
            current_is_a_basic = self.get_dropout_data_field(DDF.DropoutDataField.CIRC_MASK).is_a_basic(wired_view_ansatz_control[curr])
            next_is_a_basic = self.get_dropout_data_field(DDF.DropoutDataField.CIRC_MASK).is_a_basic(wired_view_ansatz_control[curr + 1])
            current_is_an_entangling = not current_is_a_basic
            next_is_an_entangling = not next_is_a_basic

            if current_is_a_basic and next_is_a_basic:
                res, next_idx = self.compute_right_closer_control(wired_view_ansatz_control, curr + 1, end, res)
                res[curr + 1] = next_idx
                res[curr] = next_idx
                return (res, next_idx)
            elif current_is_a_basic and next_is_an_entangling:
                res[curr] = curr + 1
                res, _ = self.compute_right_closer_control(wired_view_ansatz_control, curr + 1, end, res)
                return (res, curr + 1)
            elif current_is_an_entangling and next_is_a_basic:
                return self.compute_right_closer_control(wired_view_ansatz_control, curr + 1, end, res)
            elif current_is_an_entangling and next_is_an_entangling:
                return self.compute_right_closer_control(wired_view_ansatz_control, curr + 1, end, res)
        else:
            if not self.get_dropout_data_field(DDF.DropoutDataField.CIRC_MASK).is_a_basic(wired_view_ansatz_control[curr]):
                return (res, -1)
            if self.get_dropout_data_field(DDF.DropoutDataField.CIRC_MASK).is_a_basic(wired_view_ansatz_control[0]):
                res[curr] = res[0]
            else:
                res[curr] = 0
            return (res, res[curr])

    def f_candidate_to_dropout(self, idx):
        """
        Determines if a gate at the given index is a candidate for dropout.

        Parameters:
        -----------
        idx : int
            The index of the gate in the circuit.

        Returns:
        --------
        bool
            Returns True if the gate is a BASIC gate, False otherwise.
        """
        return self.get_dropout_data_field(DDF.DropoutDataField.CIRC_MASK).is_a_basic(idx)

    def f_find_collateral_dropout(self, idx):
        """
        Finds the collateral gates that will also be affected by dropping the gate at the given index.

        Parameters:
        -----------
        idx : int
            The index of the gate being checked.

        Returns:
        --------
        set
            A set of indices representing gates that will also be dropped as collateral effects.
        """
        relative_index = self.get_dropout_data_field(DDF.DropoutDataField.CIRC_MASK).get_gate_index_in_mask(idx)
        res = set()
        first_right_cnot = self.closer_right_entangling.get(relative_index, -1)
        
        if relative_index in self.closer_right_entangling.keys():
            if first_right_cnot > relative_index:
                offset = first_right_cnot - relative_index
                res.add(idx + offset)
            else:
                distance_to_end = self.layer_length - relative_index
                offset = distance_to_end + first_right_cnot
                if idx + offset < len(self.get_dropout_data_field(DDF.DropoutDataField.ANSATZ)):
                    res.add(idx + offset)

        return res

    def f_dropout_edit(self, idx):
        """
        Applies dropout to the gate at the given index.

        Parameters:
        -----------
        idx : int
            The index of the gate to be edited.

        Returns:
        --------
        Operation
            The gate with the applied dropout power.
        """
        operation = self.get_dropout_data_field(DDF.DropoutDataField.ANSATZ).data[idx].operation
        softness = self.get_dropout_data_field(DDF.DropoutDataField.SOFTNESS)
        is_parametric = any(isinstance(param, (Parameter, ParameterExpression)) for param in operation.params)
        
        if is_parametric:
            return self.get_dropout_data_field(DDF.DropoutDataField.GATE_POWERS_MASK).apply_power_expression(operation, softness)

        # return operation.power(self.softness)
        return operation.power(softness)

    def f_collateral_edit(self, idx):
        """
        Applies collateral dropout to the gate at the given index.

        Parameters:
        -----------
        idx : int
            The index of the gate to be edited.

        Returns:
        --------
        Operation
            The edited operation with the collateral dropout applied.
        """
        return self.f_dropout_edit(idx)


