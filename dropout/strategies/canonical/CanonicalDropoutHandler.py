import dropout.strategies.stencil.DropoutHandler as DH
import dropout.data.dictionary.DropoutDataField as DDF
from qiskit.circuit import Parameter, ParameterExpression

class CanonicalDropoutHandler(DH.DropoutHandler):
    """
    This class is a specialized instance of `DropoutDictionary` that defines the behavior of a Canonical Dropout.

    In this regime, only BASIC gates (as defined in the `CircuitMask`) are eligible for dropout. 
    When a BASIC gate is dropouted, it undergoes a transformation where the gate is "powered" based on a scaling factor `self.c`.

    Canonical dropout introduces collateral behavior:
    * Assume G is a BASIC gate to be dropouted. Let i, q represent G's index and the qubit it acts upon, respectively.
    * The collateral effects of removing G involve two key rules:
      - Remove the first ENTANGLING gate (after G) that has q as its control qubit.
      - Remove the last ENTANGLING gate (before G) that has q as its target qubit.

    This behavior ensures the dropout cascades through the circuit, modifying gates based on their relationships with the dropouted gate.

    Attributes:
    -----------
    closer_right_entangling: dict
        Maps BASIC gate indices to the first ENTANGLING gate that has the corresponding qubit 
        as its control qubit and appears after the BASIC gate.
    closer_left_entangling: dict
        Maps BASIC gate indices to the last ENTANGLING gate that has the corresponding qubit 
        as its target qubit and appears before the BASIC gate.
    layer_length: int
        The total number of gates in the current layer of the quantum circuit.

    Methods:
    --------
    __init__(dropout_data)
        Initializes the CanonicalDropoutHandler and populates data structures for managing dropout.
    
    populate_canonical_dropout_ds()
        Populates the dictionaries mapping BASIC gates to their corresponding nearest ENTANGLING gates.

    compute_right_closer_controlled_gate(wire)
        Finds the index of the nearest controlled (ENTANGLING) gate after a given BASIC gate.

    compute_right_closer_control(wired_view_ansatz_control, curr, end, res)
        Recursively computes the closest right-controlled gates for BASIC gates in a wire.

    f_candidate_to_dropout(idx)
        Checks if the gate at the given index is a BASIC gate, making it a candidate for dropout.

    f_find_collateral_dropout(idx)
        Identifies collateral gates to be dropouted when a BASIC gate at `idx` is dropouted.

    f_dropout_edit(idx)
        Applies the dropout transformation to a BASIC gate.

    f_collateral_edit(idx)
        Applies the same dropout transformation to collateral gates as the primary BASIC gate.
    """
    def __init__(self, dropout_data):
        """
        Initializes the CanonicalDropoutHandler.

        Args:
            dropout_data (DropoutData): Data related to the dropout mechanism.
        """
        super().__init__(dropout_data)
        self.populate_canonical_dropout_ds()

    def populate_canonical_dropout_ds(self):
        """
        Populate the dropout dictionary by computing collateral dropout data structures.

        This method:
        1. Identifies gates in the layer and categorizes them as either BASIC or ENTANGLING.
        2. Constructs two views of the layer: one for gates affecting qubits as control qubits and one for gates affecting qubits as targets.
        3. For each BASIC gate, it finds the closest right and left ENTANGLING gates and stores this information in dictionaries for later use.
        """
        # Initialize dictionaries to store closest entangling gate indices
        self.closer_right_entangling = {} 
        self.closer_left_entangling = {} 

        # Store the length of the layer for later use
        layer = self.get_dropout_data_field(DDF.DropoutDataField.LAYER)
        self.layer_length = len(layer.data)
        # Create views to track gates acting on each qubit
        wired_view_layer_control = [[] for _ in range(layer.num_qubits)] # Gates affecting qubits as control
        wired_view_layer_target = [[] for _ in range(layer.num_qubits)] # Gates affecting qubits as targets
        
        # Populate wired views with gate indices
        for gate_idx, (op, qubits, cbits) in enumerate(layer.data):
            if self.get_dropout_data_field(DDF.DropoutDataField.CIRC_MASK).is_a_basic(gate_idx):
                qubit_idx = layer.find_bit(qubits[0])
                wired_view_layer_control[qubit_idx.index].append(gate_idx)
                wired_view_layer_target[qubit_idx.index].append(gate_idx)
                continue

            if self.get_dropout_data_field(DDF.DropoutDataField.CIRC_MASK).is_an_entangling(gate_idx):
                last_qubit_checked = len(qubits)
                first_qubit_checked = 0

                # Determine if the gate is controlled or not and update checked indices accordingly
                if DH.DropoutHandler.is_controlled_gate(op): 
                    last_qubit_checked = op.num_ctrl_qubits
                    first_qubit_checked = op.num_ctrl_qubits 

                affected_qubits_controls = [layer.find_bit(qubits[i]) for i in range(0, last_qubit_checked)] 
                affected_qubits_targets = [layer.find_bit(qubits[i]) for i in range(first_qubit_checked, len(qubits))]
                
                # Populate wired views with controlled and target qubits
                for qubit in affected_qubits_controls:
                    wired_view_layer_control[qubit.index].append(gate_idx)
                
                for qubit in affected_qubits_targets:
                    wired_view_layer_target[qubit.index].append(gate_idx)
                
                continue

        # Compute closest right entangling gates for each BASIC gate
        right_closer_control_gate_idx = [self.compute_right_closer_controlled_gate(wire) for wire in wired_view_layer_control]
        for i in range(len(wired_view_layer_control)):
            for j in range(len(wired_view_layer_control[i])):
                x = wired_view_layer_control[i][j]
                k = right_closer_control_gate_idx[i][j]
                if self.get_dropout_data_field(DDF.DropoutDataField.CIRC_MASK).is_a_basic(x) and k != - 1:
                    first_controlled = wired_view_layer_control[i][k]
                    self.closer_right_entangling[x] = first_controlled

        # Reverse the target view to find closest left entangling gates
        wired_view_layer_target = [list(reversed(wire)) for wire in wired_view_layer_target]
        left_closer_target_gate_idx = [self.compute_right_closer_controlled_gate(wire) for wire in wired_view_layer_target]
        for i in range(len(wired_view_layer_target)):
            for j in range(len(wired_view_layer_target[i])):
                x = wired_view_layer_target[i][j]
                k = left_closer_target_gate_idx[i][j]
                if self.get_dropout_data_field(DDF.DropoutDataField.CIRC_MASK).is_a_basic(x) and k != -1: 
                    first_target = wired_view_layer_target[i][k]
                    self.closer_left_entangling[x] = first_target

    def compute_right_closer_controlled_gate(self, wire):
        """
        Find the index of the next controlled (ENTANGLING) gate in the wire after the current BASIC gate.
        
        Args:
            wire: The list of gate indices acting on a specific qubit.

        Returns:
            int: Index of the next ENTANGLING gate or -1 if none exists.
        """
        return self.compute_right_closer_control(wire, 0, len(wire), [-1 for _ in wire])[0]

    def compute_right_closer_control(self, wired_view_ansatz_control, curr, end, res):
        """
        Recursive function to compute the next ENTANGLING gate (rightward) for each BASIC gate in the qubit's wire.

        Args:
            wired_view_ansatz_control (list): List of gates (indices) acting on the control qubit in the ansatz.
            curr (int): The current index being processed in the wire.
            end (int): The total length of the wire.
            res (int): A list that stores the closest ENTANGLING gate for each BASIC gate. Initially filled with -1.

        Returns:
            res: Updated list where each BASIC gate index points to the nearest ENTANGLING gate to its right.
            next_idx: Index of the closest ENTANGLING gate after the current one.
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
            # curr = end - 1

            # if the last one is an entangling
            if not self.get_dropout_data_field(DDF.DropoutDataField.CIRC_MASK).is_a_basic(wired_view_ansatz_control[curr]): return (res, -1)

            # otherwise, it is a basic
            if self.get_dropout_data_field(DDF.DropoutDataField.CIRC_MASK).is_a_basic(wired_view_ansatz_control[0]) :
                res[curr] = res[0]
            else: 
                res[curr] = 0

            return (res, res[curr])
        
    def f_candidate_to_dropout(self, idx): 
        """
        Check if the gate at the given index is a BASIC gate, making it a candidate for dropout.

        Args:
            idx (int): Index of the gate to check.

        Returns:
            bool: True if the gate is BASIC, False otherwise.
        """
        return self.get_dropout_data_field(DDF.DropoutDataField.CIRC_MASK).is_a_basic(idx)
    
    def f_find_collateral_dropout(self, idx):
        """
        Identify collateral gates to be dropouted when a BASIC gate at `idx` is dropouted.

        Args:
            idx (int): Index of the gate to check.

        Returns:
            set: Indices of gates that are collateral dropouts due to the dropout of the gate at `idx`.
        """
        relative_index = self.get_dropout_data_field(DDF.DropoutDataField.CIRC_MASK).get_gate_index_in_mask(idx) # Is mask a good place for this method? 
        res = set()
        if relative_index in self.closer_right_entangling.keys():
            first_right_cnot = self.closer_right_entangling[relative_index]
            if first_right_cnot > relative_index: # The case where gate idx has the closest right xor at k positions on the right
                offset = first_right_cnot - relative_index
                res.add(idx + offset)
            else: # The case where the gate idx has the closest right xor in the next ansatz rep 
                distance_to_end = self.layer_length - relative_index
                offset = distance_to_end + first_right_cnot
                if idx + offset < len(self.get_dropout_data_field(DDF.DropoutDataField.ANSATZ)):
                    res.add(idx + offset)
                    

        if relative_index in self.closer_left_entangling.keys(): 
            first_left_cnot = self.closer_left_entangling[relative_index]
            if first_left_cnot < relative_index: # The case where gate idx has the closest left xor at k positions on the left
                offset = relative_index - first_left_cnot
                res.add(idx - offset)
            else: # The case where the gate idx has the closest left xor in the prev ansatz rep 
                distance_to_end = self.layer_length - first_left_cnot
                offset = distance_to_end + relative_index
                if idx - offset >= 0:
                    res.add(idx - offset)

        return res


    def f_dropout_edit(self, idx):
        """
        Apply the dropout transformation to a BASIC gate.

        Args:
            idx (int): Index of the gate to check.

        Returns:
            operation: The modified operation after dropout, powered by the scaling factor `self.c`.
        """
        operation = self.get_dropout_data_field(DDF.DropoutDataField.ANSATZ)[idx].operation
        softness = self.get_dropout_data_field(DDF.DropoutDataField.SOFTNESS)
        is_parametric = any(isinstance(param, (Parameter, ParameterExpression)) for param in operation.params)
        
        if is_parametric:
            return self.get_dropout_data_field(DDF.DropoutDataField.GATE_POWERS_MASK).apply_power_expression(operation, softness)

        # return operation.power(self.softness)
        return operation.power(softness)
    
    def f_collateral_edit(self, idx):
        """
        Apply the same dropout transformation to collateral gates as the primary BASIC gate.

        Args:
            idx (int): Index of the gate to check.

        Returns:
            operation: The modified operation after dropout.
        """
        
        return self.f_dropout_edit(idx) # They behave the same
