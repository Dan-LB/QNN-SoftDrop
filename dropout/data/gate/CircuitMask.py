from qiskit import QuantumCircuit
from dropout.data.gate.GateType import GateType

class CircuitMask:
    """
    The `CircuitMask` class provides a mechanism to describe the type of quantum gates in a specific layer of a quantum circuit
    and determine how these gates should behave in the context of the dropout process. Each gate is categorized as either
    `BASIC`, `ENTANGLING`, or `STATIC`, and this categorization is stored in a mask that maps gate types to specific positions 
    in the layer.

    The mask is particularly useful when the layer is repeated to form a Quantum Neural Network (QNN) and can help define which
    gates are essential (e.g., STATIC gates) and which can be dropped out or altered (e.g., BASIC and ENTANGLING gates).

    Attributes:
    -----------
    layer : QuantumCircuit
        A single layer of the quantum circuit. This layer will be repeated multiple times to form the full QNN.
    
    feature_map_size : int
        The size of the feature map (number of gates at the start of the layer used for data encoding). This part of the layer 
        is usually STATIC and will not change during the dropout process.
    
    data_reuploading : bool
        A flag indicating whether data reuploading is used. If `True`, the feature map will be repeated in each layer of the QNN.

    mask : list
        A list that represents the gate types for each gate in the layer. The mask uses the following gate types:
        - `BASIC`: Single-qubit gates that can be modified or dropped.
        - `ENTANGLING`: Multi-qubit gates that perform interactions between qubits.
        - `STATIC`: Gates that remain fixed, such as barriers or feature map gates.

    Methods:
    --------
    __init__(layer, feature_map_size=0, data_reuploading=False)
        Initializes the `CircuitMask` object and builds the gate type mask according to the layer's structure and input parameters.
    
    get_gate_index_in_mask(idx)
        Computes the local mask index for a global gate index in a QNN. Accounts for data reuploading or non-reuploading scenarios.
    
    get_gate_type(idx)
        Retrieves the type of gate (BASIC, ENTANGLING, STATIC) at the specified index.

    is_static(idx)
        Checks if the gate at the given index is STATIC.

    is_a_basic(idx)
        Checks if the gate at the given index is BASIC.

    is_an_entangling(idx)
        Checks if the gate at the given index is ENTANGLING.

    overwrite_mask(new_mask, feature_map_size, data_reuploading)
        Overwrites the current mask with a new mask and updates the `feature_map_size` and `data_reuploading` attributes.

    update_mask_elements_function(f, g)
        Updates mask elements based on a function `f` and a condition `g`.

    update_mask_portion_value(val, begin=-1, end=float('inf'))
        Updates a portion of the mask to a specific value (`val`), within the range from `begin` to `end`.
    
    update_mask_portion_function(f, begin=-1, end=float('inf'))
        Updates a portion of the mask using a function `f`, within the range from `begin` to `end`.
    """

    def __init__(self, layer, feature_map_size=0, data_reuploading=False):
        """
        Initializes the CircuitMask object.

        The default mask is built with the following rules:
        - The portion of the mask from index 0 to (feature_map_size - 1) is set to STATIC to prevent dropout in the feature map.
        - The remaining part of the mask (from feature_map_size to len(layer) - 1) is filled based on gate types:
          - Gates that are barriers are marked as STATIC.
          - Gates acting on more than one qubit are marked as ENTANGLING.
          - All other gates are marked as BASIC.

        Args:
            layer (QuantumCircuit): The layer of the quantum circuit that will be repeated to form the full QNN.
            feature_map_size (int, optional): The number of gates at the beginning of the layer that are part of the feature map.
                                              Defaults to 0.
            data_reuploading (bool, optional): Whether data reuploading is used, meaning the feature map is repeated with each layer.
                                               Defaults to False.
        """
        self.layer = layer
        self.feature_map_size = feature_map_size
        self.data_reuploading = data_reuploading

        # Initialize mask
        self.mask = [GateType.STATIC] * feature_map_size  # Feature map gates are STATIC
        for idx, (op, qubits, _) in enumerate(layer.data[feature_map_size:]):
            gate_type = GateType.BASIC
            if op.name.lower() == 'barrier':
                gate_type = GateType.STATIC
            elif len(qubits) > 1:
                gate_type = GateType.ENTANGLING
            self.mask.append(gate_type)

    def get_gate_index_in_mask(self, idx):
        """
        Computes the local mask index for a global gate index in a QNN.

        The index is adjusted based on whether data reuploading is used:
        - If `data_reuploading` is True, the global index is modded by the length of the mask.
        - If `data_reuploading` is False, the first layer includes the feature map, but subsequent layers do not.

        Args:
            idx (int): The global index of the gate.
        
        Returns:
            int: The local index in the mask corresponding to the global index.
        """
        if self.data_reuploading:
            idx = idx % len(self.mask)
        else:
            circ_len = len(self.mask) - self.feature_map_size
            idx = ((idx - self.feature_map_size) % circ_len) + self.feature_map_size
        return idx

    def get_gate_type(self, idx):
        """
        Retrieves the type of gate (BASIC, ENTANGLING, STATIC) at the given index.

        Args:
            idx (int): The global index of the gate.
        
        Returns:
            GateType: The type of gate at the specified index.
        """
        idx = self.get_gate_index_in_mask(idx)
        return self.mask[idx]

    def is_static(self, idx):
        """
        Checks if the gate at the given index is STATIC.

        Args:
            idx (int): The global index of the gate.

        Returns:
            bool: True if the gate is STATIC, False otherwise.
        """
        return self.get_gate_type(idx) == GateType.STATIC

    def is_a_basic(self, idx):
        """
        Checks if the gate at the given index is BASIC.

        Args:
            idx (int): The global index of the gate.

        Returns:
            bool: True if the gate is BASIC, False otherwise.
        """
        return self.get_gate_type(idx) == GateType.BASIC

    def is_an_entangling(self, idx):
        """
        Checks if the gate at the given index is ENTANGLING.

        Args:
            idx (int): The global index of the gate.

        Returns:
            bool: True if the gate is ENTANGLING, False otherwise.
        """
        return self.get_gate_type(idx) == GateType.ENTANGLING

    def overwrite_mask(self, new_mask, feature_map_size, data_reuploading):
        """
        Overwrites the existing mask with a new one and updates the `feature_map_size` and `data_reuploading` parameters.

        Args:
            new_mask (list): The new mask to replace the current mask.
            feature_map_size (int): The new feature map size.
            data_reuploading (bool): The new flag for data reuploading.
        """
        self.mask = new_mask
        self.feature_map_size = feature_map_size
        self.data_reuploading = data_reuploading

    def update_mask_elements_function(self, f, g):
        """
        Updates mask elements based on a function `f` and a condition `g`.

        For each element in the mask, the function `f` is called to compute the new value if the condition `g` evaluates to True.

        Args:
            f (function): A function that computes the new value of a mask element. Takes the element and index as parameters.
            g (function): A function that evaluates whether to update the element. Takes the element and index as parameters.
        """
        for i, elem in enumerate(self.mask):
            if g(elem, i):
                self.mask[i] = f(elem, i)

    def update_mask_portion_value(self, val, begin=-1, end=float('inf')):
        """
        Updates a portion of the mask by setting a specific value (`val`) for all elements within the range `[begin, end)`.

        Args:
            val (GateType): The new value to assign to the portion of the mask.
            begin (int, optional): The starting index of the portion to update. Defaults to -1 (start from the beginning).
            end (int, optional): The ending index (exclusive) for the portion to update. Defaults to infinity.
        """
        begin = max(0, begin)
        end = min(len(self.mask), end)
        for i in range(begin, end):
            self.mask[i] = val

    def update_mask_portion_function(self, f, begin=-1, end=float('inf')):
        """
        Updates a portion of the mask by applying a function `f` to each element within the range `[begin, end)`.

        Args:
            f (function): A function to compute the new value for each mask element.
            begin (int, optional): The starting index of the portion to update. Defaults to -1 (start from the beginning).
            end (int, optional): The ending index (exclusive) for the portion to update. Defaults to infinity.
        """
        begin = max(0, begin)
        end = min(len(self.mask), end)
        for i in range(begin, end):
            self.mask[i] = f(self.mask[i])
