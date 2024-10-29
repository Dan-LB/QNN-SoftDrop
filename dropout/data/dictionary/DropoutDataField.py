from enum import Enum

class DropoutDataField(Enum):
    """
    This class enumerates all the possible types of data that can be provided to the `Dropouter` class.
    Each field corresponds to a specific parameter or structure required by the dropout mechanism, 
    which is applied to quantum circuits (Ansatz). These fields help manage the configuration and 
    behavior of the dropout process.

    Enum Members:
    -------------
    TYPE : DropoutType
        Specifies the type of dropout regime being applied. This could include different strategies like 
        'canonical dropout' or others, and it's managed by the `DropoutType` class.

    SOFTNESS : float
        A float value that controls the scaling factor of the dropout effect. Softness indicates the intensity 
        with which gates are dropouted, where a lower value represents a stronger dropout effect.

    DROPOUT_PROB : float
        A float representing the probability of dropout for each eligible gate in the quantum circuit.
        This value dictates how likely a gate is to undergo dropout.

    LAYER : QuantumCircuit
        Refers to the layer of the quantum circuit. A `QuantumCircuit` object, which consists of repeated layers,
        builds up the Ansatz. Dropout can be applied to specific layers within the circuit.

    ANSATZ : QuantumCircuit
        The entire quantum circuit or the structural framework of the circuit to which dropout is applied.
        It is built from multiple layers, and each layer can be modified by the dropout mechanism.

    GATE_POWERS_DICT : dict
        A dictionary that holds the exponentiation rules for gates. Each key represents a gate index, 
        and the corresponding value specifies the power to which the gate operation should be raised 
        during the dropout process.

    GATE_POWERS_MASK : GatePowerExpressionMask
        An encapsulated version of the `GATE_POWERS_DICT`. This object masks the gates that should undergo 
        power transformations and provides an interface for applying these transformations based on the 
        exponentiation rules in the dictionary.

    FEATURE_MAP_SIZE : int
        An integer that represents the size of the feature map. The feature map defines the way classical data 
        is encoded into quantum states, and this size determines how much data can be mapped into the quantum circuit.

    DATA_REUPLOADING : bool
        A boolean flag indicating whether data reuploading is used in the Ansatz. Data reuploading refers to the 
        practice of feeding classical data into the quantum circuit multiple times during its execution.

    CIRC_MASK : CircuitMask
        A `CircuitMask` object that categorizes each gate in the Ansatz as either BASIC, ENTANGLING, or STATIC. 
        The mask is used to identify which gates are eligible for dropout or transformation based on their type.

    ADDITIONAL_DATA : dict
        A dictionary for storing any additional data relevant to the dropout process. This field can be used 
        to pass extra information or parameters that don't fit into the predefined categories but are 
        still needed by the dropout mechanism.
    """
    TYPE = "Type"
    SOFTNESS = "Softness"
    DROPOUT_PROB = "Prob"
    LAYER = "Layer"
    ANSATZ = "Ansatz"
    GATE_POWERS_DICT = "Gate_power_expression_dict"  # Dictionary with gate exponentiation rules
    GATE_POWERS_MASK = "Gate_power_expression_mask"  # Mask encapsulating gate powers
    FEATURE_MAP_SIZE = "Feature_map_size"            # Integer size of the feature map
    DATA_REUPLOADING = "Data_Reuploading"            # Boolean flag for data reuploading usage
    CIRC_MASK = "Circuit_mask"                       # CircuitMask object classifying gates
    ADDITIONAL_DATA = "Additional Data"              # Dictionary for any extra relevant data


