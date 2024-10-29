import dropout.main.DropoutType as DT
import dropout.data.dictionary.DropoutDataField as DDF
from qiskit import QuantumCircuit
class DropoutData:
    """
    The `DropoutData` class acts as a container for all the data necessary to configure and apply a dropout process
    to a quantum circuit. It manages a dictionary where each key is a member of the `DropoutDataField` enum, 
    and the values represent the parameters required for dropout transformations.

    The class provides functionality to retrieve and update individual fields within this structured dictionary.

    Attributes:
    -----------
    __dropout_data : dict
        A private dictionary that stores the dropout parameters. The keys are instances of the `DropoutDataField` 
        enum, and the values correspond to the specific configuration parameters.

    Methods:
    --------
    __init__(dropout_type=DT.DropoutType.NO_DROPOUT, softness=0.95, prob=0.5, layer=QuantumCircuit(1),
             ansatz=QuantumCircuit(1), gate_power_expression_dict={}, feature_map_size=0, data_reuploading=False,
             additional_data={})
        Initializes the `DropoutData` instance with default or provided values for each dropout-related parameter.
        
    get_field(field)
        Retrieves the value associated with the specified `DropoutDataField` key.

    set_field(field, new_value)
        Updates the value associated with the specified `DropoutDataField` key in the dictionary.

    Parameters:
    -----------
    dropout_type : DropoutType (default: DropoutType.NO_DROPOUT)
        Specifies the type of dropout to be applied. Default is `NO_DROPOUT`, indicating no dropout.

    softness : float (default: 0.95)
        A float representing the softness of the dropout transformation, controlling the intensity of the dropout effect.

    prob : float (default: 0.5)
        A float representing the probability that a gate will be dropped out from the quantum circuit layer.

    layer : QuantumCircuit (default: QuantumCircuit(1))
        A specific layer of the quantum circuit where dropout will be applied. Default is a single-qubit circuit.

    ansatz : QuantumCircuit (default: QuantumCircuit(1))
        The entire quantum circuit, composed of one or more layers. This is the circuit to which dropout may be applied.

    gate_power_expression_dict : dict (default: {})
        A dictionary that defines the exponentiation rules for gate transformations during the dropout process.

    feature_map_size : int (default: 0)
        An integer representing the size of the feature map, which defines how classical data is encoded into quantum states.

    data_reuploading : bool (default: False)
        A boolean flag indicating whether data reuploading is used in the quantum circuit.

    additional_data : dict (default: {})
        A dictionary for any additional parameters or information that might be relevant for the dropout process.
    """

    def __init__(self, 
                 dropout_type=DT.DropoutType.NO_DROPOUT, 
                 softness=0.95, 
                 prob=0.5,
                 layer=QuantumCircuit(1),
                 ansatz=QuantumCircuit(1),
                 gate_power_expression_dict={},
                 feature_map_size=0, 
                 data_reuploading=False, 
                 additional_data={}):
        """
        Initializes the DropoutData object with the given or default values.

        Args:
            dropout_type (DropoutType): The type of dropout regime to apply (default: DropoutType.NO_DROPOUT).
            softness (float): Controls the softness of the dropout (default: 0.95).
            prob (float): Probability of dropout for gates in the circuit (default: 0.5).
            layer (QuantumCircuit): A specific layer of the quantum circuit (default: QuantumCircuit(1)).
            ansatz (QuantumCircuit): The full quantum circuit or Ansatz (default: QuantumCircuit(1)).
            gate_power_expression_dict (dict): Dictionary of gate exponentiation rules (default: {}).
            feature_map_size (int): Size of the feature map for data encoding (default: 0).
            data_reuploading (bool): Flag for using data reuploading (default: False).
            additional_data (dict): Any additional parameters (default: {}).
        """
        self.dropout_data = {}
        self.dropout_data[DDF.DropoutDataField.TYPE] = dropout_type
        self.dropout_data[DDF.DropoutDataField.SOFTNESS] = softness
        self.dropout_data[DDF.DropoutDataField.DROPOUT_PROB] = prob
        self.dropout_data[DDF.DropoutDataField.LAYER] = layer
        self.dropout_data[DDF.DropoutDataField.ANSATZ] = ansatz
        self.dropout_data[DDF.DropoutDataField.GATE_POWERS_DICT] = gate_power_expression_dict
        self.dropout_data[DDF.DropoutDataField.FEATURE_MAP_SIZE] = feature_map_size
        self.dropout_data[DDF.DropoutDataField.DATA_REUPLOADING] = data_reuploading
        self.dropout_data[DDF.DropoutDataField.ADDITIONAL_DATA] = additional_data

    def pretty_print(self):
        print("DropoutData object with the following parameters:")
        print("Dropout type: " + str(self.dropout_data[DDF.DropoutDataField.TYPE]))
        print("Softness: " + str(self.dropout_data[DDF.DropoutDataField.SOFTNESS]))
        print("Dropout probability: " + str(self.dropout_data[DDF.DropoutDataField.DROPOUT_PROB]))
        print("Layer: \n" + str(self.dropout_data[DDF.DropoutDataField.LAYER]))
        print("Ansatz: \n" + str(self.dropout_data[DDF.DropoutDataField.ANSATZ]))
        print("Gate powers dictionary: " + str(self.dropout_data[DDF.DropoutDataField.GATE_POWERS_DICT]))
        print("Feature map size: " + str(self.dropout_data[DDF.DropoutDataField.FEATURE_MAP_SIZE]))
        print("Data reuploading: " + str(self.dropout_data[DDF.DropoutDataField.DATA_REUPLOADING]))
        print("Additional data: " + str(self.dropout_data[DDF.DropoutDataField.ADDITIONAL_DATA]))
        

    def get_field(self, field):
        """
        Retrieve the value corresponding to the provided field from the dropout data dictionary.

        Args:
            field (DropoutDataField): The field (key) from which to retrieve the value.

        Returns:
            Any: The value associated with the specified `DropoutDataField` key.
        """
        return self.dropout_data[field]

    def set_field(self, field, new_value):
        """
        Update the value for the specified field in the dropout data dictionary.

        Args:
            field (DropoutDataField): The field (key) to be updated.
            new_value (Any): The new value to assign to the specified field.

        Returns:
            None
        """
        self.dropout_data[field] = new_value

