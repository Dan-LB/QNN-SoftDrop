import dropout.main.Dropouter as D
import dropout.data.dictionary.DropoutData as DD
import dropout.data.dictionary.DropoutDataField as DDF
from torch.nn import BCEWithLogitsLoss, BCELoss, Sigmoid
from qiskit_machine_learning.connectors import TorchConnector


import torch

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

import math

import ansatz_builder as ansatz_builder

from .constants import ModelType

from custom_classes.custom_estimator_qnn import CustomEstimatorQNN
from qiskit_algorithms.gradients import ReverseEstimatorGradient

from qiskit.primitives import Estimator# <----- DI SOLITO QUESTO
#from qiskit_aer.primitives import Estimator

from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import GenericBackendV2
# from qiskit_ibm_runtime.fake_provider import FakeLondon
from qiskit import transpile


class ClassificationNet(torch.nn.Module):
    """Feedfoward neural network with 1 hidden layer"""

    def __init__(self, qnn, **kwargs):
        super().__init__()
        self.qnn = TorchConnector(qnn, **kwargs)
        self.loss_func = BCELoss()
        self.sigmoid = Sigmoid()
        self.scaling_factor = 5

    def forward(self, x):

        return self.sigmoid(self.qnn(x)*self.scaling_factor)

    def activate_dropout(self):
        self.qnn.neural_network.activate_dropout()

    def deactivate_dropout(self):
        self.qnn.neural_network.deactivate_dropout()

class RegressionNet(torch.nn.Module):
    """Feedfoward neural network with 1 hidden layer"""

    def __init__(self, qnn, **kwargs):
        super().__init__()
        self.qnn = TorchConnector(qnn, **kwargs)
        self.loss_func = torch.nn.MSELoss()
        #self.sigmoid = Sigmoid()

    def forward(self, x):
        return self.qnn(x)

    def activate_dropout(self):
        self.qnn.neural_network.activate_dropout()

    def deactivate_dropout(self):
        self.qnn.neural_network.deactivate_dropout()



def build_circuit(model_type, reps, num_qubits, l_features):
    if model_type == ModelType.CLASSIFICATOR:
        return build_classification_circuit(num_qubits = num_qubits, reps = reps)
    elif model_type == ModelType.ROTATIONAL_REGRESSOR:
        return build_regression_circuit(num_qubits = num_qubits, reps = reps)
    elif model_type == ModelType.IRIS_CLASSIFICATOR:
        return build_iris_circuit(num_qubits = 4, reps = reps)
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")

def build_regression_circuit(num_qubits=5, reps=20):

    arcsin_x = Parameter("sin^{-1}x")
    arccos_x2 = Parameter("cos^{-1}(x^2)")

    feature_map = QuantumCircuit(num_qubits, name="fm")
    for i in range(num_qubits):
        feature_map.ry(arcsin_x, i)
        feature_map.rz(arccos_x2, i)
    qc, single_layer = ansatz_builder.build_regressor_ansatz(num_qubits=num_qubits,
                                            reps=reps,
                                            more_barriers=True,
                                            feature_map=feature_map)
    
    return qc, feature_map, single_layer

def build_classification_circuit(num_qubits=5, reps=20):
    """
    Returns both the quantum circuit and the feature map used for the classification model
    """

    #raise NotImplementedError("Classification ansatz not implemented [dobbiamo ritornare anche partial_ansatz]")

    alpha1 = Parameter('alpha1') #2 sin^{-1}(x1/2)
    alpha2 = Parameter('alpha2') #2 sin^{-1}(x2/2)
    beta1 = Parameter('beta1') #2 cos^{-1}(x1^2/4)
    beta2 = Parameter('beta2') #2 cos^{-1}(x2^2/4)

    feature_map = QuantumCircuit(num_qubits, name="fm")
    for i in range(num_qubits):
        if i%2 == 0:
            feature_map.ry(alpha1, i)
            feature_map.rz(beta1, i)
        else:
            feature_map.ry(alpha2, i)
            feature_map.rz(beta2, i)

    # construct simple ansatz

    qc, single_layer = ansatz_builder.build_classification_ansatz(num_qubits=num_qubits,
                                               reps=reps,
                                               more_barriers=True,
                                               feature_map=feature_map)
    print(qc)
    print(feature_map)
    print(single_layer)
    return qc, feature_map, single_layer

def build_iris_circuit(num_qubits=4, reps=4):
    #questa cosa ha su ogni qubit H, Rz(2x[i]), Rx(pi/2), Ry(2x[i]), Rx(-pi/2)
    #e poi presa la coppia 0, 1, ha CNOT 0, 1, ZZ(2*(pi-x[0])(pi-x[1])), CNOT 0, 1 eccetera
    feature_map = QuantumCircuit(num_qubits, name="fm")
    for i in range(num_qubits):
        feature_map.h(i)
        feature_map.rz(2*Parameter(f"x{i}"), i)
        feature_map.rx(math.pi/2, i)
        feature_map.ry(2*Parameter(f"x{i}"), i)
        feature_map.rx(-math.pi/2, i)

    for i, j in range(num_qubits):
        if i > j:
            feature_map.cx(i, j)
            feature_map.rzz(2*(math.pi-Parameter(f"x{i}"))*(math.pi-Parameter(f"x{j}")), i, j)
            feature_map.cx(i, j)

    qc, single_layer = ansatz_builder.build_iris_ansatz(num_qubits=num_qubits,
                                               reps=reps,
                                               more_barriers=True,
                                               feature_map=feature_map)
    print(qc)
    print(feature_map)
    print(single_layer)

    return False

def build_dropout(dropout_data):
    dropouter = D.Dropouter(dropout_data)

    return dropouter

def build_net(l_features, qc, feature_map, dropouter, net_builder):

    estimator = Estimator()

    gradient = ReverseEstimatorGradient()

    qnn = CustomEstimatorQNN(
        estimator=estimator,
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=qc.parameters[l_features:],
        gradient=gradient,
        input_gradients=True,
        dropouter=dropouter,
    )

    return net_builder(qnn)

def build_model(model_type, circ_data, partial_dropout_data):
    
    match model_type:
        case ModelType.CLASSIFICATOR:
            l_features = 4
            net_builder = lambda qnn : ClassificationNet(qnn)
            
        case ModelType.ROTATIONAL_REGRESSOR:
            l_features = 2
            net_builder = lambda qnn : RegressionNet(qnn)
    
        case _:
            l_features = 2
            net_builder = lambda qnn : RegressionNet(qnn)


    qc, feature_map, single_layer = build_circuit(model_type, reps=circ_data['reps'], num_qubits=circ_data['num_qubit'], l_features=l_features)
    dropout_data = DD.DropoutData(dropout_type = partial_dropout_data.get_field(DDF.DropoutDataField.TYPE), 
                                    softness = partial_dropout_data.get_field(DDF.DropoutDataField.SOFTNESS),
                                    prob = partial_dropout_data.get_field(DDF.DropoutDataField.DROPOUT_PROB),
                                    layer = single_layer,
                                    ansatz = qc,
                                    gate_power_expression_dict = {},
                                    feature_map_size = len(feature_map.data),
                                    data_reuploading = True, 
                                    additional_data = partial_dropout_data.get_field(DDF.DropoutDataField.ADDITIONAL_DATA)
                                  )
    dropouter = build_dropout(dropout_data)
    net = build_net(l_features, qc, feature_map, dropouter, net_builder)

    return net, feature_map, single_layer