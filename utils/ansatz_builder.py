from enum import Enum

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from utils.constants import ModelType
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

class RotationType(Enum):
    RX = 1, 
    RZ = 2, 
    RY = 3

    def rot_to_function(rot):
        match rot: 
            case RotationType.RX: 
                return lambda circ, param, q : circ.rx(param, q)
            case RotationType.RZ: 
                return lambda circ, param, q : circ.rz(param, q)
            case RotationType.RY: 
                return lambda circ, param, q : circ.ry(param, q)

def create_rotation_layer(circ, num_qubits, rot_type, offset, ansatz_layer, barrier):
    for q in range(num_qubits):
        param = Parameter(f"theta{offset + ansatz_layer*num_qubits + q}")  # Create parameter named theta1, theta2, etc. 
        rotation = RotationType.rot_to_function(rot_type)
        #print(param)
        rotation(circ, param, q)
       
    for q in range(num_qubits - 1):
        circ.cx(q, q + 1)

    if barrier: circ.barrier()

    return circ

def build_regressor_ansatz(num_qubits=5, reps=10, more_barriers = False, feature_map = None):
    ansatz = QuantumCircuit(num_qubits, name="vf")

    # create a set of rx rotations, followed by a set of cx
    # followed by a set of rz rotations, followed by a set of cx
    # and finally, a set of rx rotations, followed by a set of cx

    num_rotations_per_rep = num_qubits * 3

    single_layer = QuantumCircuit(num_qubits, name="layer")
    single_layer.compose(feature_map, inplace=True)
    single_layer = create_rotation_layer(single_layer, num_qubits, RotationType.RX, 0, 0, more_barriers)
    single_layer = create_rotation_layer(single_layer, num_qubits, RotationType.RZ, 0, 1, more_barriers)
    single_layer = create_rotation_layer(single_layer, num_qubits, RotationType.RX, 0, 2, more_barriers)

    for r in range(reps):

        ansatz.compose(feature_map, inplace=True)
        ansatz = create_rotation_layer(ansatz, num_qubits, RotationType.RX, num_rotations_per_rep*r, 0, more_barriers)
        ansatz = create_rotation_layer(ansatz, num_qubits, RotationType.RZ, num_rotations_per_rep*r, 1, more_barriers)
        ansatz = create_rotation_layer(ansatz, num_qubits, RotationType.RX, num_rotations_per_rep*r, 2, more_barriers)


    return ansatz, single_layer

# print(build_regressor_ansatz(5, 2, True))

def build_classification_ansatz(num_qubits=5, reps=10, more_barriers = False, feature_map = None):

    #classical_bits = 2

    quantum_reg = QuantumRegister(num_qubits, name='q')
    #classical_reg = ClassicalRegister(classical_bits, name='c')

    # Create the quantum circuit with the quantum and classical registers
    ansatz = QuantumCircuit(quantum_reg, name="vf")



    n_params = 9*reps


    single_layer = QuantumCircuit(num_qubits, name="layer")
    single_layer.compose(feature_map, inplace=True)
    for i in range(num_qubits): #funziona solo per num_qubits = 5
        param = Parameter(f"theta{i}") 
        single_layer.rx(param, i)
    for i in range(num_qubits - 1):
        param = Parameter(f"theta{i+5}")
        single_layer.crx(param, i, i+1)
    single_layer.barrier()

    for r in range(reps):
        ansatz.compose(feature_map, inplace=True)
        for i in range(num_qubits): #funziona solo per num_qubits = 5
            param = Parameter(f"theta{r*9+i}") 
            ansatz.rx(param, i)
        for i in range(num_qubits - 1):
            param = Parameter(f"theta{r*9+i+5}")
            ansatz.crx(param, i, i+1)
        ansatz.barrier()
    print("lines 90 from asnatz_builder.py")
    #measure the last qubit

    #ansatz.measure_all()
    #ansatz.measure([0, 1], [0, 1])

    print(ansatz)
    return ansatz, single_layer


def build_iris_ansatz(num_qubits=4, reps=2, more_barriers = False, feature_map = None):
    # best: ZFeature4 x RealAmplitude4

    # ma meglio usare "PauliFeatureMap" da articolo iccs quantum iris
    ansatz = QuantumCircuit(num_qubits, name="vf")

    num_rotations_per_rep = num_qubits * 3

    single_layer = QuantumCircuit(num_qubits, name="layer")
    single_layer.compose(feature_map, inplace=True)
    single_layer = create_rotation_layer(single_layer, num_qubits, RotationType.RY, 0, 0, more_barriers)
    single_layer = create_rotation_layer(single_layer, num_qubits, RotationType.RY, 0, 1, more_barriers)
    single_layer = create_rotation_layer(single_layer, num_qubits, RotationType.RY, 0, 2, more_barriers)

    for r in range(reps):

        ansatz.compose(feature_map, inplace=True)
        ansatz = create_rotation_layer(ansatz, num_qubits, RotationType.RY, num_rotations_per_rep*r, 0, more_barriers)
        ansatz = create_rotation_layer(ansatz, num_qubits, RotationType.RY, num_rotations_per_rep*r, 1, more_barriers)
        ansatz = create_rotation_layer(ansatz, num_qubits, RotationType.RY, num_rotations_per_rep*r, 2, more_barriers)


    return ansatz, single_layer
    
    
