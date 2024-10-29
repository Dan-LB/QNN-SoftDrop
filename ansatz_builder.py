from enum import Enum

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import numpy as np

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
        rotation(circ, param, q)
       
    for q in range(num_qubits - 1):
        circ.cx(q, q + 1)

    if barrier: circ.barrier()

    return circ

from qiskit.quantum_info import Operator

def build_fake_regressor_ansatz(num_qubits=4, reps=1, more_barriers = False, feature_map = None):
    ansatz = QuantumCircuit(num_qubits, name="vf")
    # create a set of rx rotations, followed by a set of cx
    # followed by a set of rz rotations, followed by a set of cx
    # and finally, a set of rx rotations, followed by a set of cx

    qc1 = QuantumCircuit(2)
    qc1.x(0)
    qc1.h(1)
    custom = qc1.to_gate().control(2)

    # U = Operator(custom)
    # print(U.data)

    # custom2 = custom.power(0.0)

    # U1 = Operator(custom2)
    
    # print(U1.data)

    # # print(custom.to_matrix())
    # # print(custom2.to_matrix())

    # return 

    custom1 = qc1.to_gate()

    num_rotations_per_rep = num_qubits * 3

    for r in range(reps):

        # ansatz.compose(feature_map, inplace=True)
        
        # ansatz = create_rotation_layer(ansatz, num_qubits, RotationType.RX, num_rotations_per_rep*r, 0, more_barriers)
        ansatz.barrier()
        
        ansatz = create_rotation_layer(ansatz, num_qubits, RotationType.RZ, num_rotations_per_rep*r, 0, more_barriers)
        ansatz.append(custom, [0, 3, 1, 2])
        ansatz.append(custom1, [0, 2])
        ansatz.barrier()

        ansatz = create_rotation_layer(ansatz, num_qubits, RotationType.RX, num_rotations_per_rep*r, 1, more_barriers)
        ansatz.append(custom, [0, 3, 1, 2])
        ansatz.append(custom1, [0, 2])
        ansatz.barrier()

        ansatz = create_rotation_layer(ansatz, num_qubits, RotationType.RX, num_rotations_per_rep*r, 2, more_barriers)
        ansatz.append(custom, [0, 3, 1, 2])
        ansatz.append(custom1, [0, 2])
        ansatz.barrier()
        
    return ansatz

def build_regressor_ansatz(num_qubits = 5, reps = 10, more_barriers = False, feature_map = None):

    ansatz = QuantumCircuit(num_qubits, name="vf")

    # create a set of rx rotations, followed by a set of cx
    # followed by a set of rz rotations, followed by a set of cx
    # and finally, a set of rx rotations, followed by a set of cx
    
    num_rotations_per_rep = num_qubits * 3
    
    layer = QuantumCircuit(num_qubits, name='L')
    if feature_map is not None: 
        layer.compose(feature_map, inplace=True)
    layer = create_rotation_layer(layer, num_qubits, RotationType.RX, 0, 0, more_barriers)
    layer = create_rotation_layer(layer, num_qubits, RotationType.RZ, 0, 1, more_barriers)
    layer = create_rotation_layer(layer, num_qubits, RotationType.RX, 0, 2, more_barriers)

    for r in range(reps):
        if feature_map is not None:
            ansatz.compose(feature_map, inplace=True)
        
        ansatz = create_rotation_layer(ansatz, num_qubits, RotationType.RX, num_rotations_per_rep*r, 0, more_barriers)
        ansatz = create_rotation_layer(ansatz, num_qubits, RotationType.RZ, num_rotations_per_rep*r, 1, more_barriers)
        ansatz = create_rotation_layer(ansatz, num_qubits, RotationType.RX, num_rotations_per_rep*r, 2, more_barriers)

    return ansatz, layer
