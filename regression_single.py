from utils.model_builder import build_model
import utils.constants as constants

import time
from custom_classes.custom_estimator_qnn import CustomEstimatorQNN

from torch.nn import BCEWithLogitsLoss, Sigmoid

from qiskit_machine_learning.connectors import TorchConnector

import time

import torch
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from qiskit_algorithms.utils import algorithm_globals


# from qiskit.circuit.library import RealAmplitudes

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from qiskit.primitives import Estimator
from qiskit_algorithms.gradients import ReverseEstimatorGradient

from matplotlib import pyplot as plt
# from qiskit.primitives import Sampler

# import constants
import utils.ansatz_builder as ansatz_builder
import utils.tasks_manager as tasks_manager
import utils.constants as constants
import os
import time

import tqdm

from utils.train_and_test import train, grid_evaluator

import matplotlib.pyplot as plt
import numpy as np

from dropout.main import DropoutType as DT
import dropout.data.dictionary.DropoutData as DD
# import dropout.data.dictionary.DropoutDataField as DDF

#I want to use latex in the plots
plt.rc('text', usetex=True)

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)


#task = constants.Tasks.TEST_LOG
task = constants.Tasks.TEST
task_name = task.value


X_train, y_train, X_test, y_test, X_, y_ = tasks_manager.generate_task(task,
                                                               num_samples_train = 15,
                                                               num_samples_test = 5,
                                                               show = False,
                                                               seed = seed)
#plt.show()



epochs = 300

fX_train = tasks_manager.prepare_features(X_train, task)
fX_test = tasks_manager.prepare_features(X_test, task)
fX_ = tasks_manager.prepare_features(X_, task)

# ------------------------------------------------ DROPOUT-RELATED STUFF ------------------------------------------------------|
                                                                                                                              #|
dropout_code = "C"                                                                                                            #|
                                                                                                                              #|
circ_data_dict = {'num_qubit' : 5, 'reps' : 10}                                                                               #|
                                                                                                                              #|
softness = 0.1                                                                                                                #|
p_drop = 0.5                                                                                                                  #|
                                                                                                                              #|
partial_dropout_data = DD.DropoutData(dropout_type = DT.DropoutType.CANONICAL, softness = softness, prob = p_drop)            #|
model, feature_map, single_layer = build_model(constants.ModelType.ROTATIONAL_REGRESSOR, circ_data_dict, partial_dropout_data)#|
                                                                                                                              #|
                                                                                                                              #|
#-------------------------------------------------------------------------------------------------------------------------------


optimizer = torch.optim.Adam(model.parameters(), lr=0.01) #0.01
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(fX_train.shape)
print(y_train.shape)

train_dataset = TensorDataset(torch.Tensor(fX_train), torch.Tensor(y_train))
train_dataloader = DataLoader(train_dataset,
                              shuffle=True,
                              batch_size=20,
                              num_workers=0)

test_dataset = TensorDataset(torch.Tensor(fX_test), torch.Tensor(y_test))
test_dataloader = DataLoader(test_dataset,
                             shuffle=True,
                             batch_size=5,
                             num_workers=0)

model.to(device)


# dropouter = Dropouter(ansatz, drop)
model.activate_dropout()

print(model.qnn.neural_network.dropouter)
total_loss_train, total_loss_test, total_accuracy_train, total_accuracy_test = train(model, optimizer, train_dataloader, test_dataloader, epochs, device, verbose=True)
print(total_loss_train)
print(total_loss_test)
print(total_accuracy_train)
print(total_accuracy_test)

model.activate_dropout()
#grid_evaluator(model, 20, True)

#my path is joined "results" and task_name
my_path = os.path.join("results", task_name)

if not os.path.exists(my_path):
    os.makedirs(my_path)

model_name = f"{dropout_code}{str(int(softness*100))}_{str(int(p_drop*100))}_seed{seed}"

saving_path = os.path.join(my_path, model_name)
if not os.path.exists(saving_path):
    os.makedirs(saving_path)

data_path = os.path.join(saving_path, "data")
if not os.path.exists(data_path):
    os.makedirs(data_path)

# I want to save X_train, y_train, X_test, y_test
np.save(os.path.join(data_path, "X_train.npy"), X_train)
np.save(os.path.join(data_path, "y_train.npy"), y_train)
np.save(os.path.join(data_path, "X_test.npy"), X_test)
np.save(os.path.join(data_path, "y_test.npy"), y_test)

# I want to save total_loss_train, total_loss_test, total_accuracy_train, total_accuracy_test
np.save(os.path.join(saving_path, "loss_train.npy"), total_loss_train)
np.save(os.path.join(saving_path, "loss_test.npy"), total_loss_test)
np.save(os.path.join(saving_path, "accuracy_train.npy"), total_accuracy_train)
np.save(os.path.join(saving_path, "accuracy_test.npy"), total_accuracy_test)

#create "train_data" with X_train and y_tr
model.deactivate_dropout()
pred = model(torch.Tensor(fX_)).detach().numpy()

np.save(os.path.join(saving_path, "pred.npy"), pred)

plot_title = ""
if dropout_code == "C":
    plot_title = "Canonical Dropout"
if softness != 0:
    plot_title = f"Soft {plot_title}, $\\alpha$ = {softness}, $p$ = {p_drop}"
else:
    plot_title = f"Hard {plot_title}, $p$ = {p_drop}"

plt.plot(X_train, y_train, 'o', label='Training data')
plt.plot(X_test, y_test, 'd', label='Test data')

#plot X_ and pred as a dashed line plot
plt.plot(X_, pred, '--', label='Prediction')

plt.legend()
plt.xlabel('X', fontsize=16)
plt.ylabel('$\\sin(x)$', fontsize=16)
plt.title(plot_title, fontsize=22)
plt.savefig(os.path.join(saving_path, "plot.pdf"))
#plt.show()
    #plt.savefig(f"results/{task_name}.png")
