# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Estimator quantum neural network class"""

from __future__ import annotations

import logging
from copy import copy

# from dropouter import apply_canonical_dcropout, apply_canonical_forward_dropout, apply_rotation_dropout, apply_entangling_dropout, apply_independent_dropout

import numpy as np
from qiskit_algorithms.gradients import (
    EstimatorGradientResult,
)

from qiskit.primitives import EstimatorResult

from qiskit_machine_learning.exceptions import QiskitMachineLearningError

from qiskit_machine_learning.neural_networks import EstimatorQNN

logger = logging.getLogger(__name__)



class CustomEstimatorQNN(EstimatorQNN):

    def __init__(
        self,
        dropouter = None,
        **args
    ):
        self.dropouter = dropouter
        self.circuits_batch = None
        self.is_dropout_active = False
        # set primitive, provide default
        
        super().__init__(
            **args
        )

    def deactivate_dropout(self):
        # print("Deactivating dropout")
        self.is_dropout_active = False

    def activate_dropout(self):
        # print("Activating dropout")
        self.is_dropout_active = True
    


    def _forward_postprocess(self, num_samples: int, result: EstimatorResult) -> np.ndarray:
        """Post-processing during forward pass of the network."""
        return np.reshape(result.values, (-1, num_samples)).T

    def _forward(
        self, input_data: np.ndarray | None, weights: np.ndarray | None
    ) -> np.ndarray | None:
        """Forward pass of the neural network."""
        parameter_values_, num_samples = self._preprocess_forward(input_data, weights)
        #print("processing @ 76")
        self.circuits_batch = [self.circuit] * num_samples * self.output_shape[0]

        #print("batching @ 79")
        #print(circuits_batch[0])
        if self.dropouter is not None:
            if self.is_dropout_active == True:
                self.circuits_batch = self.modify_circuit_batch(self.circuits_batch)


        job = self.estimator.run(
            self.circuits_batch,
            [op for op in self._observables for _ in range(num_samples)],
            np.tile(parameter_values_, (self.output_shape[0], 1)),
        )

        try:
            results = job.result()
        except Exception as exc:
            raise QiskitMachineLearningError("Estimator job failed.") from exc
        return self._forward_postprocess(num_samples, results)

    def _backward_postprocess(
        self, num_samples: int, result: EstimatorGradientResult
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """Post-processing during backward pass of the network."""
        num_observables = self.output_shape[0]
        if self._input_gradients:
            input_grad = np.zeros((num_samples, num_observables, self._num_inputs))
        else:
            input_grad = None

        weights_grad = np.zeros((num_samples, num_observables, self._num_weights))
        gradients = np.asarray(result.gradients)
        
        for i in range(num_observables):
            if self._input_gradients:
                input_grad[:, i, :] = gradients[i * num_samples : (i + 1) * num_samples][
                    :, : self._num_inputs
                ]
                weights_grad[:, i, :] = gradients[i * num_samples : (i + 1) * num_samples][
                    :, self._num_inputs :
                ]
            else:
                weights_grad[:, i, :] = gradients[i * num_samples : (i + 1) * num_samples]
        return input_grad, weights_grad

    def _backward(
        self, input_data: np.ndarray | None, weights: np.ndarray | None
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """Backward pass of the network."""
        # prepare parameters in the required format
        parameter_values, num_samples = self._preprocess_forward(input_data, weights)
        input_grad, weights_grad = None, None

        if np.prod(parameter_values.shape) > 0:
            num_observables = self.output_shape[0]
            num_circuits = num_samples * num_observables

            #circuits = self.circuits_batch
            circuits = [self.circuit] * num_samples * self.output_shape[0]
            observables = [op for op in self._observables for _ in range(num_samples)]
            
            param_values = np.tile(parameter_values, (num_observables, 1))

            job = None
            if self._input_gradients:
                job = self.gradient.run(circuits, observables, param_values)
            elif len(parameter_values[0]) > self._num_inputs:
                params = [self._circuit.parameters[self._num_inputs :]] * num_circuits
                job = self.gradient.run(circuits, observables, param_values, parameters=params)
            if job is not None:
                try:
                    results = job.result()
                except Exception as exc:
                    raise QiskitMachineLearningError("Estimator job failed.") from exc

                input_grad, weights_grad = self._backward_postprocess(num_samples, results)

        return input_grad, weights_grad

    def modify_circuit_batch(self, circuits):
        
        new_circuits = [self.dropouter.apply() for _ in circuits]

        return new_circuits