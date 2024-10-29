from enum import Enum

class Tasks(Enum):
    REGRESSION_SINE = "regression_sine"
    REGRESSION_ABS = "regression_abs"
    REGRESSION_EXPONENTIAL = "regression_exponential"
    REGRESSION_SAWTOOTH = "regression_sawtooth"
    REGRESSION_GAUSSIAN = "regression_gaussian"
    CLASSIFICATION_MOONS = "classification_moons"
    TEST_LOG = "test_log"
    IRIS = "iris"
    TEST = "test"

class Optimizers(Enum):
    COBYLA = "COBYLA"
    ADAM = "ADAM"
    L_BFGS_B = "L_BFGS_B"

class ModelType(Enum): #PQC structures?
    ROTATIONAL_REGRESSOR = "ROTATIONAL_REGRESSOR"
    CLASSIFICATOR = "CLASSIFICATOR"
    IRIS_CLASSIFICATOR = "IRIS_CLASSIFICATOR"
    XXX = "XXX"