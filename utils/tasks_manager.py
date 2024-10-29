from utils.constants import Tasks
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import os

from sympy.ntheory import discrete_log

def generate_task(task: Tasks, num_samples_train = 20, num_samples_test = 9, show = True, seed = 30):
    """
    returns X_train, y_train, X_test, y_test, X_, y_
    """
    # I want to ensure that the data is always the same
    np.random.seed(seed)

    lb, ub = -1, 1
    lb *= 0.95
    ub *= 0.95

    total_samples = num_samples_train + num_samples_test
    X_ = np.linspace(lb, ub, num=50).reshape(-1, 1)
    X = np.linspace(lb, ub, num=total_samples).reshape(-1, 1)

    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    X_train_indices = indices[:num_samples_train]
    X_test_indices = indices[num_samples_train:]
    
    X_train = X[X_train_indices, :]
    X_test = X[X_test_indices, :]
    
    noise = np.random.normal(loc=0, scale=0.15, size=num_samples_train)
    noise_test = np.random.normal(loc=0, scale=0.15, size=num_samples_test)
    if task == Tasks.REGRESSION_SINE:
        f = lambda x: np.sin(math.pi*x)
    elif task == Tasks.TEST:
        # I want sign(x) as the function
        f = lambda x: np.sign(x)*2


    else:
        print(task)
        raise ValueError("Task not recognized")
    
    # I want to add to y_train Gaussian noise with amplitude equal to 0.4, zero mean and a standard deviation of 0.5.

    #print("Sto testando alcune cose... noise a 0")

    y_train = f(X_train[:, 0]) + noise
    y_test = f(X_test[:, 0]) + noise_test
    y_ = f(X_[:, 0])

    if task == Tasks.TEST_LOG:
        p = 29
        g = 11
        s = 13

        results = {x: check_log_in_range_safe(x, p, g, s) for x in range(1, p)}
        x = list(results.keys())
        y = list(results.values())
        X = np.array(x).reshape(-1, 1)/p
        y = np.array(y).reshape(-1, 1)

        X_train_indices = indices[:num_samples_train]
        X_test_indices = indices[num_samples_train:]
        
        X_train = X[X_train_indices, :]
        X_test = X[X_test_indices, :]

        y_train = y[X_train_indices]
        y_test = y[X_test_indices]

        X_ = X
        y_ = y


    save_task(task, X_train, y_train, X_test, y_test)

    test_color = '#f56d2a'
    train_color = '#3b5af5'
    function_color = '#c90000'

    if show:
        plt.plot(X_, f(X_), 'r--', color=function_color)  # Change to magenta or any other color for the function plot
        plt.scatter(X_train, y_train, color=train_color, label='Train')  # Use cyan for training points
        plt.scatter(X_test, y_test, color=test_color, label='Test')  # Use lime for testing points
        plt.legend()
        #plt.show()
        plt.savefig(f"data/task_{task.value}/plot.png")
        plt.clf()


    return X_train, y_train, X_test, y_test, X_, y_ 

def generate_classification_task(task: Tasks, num_samples_train = 15, num_samples_test = 5, show = True, seed = 0):

    np.random.seed(seed)
    total_samples = num_samples_train + num_samples_test
    #X_ = np.linspace(lb, ub, num=50).reshape(-1, 1)
    #X = np.linspace(lb, ub, num=total_samples).reshape(-1, 1)

    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    X_train_indices = indices[:num_samples_train]
    X_test_indices = indices[num_samples_train:]
    
    if task == Tasks.CLASSIFICATION_MOONS:
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=total_samples, noise=0.1, random_state=seed)
        X = X/2 # scale the data
        X_train = X[X_train_indices, :]
        X_test = X[X_test_indices, :]
        y_train = y[X_train_indices]
        y_test = y[X_test_indices]

        X_, y_ = make_moons(n_samples=50, noise=0.1, random_state=seed)
        X_ = X_/2

        
        

    save_task(task, X_train, y_train, X_test, y_test)
    if show:
        plt.scatter(X_train[y_train[:] == 0, 0], X_train[y_train[:] == 0, 1], color='r', label='Train class 0')

        plt.scatter(X_train[y_train[:] == 1, 0], X_train[y_train[:] == 1, 1], color='b', label='Train class 1')

        plt.scatter(X_[y_[:] == 0, 0], X_[y_[:] == 0, 1], color='r', label='Test class 0', marker='x')

        plt.scatter(X_[y_[:] == 1, 0], X_[y_[:] == 1, 1], color='b', label='Test class 1', marker='x')
        plt.legend()
        plt.show()

    return X_train, y_train, X_test, y_test, X_, y_

def generate_iris_task(ratio = 0.7,  show = True):
    from sklearn.datasets import load_iris

    iris_data = load_iris()
    print(iris_data.DESCR)

    features = iris_data.data
    labels = iris_data.target

    indices = np.arange(150)
    np.random.shuffle(indices)
    num_samples_train = int(ratio*150)

    X_train_indices = indices[:num_samples_train]
    X_test_indices = indices[num_samples_train:]
    
    X_train = features[X_train_indices, :]
    X_test = features[X_test_indices, :]

    y_train = labels[X_train_indices]
    y_test = labels[X_test_indices]

    return X_train, y_train, X_test, y_test
    
def prepare_features(X, task):
    if task == Tasks.CLASSIFICATION_MOONS:
        fX = np.hstack((2*np.arcsin(X[:, 0]/2), 2*np.arcsin(X[:, 1]/2), 2*np.arccos(X[:, 0]**2/4), 2*np.arccos(X[:, 1]**2/4)))
        fX = fX.reshape(-1, 4)
        return fX
    elif task in [Tasks.REGRESSION_SINE, Tasks.REGRESSION_ABS, Tasks.REGRESSION_EXPONENTIAL, Tasks.REGRESSION_SAWTOOTH, Tasks.TEST]:
        fX = np.hstack((np.arcsin(X), np.arccos(X**2)))
        return fX
    elif task == Tasks.TEST_LOG:
        fX = np.hstack((np.sin(np.pi*X*2), X))
        return fX
    elif task == Tasks.IRIS:
        return X
    else:
        raise ValueError("Task not recognized")


def save_task(task, X_train, y_train, X_test, y_test):
    # create a dir for the task
    task_dir = f"data/task_{task.value}"
    os.makedirs(task_dir, exist_ok=True)
    np.save(f"{task_dir}/X_train.npy", X_train)
    np.save(f"{task_dir}/y_train.npy", y_train)
    np.save(f"{task_dir}/X_test.npy", X_test)
    np.save(f"{task_dir}/y_test.npy", y_test)
    print(f"Task {task.value} saved at {task_dir}")

def check_log_in_range_safe(x, p=19, g=5, s=17):
    if x not in range(1, p):
        return False
    range_upper = s + (p - 3) // 2
    while range_upper > p:
        range_upper -= p
    try:
        log_gx = discrete_log(p, x, g)
        if range_upper >= s:
          return s <= log_gx <= range_upper
        else:
          return range_upper <= log_gx <= s
    except ValueError:
        # Return False if the discrete logarithm does not exist
        return None


