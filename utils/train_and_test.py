import time
import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt

import utils.tasks_manager as tasks_manager
import utils.constants as constants

def train(model, optimizer, train_dataloader, test_dataloader, epochs, device, task_type = "REGRESSION", verbose=True):
    #check if mode is "REGRESSION" or "CLASSIFICATION"
    
    assert task_type in ["REGRESSION", "CLASSIFICATION"], "Task type must be either 'REGRESSION' or 'CLASSIFICATION'"

    model.train()  # Set model to training mode
    model.activate_dropout()

    total_loss_train = []
    total_loss_test = []
    performance_on_train, performance_on_test = [], []

    epoch_bar = tqdm.tqdm(range(epochs), desc='Epoch', position=0)

    for epoch in epoch_bar:
        train_loss = 0
        train_correct = 0
        model.train() 
        model.activate_dropout()

        batch_bar = tqdm.tqdm(train_dataloader, desc='Batch', position=1, leave=False)
        for X, y in batch_bar:

            X = X.to(device)
            y = y.to(device).view(-1, 1)  # -1 infers the size for that dimension based on the number of elements

            optimizer.zero_grad()

            # Forward pass: Compute predicted y by passing X to the model
            y_pred = model(X)

            # Compute and print loss
            loss = model.loss_func(y_pred, y)
            train_loss += loss.item()
            print(f"Loss: {loss.item()}")
            if task_type == "CLASSIFICATION":
                predicted = y_pred >= 0.5
                train_correct += (predicted == y).sum().item()
                print(f"Correct: {train_correct}")

            loss.backward()
            optimizer.step()
        
        #print the number of elements in train_dataloader
        average_loss = train_loss / len(train_dataloader.dataset)
        if task_type == "CLASSIFICATION":
            performance_on_train.append(train_correct/len(train_dataloader.dataset))
        else:
            performance_on_train.append(average_loss)

        #train_accuracy = train_correct /  len(train_dataloader)
        total_loss_train.append(average_loss)
        #total_accuracy_train.append(train_accuracy)
        epoch_bar.set_description(f'avg loss: {average_loss:.4f}')


        # I want to compute test loss pl
        test_loss = 0
        model.eval()
        model.deactivate_dropout()
        with torch.no_grad():
            for X, y in test_dataloader:
                test_correct = 0
                X = X.to(device)
                y = y.to(device).view(-1, 1)
                y_pred = model(X)
                if task_type == "CLASSIFICATION":
                    predicted = y_pred >= 0.5
                    test_correct += (predicted == y).sum().item()
                loss = model.loss_func(y_pred, y)
                test_loss += loss.item()

        #test_loss /= len(test_dataloader.dataset)
        if task_type == "CLASSIFICATION":
            performance_on_test.append(test_correct/len(test_dataloader.dataset))
        else:
            performance_on_test.append(test_loss/len(test_dataloader.dataset))

        if verbose:
            print("\n")
            print(f"Test loss: {test_loss:.4f}")
            print(f"Train loss: {average_loss:.4f}")
            print("\n")


        total_loss_test.append(test_loss / len(test_dataloader.dataset))


    return total_loss_train, total_loss_test, performance_on_train, performance_on_test

#def test(model):
#    return True


def grid_evaluator(model, size=20, show=True):
    #build a grid of points in [-1, 1] x    [-1, 1]
    x = np.linspace(-0.5, 1, size)
    y = np.linspace(-0.5, 0.5, size)
    grid = np.meshgrid(x, y)

    grid = np.array([grid[0].flatten(), grid[1].flatten()]).T
    fGrid = tasks_manager.prepare_features(grid, constants.Tasks.CLASSIFICATION_MOONS)

    #fGrid_tensor = torch.from_numpy(fGrid).float().to(model.device)
    fGrid_tensor = torch.from_numpy(fGrid).float().to("cpu")

    model.eval()
    model.deactivate_dropout()

    with torch.no_grad():
        y_predicted = model(fGrid_tensor).cpu()

    #plot the points in xx
    fig, ax = plt.subplots()
    ax.scatter(grid[:, 0], grid[:, 1], c=y_predicted, cmap='coolwarm')
    plt.show()


    return True