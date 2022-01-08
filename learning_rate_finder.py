import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# Variable for time estimate
time_divisor = 30

### Parameters ###
output_neurons = 1
hidden_layers = 2 # No theoreticle reason to be more than 2
learning_rate = 0.00001
iterations = 10 #500000
days_per_segment = 7 # Usually 21 for stable stocks, and lower the more volatile
input_neurons = days_per_segment * 5
hidden_neurons = int((2 / 3) * input_neurons) # Usually 2/3 the size of the input neurons

training_files = ["data-TSLA-small.csv", "data-GME-small.csv", "data-VOO-small.csv", "data-AMD-small.csv"]
test_files = ["data-AAPL.csv"]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

### Neural Network ###
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_neurons, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, output_neurons),
        ).double()

    def forward(self, x):
        return self.linear_relu_stack(x)

### Methods ###
# Print and log
def print_and_log(string):
    with open("learning_rate_data.txt", "a") as f:
        f.write(string + "\n")


# Import data
def import_data(filenames, days_per_segment):
    set = []
    for i in filenames:
        dataImport = np.loadtxt(i, delimiter=',', skiprows=1, usecols=(1, 2, 3, 4, 5))[::-1]
        data = np.copy(np.transpose(dataImport))
        for l in range (len(data[0]) - (days_per_segment + 1)):
        # Calculate x segment
            x = np.zeros(5 * days_per_segment)
            for j in range(5):
                for k in range(days_per_segment):
                    x[j * days_per_segment + k] = (data[j, k + l] - np.mean(data[j,:])) / np.std(data[j,:])
            # Calculate y segment
            y = (data[0, l + days_per_segment] - np.mean(data[0, :])) / np.std(data[0,:])
            set.append([x, y])
    return data, set

# Train the network
def Train_network(iterations, device, segments, model, loss_fn, optimizer, Loss):
    segments_count = len(segments)
    for i in range(segments_count):
        # Initialize the tensors
        x = segments[i][0]
        inp = torch.tensor(x, device=device).double()

        y = [segments[i][1]]
        outp = torch.tensor(y, device=device).double()

        for j in range(iterations):
            # Forward pass
            y_pred = model(inp)

            # Compute and print loss
            loss = loss_fn(y_pred, outp)
            Loss[j*(i+1)] = loss.item()

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# Test the network
def Test_network(device, segments, model, loss_fn):

    segments_count = len(segments)
    y_plot_pred = np.zeros(segments_count)
    for i in range(segments_count):
        
        # Initialize the tensors
        x = segments[i][0]
        inp = torch.tensor(x, device=device).double()

        y = [segments[i][1]]
        outp = torch.tensor(y, device=device).double()

        y_pred = model(inp)
        y_plot_pred[i] = y_pred.cpu().detach().numpy() * np.std(test_data[0, :]) + np.mean(test_data[0, :])

        # Compute and print loss
        loss = loss_fn(y_pred, outp)
        
        ### TEMPORARY (This is for testing) ###
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losspercent = 0
    for i, e in enumerate(y_plot_pred):
        target = segments[i][1]* np.std(test_data[0, :]) + np.mean(test_data[0, :])
        losspercent = losspercent + abs((e-target)/target)
    losspercent = (losspercent/len(y_plot_pred))*100

    print_and_log(str(round(losspercent, 4)) + " " + str(learning_rate))

    return y_plot_pred




if __name__ == '__main__':
    for i in range(100):
        learning_rate = 0.01/(2.5**i)
        for j in range(5):
            # Import data
            training_data, training_sets = import_data(training_files, days_per_segment)
            test_data, test_sets = import_data(test_files, days_per_segment)


            model = NeuralNetwork().to(device)
            loss_fn = nn.MSELoss(reduction='sum')
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            Loss = np.zeros(iterations * len(training_sets))

            # Train the neural network
            Train_network(iterations, device, training_sets, model, loss_fn, optimizer, Loss)

            # Test the neural network
            y_plot_pred = Test_network(device, test_sets, model, loss_fn)
