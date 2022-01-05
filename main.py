from multiprocessing import Condition
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

# Initial parameters
max_input_neurons = 7 #72
max_output_neurons = 2
max_hidden_layers = 2
max_hidden_neurons = 16 #60
max_training_sets = 400
learning_rate = 0.01
max_rounds = 50 #500000


# Initialize the tensors from data file
dataImport = np.loadtxt('data-VOO.csv', delimiter=',', skiprows=1, usecols=(1,2,3,4,5))[::-1]
data = np.copy(dataImport)

trainingsets = []

while len(trainingsets) < max_training_sets and (len(trainingsets) + 1)*(max_input_neurons) < len(data):
    x = data[max_input_neurons*len(trainingsets):max_input_neurons*(len(trainingsets)+1), 0]
    y = np.array([1 if data[max_input_neurons*(len(trainingsets)+1) + 1][0] > x[-1] else 0, 1 if data[max_input_neurons*(len(trainingsets)+1) + 1][0] < x[-1] else 0])
    trainingsets.append([x, y])

# Initialize the network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = nn.Sequential(
    nn.Linear(max_input_neurons, max_hidden_neurons),
    nn.ReLU(),
    nn.Linear(max_hidden_neurons, max_hidden_neurons),
    nn.ReLU(),
    nn.Linear(max_hidden_neurons, max_hidden_neurons),
    nn.ReLU(),
    nn.Linear(max_hidden_neurons, max_output_neurons),
).double()
model.to(device)
loss_fn = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

Loss = np.zeros(max_rounds)

# Train the network
for i in range(len(trainingsets)):
    # Initialize the tensors
    x = trainingsets[i][0]
    inp = torch.tensor(x).double()

    y = trainingsets[i][1]
    outp = torch.tensor(y).double()

    for j in range(max_rounds):
        # Forward pass
        y_pred = model(inp)

        # Compute and print loss
        loss = loss_fn(y_pred, outp)
        print(f'Week: {i}, Day:{j}, Loss: {loss.item()}')
        Loss[j] = loss.item()

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Plot the loss function
plt.plot(Loss)
plt.grid(True)
plt.show()