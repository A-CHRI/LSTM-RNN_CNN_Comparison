from multiprocessing import Condition
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

# Initial parameters
max_output_neurons = 2
max_hidden_layers = 2
max_hidden_neurons = 200 #60
max_training_sets = 400
learning_rate = 0.01
max_rounds = 50 #500000
days_per_segment = 7
max_input_neurons = days_per_segment*5

# Filenames
training_file = "data-VOO.csv"
test_file = "data-AMD.csv"


# Initialize training data from data file
training_dataImport = np.loadtxt(training_file, delimiter=',', skiprows=1, usecols=(1,2,3,4,5))[::-1]
training_data = np.copy(training_dataImport)

trainingsets = []

while len(trainingsets) < max_training_sets and (len(trainingsets) + 1) * (max_input_neurons) < len(training_data):
    x = np.zeros(5*days_per_segment)
    for i in range(5):
        for j in range(days_per_segment):
            x[i*days_per_segment+j] = training_data[days_per_segment * len(trainingsets)+j, i]
    y = np.array([1 if np.mean(training_data[max_input_neurons * (len(trainingsets) + 1): max_input_neurons * (len(trainingsets) + 2) + 1][0]) > np.mean(x) else 0, 1 if np.mean(training_data[max_input_neurons*(len(trainingsets) + 1): max_input_neurons * (len(trainingsets) + 2) + 1][0]) < np.mean(x) else 0])
    trainingsets.append([x, y])

# Initialize test data from data file
test_dataImport = np.loadtxt('data-AMD.csv', delimiter=',', skiprows=1, usecols=(1,2,3,4,5))[::-1]
test_data = np.copy(test_dataImport)


testsets = []

while len(testsets) < max_training_sets and (len(testsets) + 1) * (max_input_neurons) < len(test_data):
    x = np.zeros(5*days_per_segment)
    for i in range(5):
        for j in range(days_per_segment):
            x[i*days_per_segment+j] = test_data[max_input_neurons * len(testsets) + j, i]
    y = np.array([1 if test_data[max_input_neurons * (len(testsets) + 1) + 1][0] > x[-1] else 0, 1 if test_data[max_input_neurons * (len(testsets) + 1) + 1][0] < x[-1] else 0])
    testsets.append([x, y])

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

Loss = np.zeros(max_rounds * len(trainingsets))

# Train the network
weeks_train = len(trainingsets)
for i in range(weeks_train):
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
        print(f'Done: {int((100/weeks_train) * i)}%, Week: {i}, Iteration:{j}, Loss: {loss.item()}')
        Loss[j*(i+1)] = loss.item()

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Test the network
weeks_test = len(testsets)
Losstest = np.zeros(max_rounds * len(test_data))
y_plot_pred = np.array([])
for i in range(weeks_test):
    # Initialize the tensors
    x = testsets[i][0]
    inp = torch.tensor(x).double()

    y = testsets[i][1]
    outp = torch.tensor(y).double()

    for j in range(max_rounds):
        # Forward pass
        y_pred = model(inp)
        y_plot_pred = np.append(y_plot_pred, y_pred.detach().numpy())

        # Compute and print loss
        loss = loss_fn(y_pred, outp)
        print(f'Done: {int((100/weeks_test) * i)}%, Week: {i}, Iteration:{j}, Loss: {loss.item()}')
        Losstest[j*(i+1)] = loss.item()

wrong = 0
for e in Losstest:
    if e > 0.5:
        wrong = wrong + 1
print(str(wrong/len(Losstest)))

# Plot the loss function
plt.plot(Loss)
plt.grid(True)
plt.show()

## Set up plot for the data
# AMD data
x_plot_test = np.arange(len(test_data[0]))
y_plot_test = np.array(test_data[0])
plt.plot(x_plot_test, y_plot_test, label='AMD daily close price')

# Prediction data
plt.plot(weeks_test, y_plot_pred, label='Prediction')

# Plotting
plt.grid(True)
plt.legend()
plt.show()