import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

# Initial parameters
max_output_neurons = 1
max_hidden_layers = 2
max_hidden_neurons = 60 #60
max_training_sets = 400
learning_rate = 0.01
max_rounds = 50 #500000
days_per_segment = 7
max_input_neurons = days_per_segment*5

# Filenames
training_files = ["data-VOO.csv", "data-AMD.csv"]
test_file = "data-GME.csv"


# Initialize training data from data files
trainingsets = []
for i in training_files:
    size = 0
    training_dataImport = np.loadtxt(i, delimiter=',', skiprows=1, usecols=(1,2,3,4,5))[::-1]
    training_data = np.copy(np.transpose(training_dataImport))

    #while len(trainingsets) < max_training_sets and (len(trainingsets) + 1) * (days_per_segment) < len(training_data[0]):
    # for l in range( len(training_data[0] - (days_per_segment + 1)) ):
    #     x = np.zeros(5 * days_per_segment)
    #     for j in range(5):
    #         for k in range(days_per_segment):
    #             x[j * days_per_segment + k] = training_data[j, days_per_segment * size + k]
    #     y = training_data[0, days_per_segment * size + 1]
    #     trainingsets.append([x, y])
    #     print([x,y])
    #     size += 1

    for l in range (len(training_data[0]) - (days_per_segment + 1)):
        # Calculate x segment
        x = np.zeros(5 * days_per_segment)
        for j in range(5):
            for k in range(days_per_segment):
                x[j * days_per_segment + k] = training_data[j, k + l]
        # Calculate y segment
        y = training_data[0, l + days_per_segment]
        trainingsets.append([x, y])

# Initialize test data from data file
test_dataImport = np.loadtxt(test_file, delimiter=',', skiprows=1, usecols=(1,2,3,4,5))[::-1]
test_data = np.copy(np.transpose(test_dataImport))


testsets = []

for i in test_file:
    size = 0
    test_dataImport = np.loadtxt(i, delimiter=',', skiprows=1, usecols=(1,2,3,4,5))[::-1]
    test_data = np.copy(np.transpose(test_dataImport))
    for l in range (len(test_data[0]) - (days_per_segment + 1)):
        # Calculate x segment
        x = np.zeros(5 * days_per_segment)
        for j in range(5):
            for k in range(days_per_segment):
                x[j * days_per_segment + k] = test_data[j, k + l]
        # Calculate y segment
        y = test_data[0, l + days_per_segment]
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
print("Training the network...")
segments_train = len(trainingsets)
for i in range(segments_train):
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
        print(f'Done: {int((100/segments_train) * i)}%, Segment: {i}, Iteration:{j}, Loss: {loss.item()}')
        Loss[j*(i+1)] = loss.item()

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Test the network
print("Testing the network...")
segments_test = len(testsets)
predtest = np.zeros(len(testsets))
y_plot_pred = np.array([])
for i in range(segments_test):
    # Initialize the tensors
    x = testsets[i][0]
    inp = torch.tensor(x).double()

    y = testsets[i][1]
    outp = torch.tensor(y).double()

    y_pred = model(inp)
    y_plot_pred = np.append(y_plot_pred, y_pred.detach().numpy()[0])

    # Compute and print loss
    loss = loss_fn(y_pred, outp)
    print(f'Done: {int((100/segments_test) * i)}%, Week: {i}, Loss: {loss.item()}')
    predtest[i] = loss.item()

losspercent = 0
for i, e in enumerate(predtest):
    losspercent = losspercent + (testsets[i][1]-e)/testsets[i][1]

losspercent/len(predtest)
print(losspercent)
print(y_plot_pred)


# Plot the loss function
plt.plot(Loss)
plt.grid(True)
plt.show()

## Set up plot for the data
# AMD data
x_plot_test = np.arange(len(test_data[0]))
y_plot_test = np.array(test_data[0])
plt.plot(x_plot_test, y_plot_test, label='Test file daily closing price')

# Prediction data
x_plot_pred = np.arange(len(testsets)) * days_per_segment
plt.plot(x_plot_pred, y_plot_pred, label='Prediction')

# Plotting
plt.grid(True)
plt.legend()
plt.show()