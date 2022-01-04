import numpy as np
import torch
# import matplotlib.pyplot as plt

# Initial parameters
max_input_neurons = 72
max_output_neurons = 2
max_hidden_layers = 2
max_hidden_neurons = 60
max_training_sets = 400
learning_rate = 0.01
max_rounds = 5000 #00

# Initialize the tensors
#
#
#

# Initialize the network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.nn.Sequential(
    torch.nn.Linear(max_input_neurons, max_hidden_neurons),
    torch.nn.ReLU(),
    torch.nn.Linear(max_hidden_neurons, max_hidden_neurons),
    torch.nn.ReLU(),
    torch.nn.Linear(max_hidden_neurons, max_hidden_neurons),
    torch.nn.ReLU(),
    torch.nn.Linear(max_hidden_neurons, max_output_neurons),
)
model.to(device)
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Loss = np.zeros(max_rounds)

# Train the network
for i in range(max_rounds):
    # Forward pass
    y_pred = model(x)

    # Compute and print loss
    loss = loss_fn(y_pred, y)
    print(f'Loss: {loss.item()}')
    # Loss[i] = loss.item()

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()