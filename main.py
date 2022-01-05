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
max_rounds = 5000 #00


# Initialize the tensors from data file
data = np.loadtxt('data-VOO.csv', delimiter=',', skiprows=1, usecols= (1,2,3,4,5) )

x = data[0, 0:max_input_neurons]
inp = torch.tensor(x)

y = np.array([1 if data[max_input_neurons + 1][0] > x[-1] else 0, 1 if data[max_input_neurons + 1][0] < x[-1] else 0])
outp = torch.tensor(y)


# Initialize the network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(max_input_neurons, max_hidden_neurons),
    nn.ReLU(),
    nn.Linear(max_hidden_neurons, max_hidden_neurons),
    nn.ReLU(),
    nn.Linear(max_hidden_neurons, max_hidden_neurons),
    nn.ReLU(),
    nn.Linear(max_hidden_neurons, max_output_neurons),
)
model.to(device)
loss_fn = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

Loss = np.zeros(max_rounds)

# Train the network
for i in range(max_rounds):
    # Forward pass
    y_pred = model(inp)

    # Compute and print loss
    loss = loss_fn(y_pred, outp)
    print(f'Loss: {loss.item()}')
    Loss[i] = loss.item()

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Plot the loss function
plt.plot(Loss)
plt.grid(True)

# Plot the training data
plt.subplot(121)
plt.plot(x, y, 'g.');
plt.plot(x, y_pred.cpu().detach().numpy(), 'r.')
train_error = loss_fn(y_pred, outp).item()
plt.title('Training error: {:.2f}'.format(train_error))

#Plot the test data
# x_t = torch.tensor(np.expand_dims(x_test,1), dtype=torch.float32, device=device)
# y_t = torch.tensor(np.expand_dims(y_test,1), dtype=torch.float32, device=device)
# y_t_pred = model(x_t)
# x_all = np.linspace(-1,1,1000)
# x_all_t = torch.tensor(np.expand_dims(x_all,1), dtype=torch.float32, device=device)
# y_all_t = model(x_all_t)
# plt.subplot(122)
# plt.plot(x_all, y_all_t.cpu().detach().numpy(), 'r-');
# plt.plot(x_test, y_t_pred.cpu().detach().numpy(), 'r.')
# plt.plot(x_test, y_test, 'b.');
# test_error = loss_fn(y_t_pred, y_t).item()
# plt.title('Test error: {:.2f}'.format(test_error));