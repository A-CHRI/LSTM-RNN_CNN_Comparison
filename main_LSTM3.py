import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import time
import matplotlib.pyplot as plt

### Device configuration ###
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Parameters ###
features = 5 # Close, Volume, Open, High, Low (Input_size = 5)
batch_size = 7 # 1 weeks data (Seq_length = 7)
lr_rate = 0.00001

n_input = batch_size * features  # Amount of input neurons = batch_size * features
h_size = int((2 / 3) * n_input) # Amoount of hidden neurons per hidden layer = (2/3) * input_size
n_output = 1 # Output neurons
n_layers = 2 # Amount of hidden layers


### Network model ###
class LSTM(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers):
        super(LSTM, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lstm = nn.LSTM(n_input, n_hidden, n_layers, batch_first=True) # n_input -> (batch_size, seq_len, input_size)
        self.linear = nn.Linear(n_hidden, n_output)

    def forward(self, input):
        # Initialize hidden and cell states
        self.h_0 = torch.zeros(self.n_layers, input.size(0), self.n_hidden)
        self.c_0 = torch.zeros(self.n_layers, input.size(0), self.n_hidden)
        
        # LSTM layer
        print(input.size(-1))
        input = input.view(batch_size, features, -1)
        print(input.size(-1))
        print(input)
        out, _ = self.lstm(input, (self.h_0, self.c_0)) # out -> (batch_size, seq_len, hidden_size)

        # # Reshape the output
        # out = out[:, -1, :]

        # Linear layer
        out = self.linear(out.view(len(input), -1))
        return out

### Method for creating the batches ###
def create_batches(data, batch_size):
    data_batches = []
    for i in range(len(data) - (batch_size + 1)):
        seq = torch.from_numpy(data[i:i + batch_size]).to(device)
        if i == 0:
            print(seq)
        pred = torch.tensor(data[i + batch_size, 0]).to(device)
        data_batches.append((seq, pred))
    return data_batches

### Method for printing and logging the data ###
def print_and_log(string):
    print(string)
    with open("log.txt", "a") as f:
        f.write(string + "\n")

### Main script ###
if __name__ == '__main__':
    # Print the parameter info
    # print_and_log("\nTraining on: " + str(training_files) + "\nTesting on: " + str(test_files) + "\nLogging to: log.txt")
    # print_and_log("\nInput neurons: " + str(input_neurons) + "\nOutput neurons: " + str(output_neurons) + "\nHidden layers: " + str(hidden_layers) + "\nHidden neurons: " + str(hidden_neurons))
    # print_and_log("\nLearning rate: " + str(learning_rate) + "\nIterations: " + str(iterations) + "\nDays per segment: " + str(days_per_segment))
    # print_and_log("\nDevice: " + str("CUDA" if torch.cuda.is_available() else "CPU") + "\n")

    # --- Load the data
    print_and_log("\nLoading data...")
    timer_start = time.perf_counter()

    # Import the data
    data = np.loadtxt("data-AAPL.csv", delimiter=",", skiprows=1, usecols=(1,2,3,4,5), dtype=np.float32)[::-1]

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    # Split the data into training and testing
    split = int(0.8 * len(data)) # 80% training, 20% testing
    train_data = data[:split]
    test_data = data[split:]

    # # Create the training and testing datasets
    # train_data = torch.from_numpy(train_data).view(-1).to(device)
    # test_data = torch.from_numpy(test_data).view(-1).to(device)

    # Create the batches
    train_batches = create_batches(train_data, batch_size)
    test_batches = create_batches(test_data, batch_size)

    timer_end = time.perf_counter()
    print_and_log("\nDone! (" + str(round(timer_end - timer_start, 4)) + " seconds)")
    # ---

    # --- Initialize the network
    print_and_log("\nInitializing network...")
    timer_start = time.perf_counter()

    model = LSTM(n_input, h_size, n_output, n_layers)
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

    timer_end = time.perf_counter()
    print_and_log("\nDone! (" + str(round(timer_end - timer_start, 4)) + " seconds)")
    # ---

    # Training loop
    n_epochs = 2 
    total_samples = len(train_batches)
    n_iterations = int(total_samples / batch_size)

    for epoch in range(n_epochs):
        for seq, pred in train_batches:
            # Reshape the data
            seq = seq.reshape(-1)
            print(seq.size(-1))
            # Forward pass
            y_pred = model(seq)
            loss = loss_fn(y_pred, pred)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print the loss
            print_and_log(f"Epoch: {epoch}/{n_epochs} | Loss: {loss.item()}")
