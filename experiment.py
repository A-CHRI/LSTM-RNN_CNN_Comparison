import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

### Device configuration ###
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Hyperparameters ###
features = 5 # Close, Volume, Open, High, Low (Input_size = 5)
seq_len = 21 # length of window
batch_size = 64 # Must be a power of 2
l_rate = 0.0025
n_epoch = 512 # Must be divisible by 8
n_hidden = int((2/3)*(features * seq_len)) # 2/3 input neurons
dropout = 0.2 # Dropout rate

n_input = features * seq_len
n_output = 2

### Training and test files ###
training_files = ["data/AMZN.csv", "data/BRK.csv", "data/FB.csv", "data/GOOG.csv","data/GOOGL.csv","data/JPM.csv","data/MSFT.csv","data/NVDA.csv","data/TSLA.csv","data/VOO.csv"]
test_file = ["data/AAPL.csv"]


### CNN Model ###
class CNN(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        super(CNN, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
        )

    def forward(self, x):
        x = x.reshape(len(x), -1)
        out = self.linear_relu_stack(x)
        return out


### LSTM Model ###
class LSTM(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers):
        super(LSTM, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.lstm = nn.LSTM(n_input, n_hidden, n_layers, dropout=dropout, batch_first=True) # n_input -> (batch_size, seq_len, input_size)
        self.linear = nn.Linear(n_hidden * seq_len, n_output) # n_hidden * seq_len -> (batch_size, n_output)


    def forward(self, input):
        # Initialize hidden and cell states
        h_0 = torch.zeros(self.n_layers, input.size(0), self.n_hidden).to(device)
        c_0 = torch.zeros(self.n_layers, input.size(0), self.n_hidden).to(device)

        # LSTM layer
        input = input.view(-1, seq_len, features)
        out, _ = self.lstm(input, (h_0, c_0))

        # Linear layer
        out = out.reshape(len(input), -1)
        out = self.linear(out)
        return out

### Dataset ###
class StockData(Dataset):
    def __init__(self, data_files):
        # Load data
        x_arr = []
        y_arr = []
        for i in data_files:
            data = np.loadtxt(i, delimiter=',', skiprows=1, usecols=(1,2,3,4,5))[::-1]

            for i in range(len(data) - (seq_len + 1)):
                x = data[i:i + seq_len, :]
                y = data[i + seq_len, 3:5]

                # Store data in the dataset
                x_arr.append(x)
                y_arr.append(y)

        # Scale data
        self.scaler = StandardScaler()

        # Flatten data
        x_arr = np.reshape(x_arr, (-1, 5))
        y_arr = np.reshape(y_arr, (-1, 1))

        # Scale data
        data_scaled_x = self.scaler.fit_transform(x_arr)
        data_scaled_y = self.scaler.fit_transform(y_arr)

        # Reshape data
        data_scaled_x = np.reshape(data_scaled_x, (-1, seq_len, features))
        data_scaled_y = np.reshape(data_scaled_y, (-1, n_output))
        
        # Set the tensors
        self.x = torch.tensor(data_scaled_x, dtype=torch.float).to(device)
        self.y = torch.tensor(data_scaled_y, dtype=torch.float).to(device)

        self.n_samples = self.x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

### Print and log ###
def print_and_log(string, file):
    print(string)
    with open(file, "a") as f:
        f.write(string + "\n")

if __name__ == '__main__':

    filelocs = ["out/experiment_data_LSTM.txt", "out/experiment_data_CNN.txt"]
    for modeltype in range(2):
        for run_through in range(50):
            if modeltype == 0:
                    
                # Initialize model

                model = LSTM(features, n_hidden, n_output, n_layers=2).to(device)
                loss_fn = nn.MSELoss(reduction='sum')
                optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)


                # Load the training dataset

                dataset_train = StockData(training_files)
                dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)


                # Training loop

                Loss = []

                samples = len(dataset_train)
                iterations = samples // batch_size
                for epoch in range(n_epoch):
                    for i, (x, y) in enumerate(dataloader_train):
                        x = x.view(-1, seq_len, features)
                        y = y.view(-1, n_output)

                        # Forward pass
                        y_pred = model(x)
                        loss = loss_fn(y_pred, y)
                        Loss.append(loss.item())

                        # Backward and optimize
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()


                # Load the test dataset

                dataset_test = StockData(test_file)
                dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
                y_pred_plot = np.array([])


                # Test model

                for i, (x, y) in enumerate(dataloader_test):
                    x = x.view(-1, seq_len, features)
                    y = y.view(-1, n_output)

                    # Forward pass
                    y_pred = model(x)
                    y_pred_plot = np.append(y_pred_plot, y_pred.cpu().detach().numpy())
                    loss = loss_fn(y_pred, y)


                ### Percentual deviation from target ###
                losspercent = 0
                y_pred_plot = dataset_test.scaler.inverse_transform(y_pred_plot.reshape(-1, n_output))
                plot_data = np.loadtxt(test_file[0], delimiter=',', skiprows=1, usecols=(4,5))[::-1]
                targets = plot_data[seq_len:]
                for i, e in enumerate(y_pred_plot):
                    target1 = targets[i, 0]
                    target2 = targets[i, 1]
                    losspercent = losspercent + abs((e[0]-target1)/target1)
                    losspercent = losspercent + abs((e[1]-target1)/target1)
                losspercent = (losspercent/(len(y_pred_plot)*len(y_pred_plot[0])))*100
                print_and_log(str(losspercent), filelocs[modeltype])
            
            else:
                # Initialize model

                model = CNN(n_input, n_hidden, n_output).to(device)
                loss_fn = nn.MSELoss(reduction='sum')
                optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)

                # Load the data training set

                dataset_train = StockData(training_files)
                dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

                # Training loop

                Loss = []

                samples = len(dataset_train)
                iterations = samples // batch_size
                for epoch in range(n_epoch):
                    for i, (x, y) in enumerate(dataloader_train):
                        x = x.view(-1, seq_len, features)
                        y = y.view(-1, n_output)

                        # Forward pass
                        y_pred = model(x)
                        loss = loss_fn(y_pred, y)
                        Loss.append(loss.item())

                        # Backward and optimize
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()


                # Load the test dataset

                dataset_test = StockData(test_file)
                dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
                y_pred_plot = np.array([])

                # Test model

                for i, (x, y) in enumerate(dataloader_test):
                    x = x.view(-1, seq_len, features)
                    y = y.view(-1, n_output)

                    # Forward pass
                    y_pred = model(x)
                    y_pred_plot = np.append(y_pred_plot, y_pred.cpu().detach().numpy())
                    loss = loss_fn(y_pred, y)

                ### Percentual deviation from target ###
                losspercent = 0
                y_pred_plot = dataset_test.scaler.inverse_transform(y_pred_plot.reshape(-1, n_output))
                plot_data = np.loadtxt(test_file[0], delimiter=',', skiprows=1, usecols=(4,5))[::-1]
                targets = plot_data[seq_len:]
                for i, e in enumerate(y_pred_plot):
                    target1 = targets[i, 0]
                    target2 = targets[i, 1]
                    losspercent = losspercent + abs((e[0]-target1)/target1)
                    losspercent = losspercent + abs((e[1]-target1)/target1)
                losspercent = (losspercent/(len(y_pred_plot)*len(y_pred_plot[0])))*100
                print_and_log(str(losspercent), filelocs[modeltype])
