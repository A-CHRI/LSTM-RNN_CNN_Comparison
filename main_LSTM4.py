import numpy as np
from numpy.core.fromnumeric import shape
import torch
from torch._C import dtype
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


### Device configuration ###
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Parameters ###
features = 5 # Close, Volume, Open, High, Low (Input_size = 5)
seq_len = 7 # 1 weeks data
batch_size = 1

### LSTM Model ###
class LSTM(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers):
        super(LSTM, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.lstm = nn.LSTM(n_input, n_hidden, n_layers, batch_first=True) # n_input -> (batch_size, seq_len, input_size)
        self.linear = nn.Linear(n_hidden * seq_len, n_output)


    def forward(self, input):
        # Initialize hidden and cell states
        h_0 = torch.zeros(self.n_layers, input.size(0), self.n_hidden).to(device)
        c_0 = torch.zeros(self.n_layers, input.size(0), self.n_hidden).to(device)

        # LSTM layer
        input = input.view(batch_size, seq_len, features)
        out, _ = self.lstm(input, (h_0, c_0))

        # Linear layer
        out = out.reshape(len(input), -1)
        out = self.linear(out)
        return out

### Dataset ###
class StockData(Dataset):
    def __init__(self, data_file):
        # Load data
        data = np.loadtxt(data_file, delimiter=',', skiprows=1, usecols=range(1,6))[::-1]

        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)

        x_arr = []
        y_arr = []

        for i in range(len(data_scaled) - (seq_len + 1)):
            x = data_scaled[i:i + seq_len, :]
            y = data_scaled[i + seq_len, 0]

            # Store data in the dataset
            x_arr.append(x)
            y_arr.append(y)

        self.x = torch.tensor(x_arr, dtype=torch.float).to(device)
        self.y = torch.tensor(y_arr, dtype=torch.float).to(device)

        self.n_samples = self.x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


if __name__ == '__main__':
    # # Load data
    # data_train = np.loadtxt('data-AAPL.csv', delimiter=',', skiprows=1, usecols=range(1,6))[::-1]
    # data_test = np.loadtxt('data-VOO.csv', delimiter=',', skiprows=1, usecols=range(1,6))[::-1]

    # # Scale data
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # data_train_scaled = scaler.fit_transform(data_train)
    # data_test_scaled = scaler.transform(data_test)

    # # Segment data
    # data_train_segmented = []
    # for i in range(len(data_train_scaled) - (seq_len + 1)):
    #     x = data_train_scaled[i:i + seq_len]
    #     y = data_train_scaled[i + seq_len, 0]
    #     data_train_segmented.append([x, y])

    # print(data_train_segmented[:5])

    # # Create the tensors
    # x_train = torch.tensor(np.array([i[0] for i in data_train_segmented]), dtype=torch.float).to(device)
    # y_train = torch.tensor(np.array([i[1] for i in data_train_segmented]), dtype=torch.float).to(device)

    # print(x_train[:5], y_train[:5])

    # Initialize model
    model = LSTM(features, n_hidden=24, n_output=1, n_layers=2).to(device)
    loss_fn = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Load the dataset
    dataset = StockData('data-AAPL.csv')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    loader = iter(dataloader)
    x_train, y_train = next(loader)

    # Training loop
    n_epoch = 2
    samples = len(dataset)
    iterations = samples // batch_size
    for epoch in range(n_epoch):
        for i, (x, y) in enumerate(dataloader):
            x = x.view(batch_size, seq_len, features)
            y = y.view(batch_size, 1)

            # Forward pass
            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, n_epoch, i+1, iterations, loss.item()))

    # Train model
    # n_epoch = 5
    # for epoch in range(n_epoch):
    #     start = time.time()
    #     total_loss = 0
    #     for x, y in zip(x_train, y_train):
    #         x = x.view(1, seq_len, features)
    #         y = y.view(1, 1)
    #         y_pred = model(x)
    #         loss = loss_fn(y_pred, y)
    #         total_loss += loss
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #     print('Epoch: {}, Loss: {:.6f}, Time: {:.4f}'.format(epoch + 1, total_loss, time.time() - start))

    # Test model

    # Plot results