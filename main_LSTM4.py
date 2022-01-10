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
seq_len = 4 # look back period
batch_size = 1
l_rate = 0.000046

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
        data = np.loadtxt(data_file, delimiter=',', skiprows=1, usecols=(1,2,3,4,5))[::-1]

        self.scale_min = data[:, 0].min()
        self.scale_max = data[:, 0].max()
        print(self.scale_min, self.scale_max)

        # Scale data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = self.scaler.fit_transform(data)

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
    # Initialize model
    model = LSTM(features, n_hidden=24, n_output=1, n_layers=2).to(device)
    loss_fn = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)

    # Load the training dataset
    dataset_train = StockData('data-AAPL.csv')
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)

    loader = iter(dataloader_train)
    x_train, y_train = next(loader)

    # Training loop
    n_epoch = 2
    samples = len(dataset_train)
    iterations = samples // batch_size
    for epoch in range(n_epoch):
        for i, (x, y) in enumerate(dataloader_train):
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

    # Load the test dataset
    dataset_test = StockData('data-VOO.csv')
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    y_pred_plot = np.array([])

    # Test model
    for i, (x, y) in enumerate(dataloader_test):
        x = x.view(batch_size, seq_len, features)
        y = y.view(batch_size, 1)

        # Forward pass
        y_pred = model(x)
        y_pred_plot = np.append(y_pred_plot, y_pred.cpu().detach().numpy())
        loss = loss_fn(y_pred, y)

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                .format(epoch+1, n_epoch, i+1, iterations, loss.item()))


    # Plot results
    plot_data = np.loadtxt('data-VOO.csv', delimiter=',', skiprows=1, usecols=(1))[::-1]

    scaler = MinMaxScaler(feature_range=(dataset_test.scale_min, dataset_test.scale_max))
    y_pred_plot = scaler.fit_transform(y_pred_plot.reshape(-1, 1))

    plt.plot(np.arange(len(plot_data)), plot_data, label='VOO')
    plt.plot(np.arange(len(y_pred_plot)) + seq_len, y_pred_plot, label='Prediction')
    plt.legend(loc='best')
    plt.show()