import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

### Parameters ###
features = 1
batch_size = 7
input_neurons = batch_size * features
output_neurons = 1
lstm_layers = 2
lstm_neurons = int((2 / 3) * input_neurons)


class LSTM(nn.Module):
    def __init__(self, input_neurons=1, hidden_neurons=100, output_neurons=1):
        super(LSTM, self).__init__()
        self.hidden_neurons = hidden_neurons
        self.lstm = nn.LSTM(input_neurons, hidden_neurons, num_layers=2)
        self.lstm = nn.Linear(hidden_neurons, output_neurons)
        self.hidden_layer = (torch.zeros(1, 1, hidden_neurons),
                            torch.zeros(1, 1, hidden_neurons))

    def forward(self, x):
        out, _ = self.lstm(x, self.hidden_layer)
        out = self.linear(out[0])
        return out


### Method for loading and preprocessing data ###
def Load_data(data, segment_size):
    data = np.transpose(np.loadtxt(data, delimiter=',', skiprows=1, usecols=1)[::-1])
    scalar = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scalar.fit_transform(data.reshape(-1, 1))
    training_size = int(len(data_scaled) * 0.65) # 65% of the data for training
    training_data, test_data = data_scaled[0:training_size], data_scaled[training_size:]
    trainX, trainY = Segment_data(training_data, segment_size)
    testX, testY = Segment_data(test_data, segment_size)
    return trainX, trainY, testX, testY

### Method for segmenting the data ###
def Segment_data(data, segment_size=1):
    dataX, dataY = [], []
    for i in range(len(data) - segment_size):
        a = data[i:(i + segment_size), 0]
        dataX.append(a)
        dataY.append(data[i + segment_size, 0])
    return dataX, dataY

if __name__ == '__main__':
    # Load the data
    trainX, trainY, testX, testY = Load_data('data-AAPL.csv', 7)
    print(str(trainX[:5]) + '\n' + str(trainY[:5]) + '\n' + str(testX[:5]) + '\n' + str(testY[:5]))

    model = LSTM(input_neurons, lstm_neurons, output_neurons)
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the network
    iterations = 10
    for i in range(len(trainX)):
        # Initialize the tensors
        x = torch.tensor(trainX[i], device=device).view(1, -1)
        print(x.shape)
        y = torch.tensor(trainY[i], device=device).view(1, -1)
        print(y.shape)

        for j in range(iterations):
            # Forward pass
            y_pred = model(x)

            # Compute loss
            loss = loss_fn(y_pred, y)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Create array of predictions
    predictions = np.array([])

    # Test the network
    for i in range(len(testX)):
        # Initialize the tensors
        x = torch.tensor(testX[i], device=device).view(1, -1)
        y = torch.tensor(testY[i], device=device).view(1, -1)

        # Forward pass
        y_pred = model(x)

        # Append to array
        predictions = np.append(predictions, y_pred.detach().numpy())

        # Compute loss
        loss = loss_fn(y_pred, y)

        # Print the loss
        print('Loss: ', loss.item())

    # Plot the predictions and the data
    plt.plot(predictions, label='Predictions')
    plt.plot(testY, label='Data')
    plt.legend()
    plt.show()