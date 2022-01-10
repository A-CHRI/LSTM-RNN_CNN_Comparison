import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, output_size=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, hidden_size),
                            torch.zeros(1, 1, hidden_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        prediction = self.linear(lstm_out.view(len(input_seq), -1))
        return prediction[-1]

def Create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def Create_inout_sequence(input_data, look_back=1):
    inout_seq = []
    L = len(input_data)
    for i in range(L-look_back):
        inout_seq.append((input_data[i:i+look_back], input_data[i+look_back:i+look_back+1]))
    return np.array(inout_seq)

if __name__ == '__main__':
    # Import data
    data_file = "data-AAPL.csv"
    data = np.transpose(np.loadtxt(data_file , delimiter=',', skiprows=1, usecols=1)[::-1])

    scalar = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scalar.fit_transform(data.reshape(-1, 1))

    training_size = int(len(data_scaled) * 0.65) # 65% of the data for training
    test_size = len(data_scaled) - training_size
    training_data, test_data = data_scaled[0:training_size], data_scaled[training_size:]

    # Create the datasets
    look_back = 100
    trainX, trainY = Create_dataset(training_data, look_back)
    testX, testY = Create_dataset(test_data, look_back)
    train_inout_seq = Create_inout_sequence(trainX, look_back)

    # Reshape
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # Initialize the model
    model = LSTM()
    loss_fn = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    for epoch in range(epochs):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_size),
                                 torch.zeros(1, 1, model.hidden_size))
            y_pred = model(seq)
            loss = loss_fn(y_pred, labels)
            loss.backward()
            optimizer.step()
        
        if epoch % 25 == 1:
            print(f'Epoch: {epoch}, loss: {loss.item()}')
    print(f'Epoch: {epoch}, loss: {loss.item()}')