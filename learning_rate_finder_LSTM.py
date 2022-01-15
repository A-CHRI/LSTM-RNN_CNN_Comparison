import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

### Device configuration ###
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Parameters ###
features = 5 # Close, Volume, Open, High, Low (Input_size = 5)
seq_len = 7 # look back period
batch_size = 64 # Must be a power of 2
l_rate = 0.00005
n_epoch = 512 # Must be divisible by 8
n_hidden = 24 # 2/3 input neurons

n_input = features * seq_len
n_output = 1

### Training and test files ###
training_files = ["data/AMZN.csv", "data/BRK.csv", "data/FB.csv", "data/GOOG.csv","data/GOOGL.csv","data/JPM.csv","data/MSFT.csv","data/NVDA.csv","data/TSLA.csv","data/VOO.csv"]
test_file = ["data/data-AAPL.csv"]
### Parameters ###
output_neurons = 1
hidden_layers = 2 # No theoreticle reason to be more than 2
learning_rate = 0.00001
iterations = 10 #500000
days_per_segment = 7 # Usually 21 for stable stocks, and lower the more volatile
input_neurons = days_per_segment * 5
hidden_neurons = int((2 / 3) * input_neurons) # Usually 2/3 the size of the input neurons


device = 'cuda' if torch.cuda.is_available() else 'cpu'

### Neural Network ###
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
        input = input.view(-1, seq_len, features)
        out, _ = self.lstm(input, (h_0, c_0))

        # Linear layer
        out = out.reshape(len(input), -1)
        out = self.linear(out)
        return out

class StockData(Dataset):
    def __init__(self, data_files):
        # Load data
        x_arr = []
        y_arr = []
        for i in data_files:
            data = np.loadtxt(i, delimiter=',', skiprows=1, usecols=(1,2,3,4,5))[::-1]

            for i in range(len(data) - (seq_len + 1)):
                x = data[i:i + seq_len, :]
                y = data[i + seq_len, 0]

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
        data_scaled_y = np.reshape(data_scaled_y, (-1, 1))
        
        # Set the tensors
        self.x = torch.tensor(data_scaled_x, dtype=torch.float).to(device)
        self.y = torch.tensor(data_scaled_y, dtype=torch.float).to(device)

        self.n_samples = self.x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

### Print and log ###
def print_and_log(string):
    print(string)
    with open("learning_rate_data_LSTM.txt", "a") as f:
        f.write(string + "\n")



if __name__ == '__main__':
    # Clear the log
    with open("learning_rate_data_LSTM.txt", "w") as f:
        f.write("")
    for a in range(25):
        for j in range(5):
            learning_rate = 0.01/(2**a)


            model = LSTM(features, n_hidden, n_output, n_layers=2).to(device)
            loss_fn = nn.MSELoss(reduction='sum')
            optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)


            dataset_train = StockData(training_files)
            dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)


            Loss = []

            samples = len(dataset_train)
            iterations = samples // batch_size
            for epoch in range(n_epoch):
                for i, (x, y) in enumerate(dataloader_train):
                    x = x.view(-1, seq_len, features)
                    y = y.view(-1, 1)

                    # Forward pass
                    y_pred = model(x)
                    loss = loss_fn(y_pred, y)
                    Loss.append(loss.item())

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if (i+1) % 25 == 0:
                        loss_sci = "{:.4e}".format(loss.item())
                        print(f"{f'Epoch {epoch+1}/{n_epoch}':^20} | {f'Step {i+1}/{iterations}':^20} | {f'Loss: {loss_sci}':^20} \r", end="")


            dataset_test = StockData(test_file)
            dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
            y_pred_plot = np.array([])


            for i, (x, y) in enumerate(dataloader_test):
                x = x.view(-1, seq_len, features)
                y = y.view(-1, 1)

                # Forward pass
                y_pred = model(x)
                y_pred_plot = np.append(y_pred_plot, y_pred.cpu().detach().numpy())
                loss = loss_fn(y_pred, y)

                if (i+1) % 25 == 0:
                    loss_sci = "{:.4e}".format(loss.item())
                    print(f"{f'Epoch {epoch+1}/{n_epoch}':^20} | {f'Step {i+1}/{iterations}':^20} | {f'Loss: {loss_sci}':^20} \r", end="")

            losspercent = 0
            # Prediction data
            y_pred_plot = dataset_test.scaler.inverse_transform(y_pred_plot.reshape(-1, 1))
            plot_data = np.loadtxt(test_file[0], delimiter=',', skiprows=1, usecols=(1))[::-1]
            targets = plot_data[seq_len:]
            for i, e in enumerate(y_pred_plot):
                target = targets[i]
                losspercent = losspercent + abs((e-target)/target)
            losspercent = (losspercent/len(y_pred_plot))*100
            print_and_log(str(round(losspercent[0], 4)) + " " + str(learning_rate))




