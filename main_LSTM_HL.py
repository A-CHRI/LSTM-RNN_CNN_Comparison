import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
training_files = ["data/TSLA.csv", "data/GME.csv", "data/VOO.csv", "data/AMD.csv"]
test_file = ["data/AAPL.csv"]

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
def print_and_log(string):
    print(string)
    with open("out/log_LSTM_HL.txt", "a") as f:
        f.write(string + "\n")

if __name__ == '__main__':
    # Clear the log
    with open("out/log_LSTM_HL.txt", "w") as f:
        f.write("")

    # Print the parameter info
    print_and_log(
        "-"*80 + "\n" + f"{'LSTM Model - High & Low':^80}" + "\n" + 
        "-"*80 + "\n" + f"{'Files:':<80}" + "\n" + "-"*80 + 
        f"\nTraining files: \n{training_files}" + "\n"
        f"\nTesting files: \n{str(test_file)}" +
        "\n" + "-"*80 + "\n" + f"{'Data info:':<40}{'Training info:':<40}" + "\n" + "-"*80 + 
        f"\n{'Features:':<30}{features:<10}{'Learning rate:':<30}{l_rate:<10}" +
        f"\n{'Sequence Length:':<30}{seq_len:<10}{'Epochs:':<30}{n_epoch:<10}" +
        f"\n{'Batch Size:':<30}{batch_size:<10}{'Device:':<30}{'CUDA' if torch.cuda.is_available() else 'CPU':<10}" + 
        f"\n{'Output:':<30}{n_output:<10}{'Hidden cells:':<30}{n_hidden:<10}" + "\n" + "-"*80
    )

    # Initialize model
    print_and_log("\nInitializing model...")
    timer_start = time.perf_counter()

    model = LSTM(features, n_hidden, n_output, n_layers=2).to(device)
    loss_fn = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)

    timer_end = time.perf_counter()
    print_and_log('Model initialized! (' + str(round(timer_end - timer_start, 4)) + ' seconds )')

    # Load the training dataset
    print_and_log("\nLoading training dataset...")
    timer_start = time.perf_counter()

    dataset_train = StockData(training_files)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    timer_end = time.perf_counter()
    print_and_log('Training dataset loaded! (' + str(round(timer_end - timer_start, 4)) + ' seconds )')

    # Training loop
    print_and_log("\nTraining...")
    timer_start = time.perf_counter()

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

            if (i+1) % 25 == 0:
                loss_sci = "{:.4e}".format(loss.item())
                print(f"{f'Epoch {epoch+1}/{n_epoch}':^20} | {f'Step {i+1}/{iterations}':^20} | {f'Loss: {loss_sci}':^20} \r", end="")

    timer_end = time.perf_counter()
    print_and_log('\nTraining finished! (' + str(round(timer_end - timer_start, 4)) + ' seconds )')

    # Load the test dataset
    print_and_log("\nLoading test dataset...")
    timer_start = time.perf_counter()

    dataset_test = StockData(test_file)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    y_pred_plot = np.array([])

    timer_end = time.perf_counter()
    print_and_log('Test dataset loaded! (' + str(round(timer_end - timer_start, 4)) + ' seconds )')

    # Test model
    print_and_log("\nTesting...")
    timer_start = time.perf_counter()

    for i, (x, y) in enumerate(dataloader_test):
        x = x.view(-1, seq_len, features)
        y = y.view(-1, n_output)

        # Forward pass
        y_pred = model(x)
        y_pred_plot = np.append(y_pred_plot, y_pred.cpu().detach().numpy())
        loss = loss_fn(y_pred, y)

        if (i+1) % 25 == 0:
            loss_sci = "{:.4e}".format(loss.item())
            print(f"{f'Epoch {epoch+1}/{n_epoch}':^20} | {f'Step {i+1}/{iterations}':^20} | {f'Loss: {loss_sci}':^20} \r", end="")

    timer_end = time.perf_counter()
    print_and_log('\nTesting finished! (' + str(round(timer_end - timer_start, 4)) + ' seconds )')

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
    print_and_log(f"\nThe mean percentual deviation from targets of this model is {losspercent}%")

    # Logs the last 30 days of the test dataset
    plot_data = np.loadtxt(test_file[0], delimiter=',', skiprows=1, usecols=(4,5))[::-1]
    plot_data_30 = plot_data[-30:]
    pred_data_30 = y_pred_plot[-30:]
    print_and_log("\nTest dataset last 30 days:" + "\n" + "-"*80)
    print_and_log(f"{'Actual high':<20}{'Actual Low':<20}{'Predicted high':<20}{'Predicted low':<20}")
    for i, e in enumerate(plot_data_30):
        print(f"{round(e[0],4):<20}{round(e[1],4):<20}{round(pred_data_30[i][0],4):<20}{round(pred_data_30[i][1],4):<20}")
    

    ### Plotting ###
    fig = plt.figure(figsize=(10, 5))
    sub = fig.subfigures(2, 1)
    (top_left, top_right) = sub[0].subplots(1, 2)
    bottom = sub[1].subplots(1, 1)

    # Set up the loss plot
    top_left.plot(Loss, label="Loss function")
    top_left.legend()
    top_left.grid(True)

    # Set up the zoomed plot
    top_right.plot(np.arange(len(plot_data[:, 0])), plot_data[:, 0], label='High & Low', color='darkgray')
    top_right.plot(np.arange(len(plot_data[:, 0])), plot_data[:, 1], color='darkgray')
    top_right.fill_between(np.arange(len(plot_data[:, 0])), plot_data[:, 0], plot_data[:, 1], color='darkgray', alpha=0.2)

    top_right.plot(np.arange(len(y_pred_plot[:, 0])) + seq_len, y_pred_plot[:, 0], label='Predicted High', linestyle='--', color='green')
    top_right.plot(np.arange(len(y_pred_plot[:, 0])) + seq_len, y_pred_plot[:, 1], label='Predicted Low', linestyle='--', color='crimson')

    top_right.axis(
        xmin=len(plot_data[:, 0]) - 30, xmax=len(plot_data[:, 0]),
        ymin=np.min(y_pred_plot[:, 1][-30:]) - 20, ymax=np.max(y_pred_plot[:, 0][-30:]) + 20
        ) # Zoom in on the last 30 days
    top_right.legend()
    top_right.grid(True)

    ### Bottom plot
    # Real data
    bottom.plot(np.arange(len(plot_data[:, 0])), plot_data[:, 0], label='High & Low', color='darkgray')
    bottom.plot(np.arange(len(plot_data[:, 0])), plot_data[:, 1], color='darkgray')
    bottom.fill_between(np.arange(len(plot_data[:, 0])), plot_data[:, 0], plot_data[:, 1], color='darkgray', alpha=0.2)

    # Prediction data
    bottom.plot(np.arange(len(y_pred_plot[:, 0])) + seq_len, y_pred_plot[:, 0], label='Predicted High', linestyle='--', color='green')
    bottom.plot(np.arange(len(y_pred_plot[:, 0])) + seq_len, y_pred_plot[:, 1], label='Predicted Low', linestyle='--', color='crimson')

    bottom.legend()
    bottom.grid(True)

    sub[0].suptitle('LSTM - High & Low', fontsize=16)
    top_left.set_title('Loss function')
    top_right.set_title('Predicted closing price (last 30 days)')
    bottom.set_title('Predicted closing price')

    plt.get_current_fig_manager().window.state('zoomed')
    plt.savefig("out/plot_LSTM_HL.png")
    plt.show()