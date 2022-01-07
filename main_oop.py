import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

### Parameters ###
output_neurons = 1
hidden_layers = 2 # No theoreticle reason to be more than 2
learning_rate = 0.00001
iterations = 10 #500000
days_per_segment = 7 # Usually 21 for stable stocks, and lower the more volatile
input_neurons = days_per_segment * 5
hidden_neurons = int((2 / 3) * input_neurons) # Usually 2/3 the size of the input neurons

training_files = ["data-AAPL-small.csv", "data-TSLA-small.csv", "data-VOO-small.csv", "data-AMD-small.csv"]
test_files = ["data-GME.csv"]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

### Neural Network ###
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_neurons, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, output_neurons),
        ).double()

    def forward(self, x):
        return self.linear_relu_stack(x)

### Methods ###
# Print and log
log_file = open("log.txt", "w")
def print_and_log(string):
    print(string)
    log_file.write(string + "\n")

# Import data
def import_data(filenames, days_per_segment):
    print_and_log("Importing data...")
    set = []
    for i in filenames:
        dataImport = np.loadtxt(i, delimiter=',', skiprows=1, usecols=(1, 2, 3, 4, 5))[::-1]
        data = np.copy(np.transpose(dataImport))
        print_and_log("File " + str(i) + " imported.\nNormalizing...")
        timer_start = time.perf_counter()

        for l in range (len(data[0]) - (days_per_segment + 1)):
        # Calculate x segment
            x = np.zeros(5 * days_per_segment)
            for j in range(5):
                for k in range(days_per_segment):
                    x[j * days_per_segment + k] = (data[j, k + l] - np.mean(data[j,:])) / np.std(data[j,:])
            # Calculate y segment
            y = (data[0, l + days_per_segment] - np.mean(data[0, :])) / np.std(data[0,:])
            set.append([x, y])
        timer_end = time.perf_counter()
        print_and_log("Done! (" + str(round(timer_end - timer_start, 4)) + " seconds).")
    return data, set

# Train the network
def Train_network(iterations, device, segments, model, loss_fn, optimizer, Loss):
    print_and_log("\nTraining the network... (" + str(len(training_sets)) + " segments to train on)")
    timer_start = time.perf_counter()

    segments_count = len(segments)
    for i in range(segments_count):
        # Initialize the tensors
        x = segments[i][0]
        inp = torch.tensor(x, device=device).double()

        y = [segments[i][1]]
        outp = torch.tensor(y, device=device).double()

        for j in range(iterations):
            # Forward pass
            y_pred = model(inp)

            # Compute and print loss
            loss = loss_fn(y_pred, outp)
            print(f'Training: {int((100/segments_count) * i)}%, Segment: {i}, Iteration:{j}, Loss: {loss.item()}')
            Loss[j*(i+1)] = loss.item()

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    timer_end = time.perf_counter()
    print_and_log("Done! (" + str(round(timer_end - timer_start, 4)) + " seconds).")

# Test the network
def Test_network(device, segments, model, loss_fn):
    print_and_log("\nTesting the network... (" + str(len(test_sets)) + " segments to test)")
    timer_start = time.perf_counter()

    segments_count = len(segments)
    predtest = np.zeros(len(segments))
    y_plot_pred = np.zeros(segments_count)
    for i in range(segments_count):
        # Initialize the tensors
        x = segments[i][0]
        inp = torch.tensor(x, device=device).double()

        y = segments[i][1]
        outp = torch.tensor(y, device=device).double()

        y_pred = model(inp)
        y_plot_pred[i] = y_pred.cpu().detach().numpy() * np.std(test_data[0, :]) + np.mean(test_data[0, :])

        # Compute and print loss
        loss = loss_fn(y_pred, outp)
        print(f'Testing: {int((100/segments_count) * i)}%, Segment: {i}, Loss: {loss.item()}')
        predtest[i] = loss.item()

    print_and_log("Calculating loss percentage...")
    losspercent = 0
    for i, e in enumerate(predtest):
        losspercent = losspercent + (test_sets[i][1]-e)/test_sets[i][1]
    losspercent = losspercent/len(predtest)
    print_and_log("Done! Loss percentage: " + str(round(losspercent, 4)) + "%")

    timer_end = time.perf_counter()
    print_and_log("Done! (" + str(round(timer_end - timer_start, 4)) + " seconds).")

    return y_plot_pred

# Plotting
def Plot(Loss, y_plot_pred, test_sets):
    # Set up plot for the data
    fig, (loss_plot, network_plot) = plt.subplots(2, 1)
    fig.suptitle('Loss and Network Output')

    # Set up the loss plot
    loss_plot.set_ylabel("Loss function")
    loss_plot.plot(Loss, label="Loss function")
    loss_plot.legend()
    loss_plot.grid(True)

    # set up the test plot
    network_plot.set_ylabel("Neural network test")
    x_plot_test = np.arange(len(test_data[0]))
    y_plot_test = np.array(test_data[0])
    network_plot.plot(x_plot_test, y_plot_test, label='Test files daily closing price')

    # Prediction data
    x_plot_pred = np.arange(len(test_sets)) + days_per_segment
    network_plot.plot(x_plot_pred, y_plot_pred, label='Prediction')

    # Plotting
    network_plot.grid(True)
    network_plot.legend()
    plt.show()


if __name__ == '__main__':
    # Print the parameter info
    print_and_log("\nTraining on: " + str(training_files) + "\nTesting on: " + str(test_files) + "\nLogging to: log.txt")
    print_and_log("\nInput neurons: " + str(input_neurons) + "\nOutput neurons: " + str(output_neurons) + "\nHidden layers: " + str(hidden_layers) + "\nHidden neurons: " + str(hidden_neurons))
    print_and_log("\nLearning rate: " + str(learning_rate) + "\nIterations: " + str(iterations) + "\nDays per segment: " + str(days_per_segment))
    print_and_log("\nDevice: " + str("CUDA" if torch.cuda.is_available() else "CPU"))
    input("\nPress enter to continue...")

    # Import data
    training_data, training_sets = import_data(training_files, days_per_segment)
    test_data, test_sets = import_data(test_files, days_per_segment)

    # Create the neural network
    print_and_log("\nInitializing network...")
    timer_start = time.perf_counter()

    model = NeuralNetwork().to(device)
    loss_fn = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print_and_log("Model: " + str(model))
    Loss = np.zeros(iterations * len(training_sets))

    timer_end = time.perf_counter()
    print_and_log("Done! (" + str(round(timer_end - timer_start, 4)) + " seconds).")

    # Train the neural network
    Train_network(iterations, device, training_sets, model, loss_fn, optimizer, Loss)

    # Test the neural network
    y_plot_pred = Test_network(device, test_sets, model, loss_fn)

    # Flush the log file
    log_file.flush()

    # Plot the data
    Plot(Loss, y_plot_pred, test_sets)