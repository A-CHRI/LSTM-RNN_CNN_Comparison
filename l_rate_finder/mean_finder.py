import numpy as np

data = np.loadtxt("out/experiment_data_CNN.txt")

print(np.std(data))

data = np.loadtxt("out/experiment_data_LSTM.txt")

print(np.std(data))