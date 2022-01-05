import numpy as np
import torch
import matplotlib.pyplot as plt

# Initial parameters
max_input_neurons = 7 #72
max_output_neurons = 2
max_hidden_layers = 2
max_hidden_neurons = 16 #60
max_training_sets = 400
learning_rate = 0.01
max_rounds = 5000 #00


# Initialize the tensors from data file
data = np.loadtxt('data-VOO.csv', delimiter=',')
fulldata = data.copy()

trainingsets = np.array([])

for i in max_training_sets:
    x = data[4, 0:max_input_neurons]
    y = [1 if data[4, max_input_neurons + 1] > x[-1] else 0, 1 if data[4, max_input_neurons + 1] < x[-1] else 0]
    trainingsets = np.append(trainingsets, np.array([x, y]))

    for i in max_input_neurons:
        data = np.delete(data, (0), axis=0)



