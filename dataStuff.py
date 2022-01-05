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
data = np.loadtxt('data-VOO.csv', delimiter=',', skiprows=1, usecols= (1,2,3,4,5) )
fulldata = data.copy()

trainingsets = []
print(data)

while len(trainingsets) < max_training_sets and (len(trainingsets) + 1)*(max_input_neurons) < len(data):
    x = data[max_input_neurons*len(trainingsets):max_input_neurons*(len(trainingsets)+1), 0]
    y = np.array([1 if data[max_input_neurons*(len(trainingsets)+1) + 1][0] > x[-1] else 0, 1 if data[max_input_neurons*(len(trainingsets)+1) + 1][0] < x[-1] else 0])
    trainingsets.append([x, y])

print(trainingsets)
