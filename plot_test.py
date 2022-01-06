import numpy as np
import matplotlib.pyplot as plt

# Filenames
training_file = "data-VOO.csv"
test_file = "data-AMD.csv"


# Initialize training data from data file
training_dataImport = np.loadtxt(training_file, delimiter=',', skiprows=1, usecols=(1,2,3,4,5))[::-1]
training_data = np.copy(training_dataImport)

# Initialize test data from data file
test_dataImport = np.loadtxt(test_file, delimiter=',', skiprows=1, usecols=(1,2,3,4,5))[::-1]
test_data = np.copy(test_dataImport)

print(test_data)

# Set up plot for the data
x_plot = np.arange(1, len(test_data[0]))
y_plot = np.array([test_data[0]])

print(x_plot)
print(y_plot)

plt.plot(x_plot, y_plot, label='AMD Data')