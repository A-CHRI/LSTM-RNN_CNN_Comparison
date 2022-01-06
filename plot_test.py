import numpy as np
import matplotlib.pyplot as plt

# Filenames
training_file = "data-VOO.csv"
test_file = "data-AMD.csv"


# Initialize training data from data file
training_dataImport = np.loadtxt(training_file, delimiter=',', skiprows=1, usecols=(1,2,3,4,5))[::-1]
training_data = np.copy(np.transpose(training_dataImport))

# Initialize test data from data file
test_dataImport = np.loadtxt(test_file, delimiter=',', skiprows=1, usecols=(1,2,3,4,5))[::-1]
test_data = np.copy(np.transpose(test_dataImport))

## Set up plot for the data
# AMD data
x_plot_test = np.arange(len(test_data[0]))
y_plot_test = np.array(test_data[0])
plt.plot(x_plot_test, y_plot_test, label='AMD daily close price')


# Plotting
plt.grid(True)
plt.legend()
plt.show()