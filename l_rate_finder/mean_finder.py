import numpy as np

data = np.loadtxt("out/CNN.txt")

print(np.std(data))

data = np.loadtxt("out/LSTM.txt")

print(np.std(data))

test_file = ["data/AAPL.csv"]
seq_len = 21 # length of window


### Percentual deviation from target ###
losspercent = 0
plot_data = np.loadtxt(test_file[0], delimiter=',', skiprows=1, usecols=(4,5))[::-1]
targets = plot_data[seq_len:]
y_pred_plot = plot_data[seq_len-1:-1]
for i, e in enumerate(y_pred_plot):
    target1 = targets[i, 0]
    target2 = targets[i, 1]
    losspercent = losspercent + abs((e[0]-target1)/target1)
    losspercent = losspercent + abs((e[1]-target1)/target1)
losspercent = (losspercent/(len(y_pred_plot)*len(y_pred_plot[0])))*100
print(str(losspercent))