import numpy as np

data = np.loadtxt("learning_rate/learning_rate_data_LSTM.txt", delimiter=' ')
dic = {}
for i in range(int(len(data)/5)):
    mean = np.mean(data[i*5:i*5+5, 0])
    lr = data[i*5, 1]
    print(f"Learning rate is {lr}, mean error is {mean}%")
    dic[lr] = mean
mini = min(dic, key=dic.get)
print(mini)
print(dic[mini])