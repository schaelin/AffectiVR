from utils.test_util import show_test_result
import numpy as np

# K-Fold EER, FRR calc and print
EER = []
FRR = []
for i in range(5):
    path = '../result/proposed_resnet.npy'
    EER, FRR = show_test_result(path, True)

    EER.append(round(EER, 2))
    FRR.append(round(FRR, 2))

EER.sort()
FRR.sort()

print(EER, FRR)
print(np.mean(EER), np.mean(FRR))

# Show loss
import pickle
import matplotlib.pyplot as plt

path = '../result/history'
file = open(path, 'rb')
data = pickle.load(file)

plt.plot(data['loss'])
plt.plot(data['val_loss'])
plt.show()
