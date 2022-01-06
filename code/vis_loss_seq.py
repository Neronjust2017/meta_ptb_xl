import numpy as np
import matplotlib.pyplot as plt

f = '/home/weiyuhua/ecg_ptbxl/output/baseline_inception1d_super/models/your_inception1d/loss_squence.npy'
loss_seq = np.load(f)

f = '/home/weiyuhua/ecg_ptbxl/output/baseline_inception1d_super/models/your_inception1d/target.npy'
target = np.load(f)

def plot(lsq, title, target, path='/home/weiyuhua/ecg_ptbxl/output/baseline_inception1d_super/models/your_inception1d/'):
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplot(2, 3, 1)
    plt.plot(lsq[:, 0], c='r', label='cla1: {}'.format(target[0]))
    plt.subplot(2, 3, 2)
    plt.plot(lsq[:, 1], c='r', label='cla2: {}'.format(target[1]))
    plt.subplot(2, 3, 3)
    plt.plot(lsq[:, 2], c='r', label='cla3: {}'.format(target[2]))
    plt.subplot(2, 3, 4)
    plt.plot(lsq[:, 3], c='r', label='cla4: {}'.format(target[3]))
    plt.subplot(2, 3, 5)
    plt.plot(lsq[:, 4], c='r', label='cla5: {}'.format(target[4]))

    fig.legend()
    plt.title(title)
    plt.savefig(path)
    fig.show()

for i in range(len(loss_seq)):
    plot(loss_seq[:,i,:], i, target[i], path='/home/weiyuhua/ecg_ptbxl/output/baseline_inception1d_super/models/your_inception1d/plots/'+str(i))
