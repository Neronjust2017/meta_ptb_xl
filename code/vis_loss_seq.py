import numpy as np
import matplotlib.pyplot as plt

f = '/home/wxd/mnt/wyh/ecg_ptbxl/output/baseline_inception1d_superdiagnostic_NCAR_0.4_1/models/your_inception1d/loss_squence.npy'
loss_seq = np.load(f)

f = '/home/wxd/mnt/wyh/ecg_ptbxl/output/baseline_inception1d_superdiagnostic_NCAR_0.4_1/models/your_inception1d/target.npy'
target = np.load(f)

def plot(lsq, title, target, path):
    fig, ax = plt.subplots(figsize=(10, 10))

    plt.subplot(2, 3, 1)
    if target[0] == target[5]:
        c = 'b'
    else:
        c = 'r'
    plt.plot(lsq[:, 0], c=c, label='cla1: %d %d' % (target[0], target[5]))

    plt.subplot(2, 3, 2)
    if target[1] == target[6]:
        c = 'b'
    else:
        c = 'r'
    plt.plot(lsq[:, 1], c=c, label='cla2: %d %d' % (target[1], target[6]))

    plt.subplot(2, 3, 3)
    if target[2] == target[7]:
        c = 'b'
    else:
        c = 'r'
    plt.plot(lsq[:, 2], c=c, label='cla3: %d %d' % (target[2], target[7]))

    plt.subplot(2, 3, 4)
    if target[3] == target[8]:
        c = 'b'
    else:
        c = 'r'
    plt.plot(lsq[:, 3], c=c, label='cla4: %d %d' % (target[3], target[8]))

    plt.subplot(2, 3, 5)
    if target[4] == target[9]:
        c = 'b'
    else:
        c = 'r'
    plt.plot(lsq[:, 4], c=c, label='cla5: %d %d' % (target[4], target[9]))

    fig.legend()
    plt.title(title)
    plt.savefig(path)
    fig.show()

for i in range(len(loss_seq)):
    plot(loss_seq[:,i,:], i, target[i], path='/home/wxd/mnt/wyh/ecg_ptbxl/output/baseline_inception1d_superdiagnostic_NCAR_0.4_1/models/your_inception1d/plots/'+str(i))
