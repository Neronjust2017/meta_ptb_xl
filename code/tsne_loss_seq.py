import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

ratios = [0.2, 0.4, 0.6, 0.8]

for ratio in ratios:
    lr = 0.01

    f = '/home/wxd/mnt/wyh/ecg_ptbxl/output/lr_{}/baseline_inception1d_superdiagnostic_NCAR_{}_1/models/your_inception1d/loss_squence.npy'.format(lr, ratio)
    loss_seq = np.load(f)

    f = '/home/wxd/mnt/wyh/ecg_ptbxl/output/lr_{}/baseline_inception1d_superdiagnostic_NCAR_{}_1/models/your_inception1d/target.npy'.format(lr, ratio)
    target = np.load(f)


    for i in range(5):

        label_of_noise = np.zeros(len(target))
        tmp = target[:,i] - target[:, i+5]
        false_indexes = np.where(tmp != 0)
        label_of_noise[false_indexes] = 1

        X_tsne = TSNE(n_components=2, random_state=33).fit_transform(np.swapaxes(loss_seq[:,:,i], 0, 1))
        plt.figure(figsize=(10, 10))
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=label_of_noise, s=4, label="class".format(i))
        plt.legend()
        plt.savefig('lr={}_{}_{}.png'.format(lr, ratio,i))
        plt.show()
        plt.close()