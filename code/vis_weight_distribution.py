import numpy as np
import matplotlib.pyplot as plt
from models.meta_inception1d import meta_inception1d, VCNN
import os
import torch

def plot_distribution(weights_clean, weights_noise, corruptionProb, save_path):
    """
    绘制直方图
    data:必选参数，绘图数据
    bins:直方图的长条形数目，可选项，默认为10
    normed:是否将得到的直方图向量归一化，可选项，默认为0，代表不归一化，显示频数。normed=1，表示归一化，显示频率。
    facecolor:长条形的颜色
    edgecolor:长条形边框的颜色
    alpha:透明度
    """
    plt.hist(weights_clean, bins=20, facecolor="red", edgecolor="red", alpha=1.0, rwidth=0.4,label="clean")
    plt.hist(weights_noise, bins=20, facecolor="blue", edgecolor="blue", alpha=1.0, rwidth=0.4, label="noise")

    # 显示横轴标签
    plt.xlabel("Weight")
    # 显示纵轴标签
    plt.ylabel("Numbers")
    # 显示图标题

    corruptionProb = int(corruptionProb * 100)
    plt.title("{} noise".format(corruptionProb))

    plt.legend(loc='upper center')
    plt.savefig(save_path)
    plt.show()

ratio = 0.3

loss_sequence_path = "/home/wxd/mnt/wyh/ecg_ptbxl/output/baseline_inception1d_superdiagnostic_NCAR_{}_1/models/your_inception1d/loss_squence.npy".format(ratio)
target_path = "/home/wxd/mnt/wyh/ecg_ptbxl/output/baseline_inception1d_superdiagnostic_NCAR_{}_1/models/your_inception1d/target.npy".format(ratio)

# 加载loss sequence
loss_seq = np.load(loss_sequence_path)
loss_seq = np.transpose(loss_seq, (1,2,0))

# 加载标签数据
target = np.load(target_path)

# 加载模型，输出训练数据对应的weight
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
subnet = VCNN(hidden1=100, output=5, k_size=5)

for epoch in [9, 10, 19, 29, 39]:
    model_path = "/home/wxd/mnt/wyh/ecg_ptbxl/output/meta_inception1d_superdiagnostic_NCAR_{}_1/models/meta_inception1d/vcnn-{}.path".format(ratio, epoch)
    checkpoint = torch.load(model_path)
    subnet.load_state_dict(checkpoint['state_dict'])
    subnet.to(device)

    num_batch = 500

    loss_seq_ = torch.from_numpy(loss_seq).to(device, dtype=torch.float32)
    batchs = np.linspace(0, loss_seq.shape[0], num_batch, dtype=int)

    sample_weights = []

    for i in range(len(batchs)-1):
        v_lambda = subnet(loss_seq_[batchs[i]:batchs[i+1]])
        sample_weights.append(v_lambda.detach().cpu().numpy())

    sample_weights = np.concatenate(sample_weights, axis=0)

    # 每一类分开观察
    for i in range(5):
        tmp = target[:, i] - target[:, i + 5]
        noise_ids = np.where(tmp != 0)[0]
        clean_ids = np.where(tmp == 0)[0]
        plot_distribution(sample_weights[clean_ids, i], sample_weights[noise_ids, i], ratio, save_path="weight_dist/{}_{}_{}.png".format(ratio, epoch, i))



