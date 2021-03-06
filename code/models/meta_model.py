import numpy as np
from models.timeseries_utils import *

from pathlib import Path
from functools import partial


from models.meta_inception1d import meta_inception1d, VCNN
from models.basic_conv1d import fcn,fcn_wang,schirrmeister,sen,basic1d,weight_init

import os
import math

from models.base_model import ClassificationModel
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from fastai.torch_core import to_np
from torch.utils.data import TensorDataset

#for lrfind
import matplotlib
import matplotlib.pyplot as plt

from utils.utils import apply_thresholds, challenge_metrics, roc_auc_score

#eval for early stopping
from utils.utils import evaluate_experiment

import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class YourModel(ClassificationModel):
#     def __init__(self, name, n_classes,  sampling_frequency, outputfolder, input_shape):
#         self.name = name
#         self.n_classes = n_classes
#         self.sampling_frequency = sampling_frequency
#         self.outputfolder = outputfolder
#         self.input_shape = input_shape
#
#     def fit(self, X_train, y_train, X_val, y_val):
#         pass
#
#     def predict(self, X):
#         pass

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


def evaluate_experiment(y_true, y_pred, thresholds=None):
    results = {}

    if not thresholds is None:
        # binary predictions
        y_pred_binary = apply_thresholds(y_pred, thresholds)
        # PhysioNet/CinC Challenges metrics
        challenge_scores = challenge_metrics(y_true, y_pred_binary, beta1=2, beta2=2)
        results['F_beta_macro'] = challenge_scores['F_beta_macro']
        results['G_beta_macro'] = challenge_scores['G_beta_macro']

    # label based metric
    results['macro_auc'] = roc_auc_score(y_true, y_pred, average='macro')

    # df_result = pd.DataFrame(results, index=[0])
    # return df_result
    return results

def progress(data_loader, batch_idx):
    base = '[{}/{} ({:.0f}%)]'
    if hasattr(data_loader, 'n_samples'):
        current = batch_idx * data_loader.batch_size
        total = data_loader.n_samples
    else:
        current = batch_idx
        total = len(data_loader)
    return base.format(current, total, 100.0 * current / total)

def save_checkpoint(model, optimizer, checkpoint_dir, name="model.pth"):
    arch = type(model).__name__
    state = {
        'arch': arch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    path = os.path.join(checkpoint_dir, name)

    torch.save(state, path)

def load_checkpoint(model, resume_path):
    """
    Resume from saved checkpoints

    :param resume_path: Checkpoint path to be resumed
    """

    path = os.path.join(resume_path, 'model.pth')
    checkpoint = torch.load(path)
    # load architecture params from checkpoint.

    # print(path)
    # print(checkpoint['state_dict'])

    model.load_state_dict(checkpoint['state_dict'])

    return model


def fmax_metric(targs, preds):
    return evaluate_experiment(targs, preds)["Fmax"]


def auc_metric(targs, preds):
    return evaluate_experiment(targs, preds)["macro_auc"]


def mse_flat(preds, targs):
    return torch.mean(torch.pow(preds.view(-1) - targs.view(-1), 2))


def nll_regression(preds, targs):
    # preds: bs, 2
    # targs: bs, 1
    preds_mean = preds[:, 0]
    # warning: output goes through exponential map to ensure positivity
    preds_var = torch.clamp(torch.exp(preds[:, 1]), 1e-4, 1e10)
    # print(to_np(preds_mean)[0],to_np(targs)[0,0],to_np(torch.sqrt(preds_var))[0])
    return torch.mean(torch.log(2 * math.pi * preds_var) / 2) + torch.mean(
        torch.pow(preds_mean - targs[:, 0], 2) / 2 / preds_var)


def nll_regression_init(m):
    assert (isinstance(m, nn.Linear))
    nn.init.normal_(m.weight, 0., 0.001)
    nn.init.constant_(m.bias, 4)

class MetaModel(ClassificationModel):
    def __init__(self, name, n_classes, freq, outputfolder, input_shape, pretrained=False, input_size=2.5,
                 input_channels=12, chunkify_train=False, chunkify_valid=False, bs=128, ps_head=0.5, lin_ftrs_head=[128],
                 wd=1e-2, epochs=50, lr=1e-2, kernel_size=5, loss="binary_cross_entropy", pretrainedfolder=None,
                 n_classes_pretrained=None, gradual_unfreezing=True, discriminative_lrs=True, epochs_finetuning=30,
                 early_stopping=None, aggregate_fn="max", concat_train_val=False):
        super().__init__()

        self.name = name
        self.num_classes = n_classes if loss != "nll_regression" else 2
        self.target_fs = freq
        self.outputfolder = Path(outputfolder)

        self.input_size = int(input_size * self.target_fs)
        self.input_channels = input_channels

        self.chunkify_train = chunkify_train
        self.chunkify_valid = chunkify_valid

        self.chunk_length_train = 2 * self.input_size  # target_fs*6
        # self.chunk_length_valid = self.input_size
        self.chunk_length_valid = 2 * self.input_size

        self.min_chunk_length = self.input_size  # chunk_length

        self.stride_length_train = self.input_size  # chunk_length_train//8
        # self.stride_length_valid = self.input_size // 2  # chunk_length_valid
        self.stride_length_valid = self.input_size

        self.copies_valid = 0  # >0 should only be used with chunkify_valid=False

        self.bs = bs
        self.ps_head = ps_head
        self.lin_ftrs_head = lin_ftrs_head
        self.wd = wd
        self.epochs = epochs
        self.lr = lr
        self.kernel_size = kernel_size
        self.loss = loss
        self.input_shape = input_shape

        if pretrained == True:
            if (pretrainedfolder is None):
                pretrainedfolder = Path('../output/exp0/models/' + name.split("_pretrained")[0] + '/')
            if (n_classes_pretrained is None):
                n_classes_pretrained = 71

        self.pretrainedfolder = None if pretrainedfolder is None else Path(pretrainedfolder)
        self.n_classes_pretrained = n_classes_pretrained
        self.discriminative_lrs = discriminative_lrs
        self.gradual_unfreezing = gradual_unfreezing
        self.epochs_finetuning = epochs_finetuning

        self.early_stopping = early_stopping
        self.aggregate_fn = aggregate_fn
        self.concat_train_val = concat_train_val

    def fit(self, X_train, y_train, X_val, y_val, X_test, y_test):
        # convert everything to float32
        X_train = [l.astype(np.float32) for l in X_train]
        X_val = [l.astype(np.float32) for l in X_val]
        X_test = [l.astype(np.float32) for l in X_test]
        y_train = [l.astype(np.float32) for l in y_train]
        y_val = [l.astype(np.float32) for l in y_val]
        y_test = [l.astype(np.float32) for l in y_test]

        if (self.concat_train_val):
            X_train += X_val
            y_train += y_val

        data_loader_train, data_loader_valid, data_loader_test, loss_fn, model, vcnn = self._get_learner(X_train, y_train, X_val, y_val, X_test, y_test)

        optimizer = Adam(model.params(), lr=self.lr)
        optimizer_subnet = Adam(vcnn.params(), lr=1e-3)

        # lr_scheduler = StepLR(optimizer, step_size=int(self.epochs*0.8), gamma=0.1)
        # lr_scheduler = StepLR(optimizer, step_size=int(self.epochs*0.8), gamma=0.1)
        lr_scheduler = CosineAnnealingLR(optimizer, self.lr)

        model.to(device)
        model.apply(weight_init)

        vcnn.to(device)
        vcnn.apply(weight_init)

        # training

        # pynvml.nvmlInit()
        # handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 0表示第一块显卡
        # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

        train_loss_squence = np.load(str(self.outputfolder).replace('meta', 'baseline', 1).replace('meta', 'your') + '/loss_squence.npy')
        # epochs, n, 71
        train_loss_squence = np.transpose(train_loss_squence, (1,2,0))
        train_loss_squence = torch.from_numpy(train_loss_squence)
        train_loss_squence = train_loss_squence.to(device, dtype=torch.float)
        train_sample_weights = np.zeros(shape=(self.epochs, train_loss_squence.shape[0], self.num_classes))

        meta_loader_iter = iter(data_loader_valid)

        for epoch in range(self.epochs):

            for batch_idx, (data, target, id) in enumerate(data_loader_train):
                data, target = data.to(device), target.to(device)
                target = target[:, :self.num_classes]

                model.train()
                meta_model = meta_inception1d(num_classes=self.num_classes, input_channels=self.input_channels, use_residual=True,
                                ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head,
                                kernel_size=8 * self.kernel_size).to(device)
                meta_model.load_state_dict(model.state_dict())

                output = meta_model(data)
                loss = loss_fn(output, target, reduce=False)

                vcnn.train()
                v_lambda = vcnn(train_loss_squence[id])
                l_f_meta = torch.sum(loss * v_lambda)/len(loss)

                meta_model.zero_grad()
                grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)

                meta_model.update_params(lr_inner=self.lr, source_params=grads)
                del grads

                # meta data
                try:
                    data_val, target_val, _ = next(meta_loader_iter)
                except StopIteration:
                    meta_loader_iter = iter(data_loader_valid)
                    data_val, target_val, _ = next(meta_loader_iter)

                data_val, target_val = data_val.to(device), target_val.to(device)

                # print("\n")
                # print("train loader", len(data_loader_train), batch_idx)
                # print("valid loader", len(data_loader_valid))
                # print(data_val.shape)
                #
                # print(meminfo.total / 1024 ** 2 )  # 第二块显卡总的显存大小
                # print(meminfo.used / 1024 ** 2 )  # 这里是字节bytes，所以要想得到以兆M为单位就需要除以1024**2
                # print("\n")
                if batch_idx % 10 == 0:
                    print("Train Epoch: {} {}/{}".format(epoch, batch_idx, len(data_loader_train)))

                y_g_hat = meta_model(data_val)
                l_g_meta = loss_fn(y_g_hat, target_val)

                optimizer_subnet.zero_grad()
                l_g_meta.backward()
                optimizer_subnet.step()

                output = model(data)
                loss = loss_fn(output, target, reduce=False)

                with torch.no_grad():
                    w_new = vcnn(train_loss_squence[id])
                    # print(w_new)
                loss = torch.sum(loss * w_new)/len(loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            lr_scheduler.step(epoch)

            # save model
            save_checkpoint(model, optimizer, self.outputfolder, name="model-{}.path".format(epoch))
            save_checkpoint(vcnn, optimizer_subnet, self.outputfolder, name="vcnn-{}.path".format(epoch))

            # weight distribution
            sample_weights = []
            targets = []
            for batch_idx, (data, target, id) in enumerate(data_loader_train):
                v_lambda = vcnn(train_loss_squence[id])
                sample_weights.append(v_lambda.detach().cpu().numpy())
                targets.append(target.detach().cpu().numpy())

            sample_weights = np.concatenate(sample_weights, axis=0)
            targets = np.concatenate(targets, axis=0)

            train_sample_weights[epoch] = sample_weights

            print(train_sample_weights[:epoch, :100])

            # 每一类分开观察
            for i in range(5):
                tmp = targets[:, i] - targets[:, i + 5]
                noise_ids = np.where(tmp != 0)[0]
                clean_ids = np.where(tmp == 0)[0]
                plot_distribution(sample_weights[clean_ids, i], sample_weights[noise_ids, i], self.noise_ratio,
                                  save_path=os.path.join(self.outputfolder, "{}_{}_{}.png".format(self.noise_ratio, epoch, i)))

            # validation
            train_loss = 0
            valid_loss = 0
            test_loss = 0

            with torch.no_grad():

                train_target = []
                train_pred = []
                for batch_idx, (data, target, id) in enumerate(data_loader_train):
                    data, target = data.to(device), target.to(device)
                    target = target[:, :self.num_classes]

                    output = model(data)

                    if (self.loss == "binary_cross_entropy"):
                        output = nn.Sigmoid()(output)

                    loss = loss_fn(output, target, reduce=False)
                    train_loss += loss.detach().cpu().mean()
                    train_target.append(target.detach().cpu())
                    train_pred.append(output.detach().cpu())

                train_target = np.concatenate(train_target, axis=0)
                train_pred = np.concatenate(train_pred, axis=0)
                train_metric = evaluate_experiment(train_target, train_pred, thresholds=None)

                valid_target = []
                valid_pred = []
                for batch_idx, (data, target, _) in enumerate(data_loader_valid):
                    data, target = data.to(device), target.to(device)
                    output = model(data)

                    if (self.loss == "binary_cross_entropy"):
                        output = nn.Sigmoid()(output)

                    loss = loss_fn(output, target)
                    valid_loss += loss.detach().cpu()

                    valid_target.append(target.detach().cpu())
                    valid_pred.append(output.detach().cpu())

                valid_target = np.concatenate(valid_target, axis=0)
                valid_pred = np.concatenate(valid_pred, axis=0)
                valid_metric = evaluate_experiment(valid_target, valid_pred, thresholds=None)

                test_target = []
                test_pred = []
                for batch_idx, (data, target, _) in enumerate(data_loader_test):
                    data, target = data.to(device), target.to(device)
                    output = model(data)

                    if (self.loss == "binary_cross_entropy"):
                        output = nn.Sigmoid()(output)

                    loss = loss_fn(output, target)
                    test_loss += loss.detach().cpu()

                    test_target.append(target.detach().cpu())
                    test_pred.append(output.detach().cpu())

                test_target = np.concatenate(test_target, axis=0)
                test_pred = np.concatenate(test_pred, axis=0)
                test_metric = evaluate_experiment(test_target, test_pred, thresholds=None)

                print(
                    'Train Epoch: {} Train Loss: {:.6f} Valid Loss: {:.6f} Test Loss: {:.6f} Train Metric: {:.6f} Valid Metric: {:.6f} Test Metric: {:.6f}'.
                    format(epoch, train_loss / len(data_loader_train), valid_loss / len(data_loader_valid),
                           test_loss / len(data_loader_test),
                           train_metric['macro_auc'], valid_metric['macro_auc'], test_metric['macro_auc']))

                wandb.log({"train loss": train_loss / len(data_loader_train)})
                wandb.log({"valid loss": valid_loss / len(data_loader_valid)})
                wandb.log({"test loss": test_loss / len(data_loader_test)})

                wandb.log({"train metric": train_metric['macro_auc']})
                wandb.log({"valid metric": valid_metric['macro_auc']})
                wandb.log({"test metric": test_metric['macro_auc']})

        # save
        save_checkpoint(model, optimizer, self.outputfolder)
        np.save(os.path.join(self.outputfolder, 'train_sample_weights.npy'), train_sample_weights)


    def predict(self, X, Y):
        X = [l.astype(np.float32) for l in X]
        Y = [l.astype(np.float32) for l in Y]
        # y_dummy = [np.ones(self.num_classes, dtype=np.float32) for _ in range(len(X))]

        data_loader, _, _, loss_fn, model, _ = self._get_learner(X, Y, X, Y, X, Y)

        # print(model.parameters())

        # print(self.outputfolder)

        model = load_checkpoint(model, self.outputfolder)
        model.to(device)

        # print("@@@@@@@@@@@@@@@@@@@")
        # print(model.parameters())

        data_loader.__setattr__("shuffle", False)

        # 直接在这里evaluate

        targets = []
        outputs = []
        with torch.no_grad():
            for batch_idx, (data, target, _) in enumerate(data_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                # print(output[0])
                if(self.loss == "binary_cross_entropy"):
                    output = nn.Sigmoid()(output)
                    # print(output[0])
                else:
                    exit(1)
                targets.append(target.detach().cpu())
                outputs.append(output.detach().cpu())

        targets = np.concatenate(targets, axis=0)
        preds = np.concatenate(outputs, axis=0)
        metric = evaluate_experiment(targets, preds, thresholds=None)

        return metric

        # idmap = data_loader.dataset.get_id_mapping()
        #
        # return aggregate_predictions(preds, idmap=idmap,
        #                              aggregate_fn=np.mean if self.aggregate_fn == "mean" else np.amax)

    def _get_learner(self, X_train, y_train, X_val, y_val, X_test, y_test, num_classes=None):

        df_train = pd.DataFrame({"data": range(len(X_train)), "label": y_train})
        df_valid = pd.DataFrame({"data": range(len(X_val)), "label": y_val})
        df_test = pd.DataFrame({"data": range(len(X_test)), "label": y_test})

        # timeseries.ToTensor()
        tfms_ptb_xl = [ToTensor()]

        # TimeseriesDatasetCrops Dataset
        ds_train = TimeseriesDatasetCrops(df_train, self.input_size, num_classes=self.num_classes,
                                          chunk_length=self.chunk_length_train if self.chunkify_train else 0,
                                          min_chunk_length=self.min_chunk_length, stride=self.stride_length_train,
                                          transforms=tfms_ptb_xl, annotation=False, col_lbl="label", npy_data=X_train)
        ds_valid = TimeseriesDatasetCrops(df_valid, self.input_size, num_classes=self.num_classes,
                                          chunk_length=self.chunk_length_valid if self.chunkify_valid else 0,
                                          min_chunk_length=self.min_chunk_length, stride=self.stride_length_valid,
                                          transforms=tfms_ptb_xl, annotation=False, col_lbl="label", npy_data=X_val)

        ds_test = TimeseriesDatasetCrops(df_test, self.input_size, num_classes=self.num_classes,
                                          chunk_length=self.chunk_length_valid if self.chunkify_valid else 0,
                                          min_chunk_length=self.min_chunk_length, stride=self.stride_length_valid,
                                          transforms=tfms_ptb_xl, annotation=False, col_lbl="label", npy_data=X_test)

        # data loader
        data_loader_train = DataLoader(ds_train, batch_size=self.bs, shuffle=True, pin_memory=True, num_workers=32)
        data_loader_valid = DataLoader(ds_valid, batch_size=self.bs, shuffle=False, pin_memory=True, num_workers=32)
        data_loader_test = DataLoader(ds_test, batch_size=self.bs, shuffle=False, pin_memory=True, num_workers=32)
        
        # loss
        if (self.loss == "binary_cross_entropy"):
            loss = F.binary_cross_entropy_with_logits
        elif (self.loss == "cross_entropy"):
            loss = F.cross_entropy
        elif (self.loss == "mse"):
            loss = mse_flat
        elif (self.loss == "nll_regression"):
            loss = nll_regression
        else:
            print("loss not found")
            assert (True)

        self.input_channels = self.input_shape[-1]

        print("model:", self.name)  # note: all models of a particular kind share the same prefix but potentially a different postfix such as _input256
        num_classes = self.num_classes if num_classes is None else num_classes

        # inception

        if (self.name == "meta_inception1d"):  # note: order important for string capture
            model = meta_inception1d(num_classes=num_classes, input_channels=self.input_channels, use_residual=True,
                                ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head,
                                kernel_size=8 * self.kernel_size)
            subnet = VCNN(hidden1=100, output=5, k_size=5)
        else:
            print("Model not found.")
            assert (True)

        return data_loader_train, data_loader_valid, data_loader_test, loss, model, subnet





