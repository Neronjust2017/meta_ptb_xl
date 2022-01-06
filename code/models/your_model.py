import numpy as np
from models.timeseries_utils import *

from pathlib import Path
from functools import partial

from models.resnet1d import resnet1d18,resnet1d34,resnet1d50,resnet1d101,resnet1d152,resnet1d_wang,resnet1d,wrn1d_22
from models.xresnet1d import xresnet1d18,xresnet1d34,xresnet1d50,xresnet1d101,xresnet1d152,xresnet1d18_deep,xresnet1d34_deep,xresnet1d50_deep,xresnet1d18_deeper,xresnet1d34_deeper,xresnet1d50_deeper
from models.inception1d import inception1d
from models.basic_conv1d import fcn,fcn_wang,schirrmeister,sen,basic1d,weight_init
from models.rnn1d import RNN1d
import math

from models.base_model import ClassificationModel
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from fastai.torch_core import to_np

#for lrfind
import matplotlib
import matplotlib.pyplot as plt

from utils.utils import apply_thresholds, challenge_metrics, roc_auc_score

#eval for early stopping
from utils.utils import evaluate_experiment

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

def save_checkpoint(model, optimizer, checkpoint_dir):
    arch = type(model).__name__
    state = {
        'arch': arch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    path = os.path.join(checkpoint_dir, 'model.pth')

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

class YourModel(ClassificationModel):
    def __init__(self, name, n_classes, freq, outputfolder, input_shape, pretrained=False, input_size=2.5,
                 input_channels=12, chunkify_train=False, chunkify_valid=True, bs=128, ps_head=0.5, lin_ftrs_head=[128],
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
        self.chunk_length_valid = self.input_size

        self.min_chunk_length = self.input_size  # chunk_length

        self.stride_length_train = self.input_size  # chunk_length_train//8
        self.stride_length_valid = self.input_size // 2  # chunk_length_valid

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

    def fit(self, X_train, y_train, X_val, y_val):
        # convert everything to float32
        X_train = [l.astype(np.float32) for l in X_train]
        X_val = [l.astype(np.float32) for l in X_val]
        y_train = [l.astype(np.float32) for l in y_train]
        y_val = [l.astype(np.float32) for l in y_val]

        if (self.concat_train_val):
            X_train += X_val
            y_train += y_val

        data_loader_train, data_loader_valid, loss_fn, model = self._get_learner(X_train, y_train, X_val, y_val)
        optimizer = Adam(model.parameters(), lr=self.lr)
        # lr_scheduler = StepLR(optimizer, step_size=int(self.epochs*0.8), gamma=0.1)
        # lr_scheduler = StepLR(optimizer, step_size=int(self.epochs*0.8), gamma=0.1)
        lr_scheduler = CosineAnnealingLR(optimizer, self.lr)

        model.to(device)
        model.apply(weight_init)

        # training

        train_loss_squence = np.zeros(shape=(self.epochs, len(data_loader_train.dataset), self.num_classes))
        train_loss_target = np.zeros(shape=(len(data_loader_train.dataset), 2 * self.num_classes))
        train_ids = []

        for epoch in range(self.epochs):

            for batch_idx, (data, target, id) in enumerate(data_loader_train):
                data, target = data.to(device), target.to(device)
                target = target[:, :self.num_classes]
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()

                # if batch_idx % log_step == 0:
                #     print('Train Epoch: {} {} Loss: {:.6f}'.format(epoch, progress(data_loader_train, batch_idx), loss.item()))
            lr_scheduler.step(epoch)

            # validation
            train_loss = 0
            valid_loss = 0

            with torch.no_grad():

                train_target = []
                train_pred = []
                for batch_idx, (data, target, id) in enumerate(data_loader_train):
                    data, target = data.to(device), target.to(device)
                    train_loss_target[id] = target.detach().cpu()
                    target = target[:, :self.num_classes]
                    output = model(data)

                    if (self.loss == "binary_cross_entropy"):
                        output = nn.Sigmoid()(output)

                    loss = loss_fn(output, target, reduce=False)
                    train_loss += loss.detach().cpu().mean()
                    train_target.append(target.detach().cpu())
                    train_pred.append(output.detach().cpu())

                    train_ids.append(id)
                    train_loss_squence[epoch, id] = loss.detach().cpu()

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

                print('Train Epoch: {} Train Loss: {:.6f} Valid Loss: {:.6f} Train Metric: {:.6f} Valid Metric: {:.6f}'.format(epoch,
                                                                                                    train_loss / len(data_loader_train),
                                                                                                    valid_loss / len(data_loader_valid),
                                                                                                    train_metric['macro_auc'],
                                                                                                    valid_metric['macro_auc']))

            # save loss sequence
            np.save(os.path.join(self.outputfolder, 'loss_squence.npy'), train_loss_squence)
            np.save(os.path.join(self.outputfolder, 'target.npy'), train_loss_target)


        # save
        save_checkpoint(model, optimizer, self.outputfolder)


    def predict(self, X, Y):
        X = [l.astype(np.float32) for l in X]
        Y = [l.astype(np.float32) for l in Y]
        # y_dummy = [np.ones(self.num_classes, dtype=np.float32) for _ in range(len(X))]

        data_loader, data_loader_tmp, loss_fn, model = self._get_learner(X, Y, X, Y)

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

    def _get_learner(self, X_train,y_train, X_val, y_val, num_classes=None):


        df_train = pd.DataFrame({"data": range(len(X_train)), "label": y_train})
        df_valid = pd.DataFrame({"data": range(len(X_val)), "label": y_val})

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

        # data loader
        data_loader_train = DataLoader(ds_train, batch_size=self.bs, shuffle=True, pin_memory=True, num_workers=32)
        data_loader_valid = DataLoader(ds_valid, batch_size=self.bs, shuffle=False, pin_memory=True, num_workers=32)

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

        # resnet
        if (self.name.startswith("your_resnet1d101")):
            model = resnet1d101(num_classes=num_classes, input_channels=self.input_channels, inplanes=128,
                            kernel_size=self.kernel_size, ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head)
        # inception

        elif (self.name == "your_inception1d"):  # note: order important for string capture
            model = inception1d(num_classes=num_classes, input_channels=self.input_channels, use_residual=True,
                                ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head,
                                kernel_size=8 * self.kernel_size)
        else:
            print("Model not found.")
            assert (True)

        return data_loader_train, data_loader_valid, loss, model





