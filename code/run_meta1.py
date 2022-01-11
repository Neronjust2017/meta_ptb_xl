from experiments.scp_experiment import SCP_Experiment
from utils import utils
# model configs
from configs.fastai_configs import *
from configs.wavelet_configs import *
from configs.your_configs import *
from configs.meta_configs import *

import os
import argparse
import torch
import wandb

def main(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    torch.manual_seed(args.seed)  # cpu
    torch.cuda.manual_seed(args.seed)  # gpu

    datafolder = '../data/ptbxl/'
    outputfolder = '../output/'

    models = [
        conf_meta_inception1d
        ]

    ##########################################
    # STANDARD SCP EXPERIMENTS ON PTBXL
    ##########################################

    name = 'meta_inception1d_{}_{}_{}_{}'.format(args.task, args.corruption_type, args.corruption_prob, args.seed)
    wandb.init(project="ptb_meta", name=name)
    experiments = [
        (name, args.task)
    ]

    for name, task in experiments:
        e = SCP_Experiment(name, task, datafolder, outputfolder, models, noise_type=args.corruption_type, noise_ratio=args.corruption_prob)

        # Load PTB-XL data
        # Select relevant data and convert to one-hot
        # split data
        # Preprocess signal data
        # save train and test labels
        e.prepare()

        # load model and fit
        e.perform()

        # predict
        # e.evaluate()

    # generate greate summary table
    utils.generate_ptbxl_summary_table()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--corruption_prob', type=float, default=0.3, help='label noise')
    parser.add_argument('--corruption_type', '-ctype', type=str, default='NCAR')
    parser.add_argument('--task', type=str, default='superdiagnostic',
                        help='all, diagnostic, subdiagnostic, superdiagnostic, form, rhythm')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--np_seed', type=int, default=202002)
    parser.add_argument('--cuda', type=str, default='0', help='cuda visible device')
    parser.set_defaults(augment=True)

    args = parser.parse_args()
    main(args)
