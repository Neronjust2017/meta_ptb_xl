from experiments.scp_experiment import SCP_Experiment
from utils import utils
# model configs
from configs.fastai_configs import *
from configs.wavelet_configs import *
from configs.your_configs import *
from configs.meta_configs import *

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    
    datafolder = '../data/ptbxl/'
    outputfolder = '../output/'

    models = [
        conf_meta_inception1d
        ]

    ##########################################
    # STANDARD SCP EXPERIMENTS ON PTBXL
    ##########################################

    experiments = [
        ('meta_inception1d_super_NCAR', 'superdiagnostic')
    ]

    for name, task in experiments:
        e = SCP_Experiment(name, task, datafolder, outputfolder, models)

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
    main()
