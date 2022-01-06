from utils import utils
import os
import pickle
import pandas as pd
import numpy as np
import multiprocessing
from itertools import repeat

class SCP_Experiment():
    '''
        Experiment on SCP-ECG statements. All experiments based on SCP are performed and evaluated the same way.
    '''

    def __init__(self, experiment_name, task, datafolder, outputfolder, models, sampling_frequency=100, min_samples=0, train_fold=8, val_fold=9, test_fold=10, folds_type='strat',
                 # for label noise
                 seed=0,
                 noise_type='NCAR',
                 noise_ratio=0.1

                 ):
        self.seed = seed
        self.noise_type = noise_type
        self.noise_ratio = noise_ratio

        self.models = models
        self.min_samples = min_samples
        self.task = task
        self.train_fold = train_fold
        self.val_fold = val_fold
        self.test_fold = test_fold
        self.folds_type = folds_type
        self.experiment_name = experiment_name
        self.outputfolder = outputfolder
        self.datafolder = datafolder
        self.sampling_frequency = sampling_frequency

        # create folder structure if needed
        if not os.path.exists(self.outputfolder+self.experiment_name):
            os.makedirs(self.outputfolder+self.experiment_name)
            if not os.path.exists(self.outputfolder+self.experiment_name+'/results/'):
                os.makedirs(self.outputfolder+self.experiment_name+'/results/')
            if not os.path.exists(outputfolder+self.experiment_name+'/models/'):
                os.makedirs(self.outputfolder+self.experiment_name+'/models/')
            if not os.path.exists(outputfolder+self.experiment_name+'/data/'):
                os.makedirs(self.outputfolder+self.experiment_name+'/data/')

    def prepare(self):
        # Load PTB-XL data
        self.data, self.raw_labels = utils.load_dataset(self.datafolder, self.sampling_frequency)
        # self.data: ndarray (21837, 1000, 12)
        # self.raw_labels: DataFrame (21837, 27)

        # Preprocess label data
        self.labels = utils.compute_label_aggregations(self.raw_labels, self.datafolder, self.task)
        # self.labels: DataFrame (21837, 30)

        # Select relevant data and convert to one-hot
        # 转换为one-hot标签
        # all 71
        # super 5
        self.data, self.labels, self.Y, _ = utils.select_data(self.data, self.labels, self.task, self.min_samples, self.outputfolder+self.experiment_name+'/data/')
        self.input_shape = self.data[0].shape

        # self.data: (21430, 1000, 12)
        # self.Y: (21430, 5) super

        # 10th fold for testing (9th for now)

        # train_fold = 8
        # val_fold = 9
        # test_fold = 10

        self.X_test = self.data[self.labels.strat_fold == self.test_fold]
        self.y_test = self.Y[self.labels.strat_fold == self.test_fold]
        # 9th fold for validation (8th for now)
        self.X_val = self.data[self.labels.strat_fold == self.val_fold]
        self.y_val = self.Y[self.labels.strat_fold == self.val_fold]
        # rest for training
        self.X_train = self.data[self.labels.strat_fold <= self.train_fold]
        self.y_train = self.Y[self.labels.strat_fold <= self.train_fold]

        # corrupt labels
        if self.noise_type == 'NCAR':
            np.random.seed(self.seed)
            self.y_train_noisy = np.zeros(shape=self.y_train.shape)
            for i in range(self.y_train.shape[0]):
                for j in range(self.y_train.shape[1]):
                    r = np.random.uniform(0, 1)
                    if r <= self.noise_ratio:
                        self.y_train_noisy[i, j] = self.y_train[i, j] ^ 1
                    else:
                        self.y_train_noisy[i, j] = self.y_train[i, j]

        elif self.noise_type == 'NAR':
            exit(1)

        # Preprocess signal data
        # 数据预处理
        self.X_train, self.X_val, self.X_test = utils.preprocess_signals(self.X_train, self.X_val, self.X_test, self.outputfolder+self.experiment_name+'/data/')
        self.n_classes = self.y_train.shape[1]

        # save train and test labels
        self.y_train_noisy.dump(self.outputfolder + self.experiment_name+ '/data/y_train_noisy.npy')
        self.y_train.dump(self.outputfolder + self.experiment_name+ '/data/y_train.npy')
        self.y_val.dump(self.outputfolder + self.experiment_name+ '/data/y_val.npy')
        self.y_test.dump(self.outputfolder + self.experiment_name+ '/data/y_test.npy')

        modelname = 'naive'
        # create most naive predictions via simple mean in training
        mpath = self.outputfolder+self.experiment_name+'/models/'+modelname+'/'
        # create folder for model outputs
        if not os.path.exists(mpath):
            os.makedirs(mpath)
        if not os.path.exists(mpath+'results/'):
            os.makedirs(mpath+'results/')

        # mean_y = np.mean(self.y_train, axis=0)
        # np.array([mean_y]*len(self.y_train)).dump(mpath + 'y_train_pred.npy')
        # np.array([mean_y]*len(self.y_test)).dump(mpath + 'y_test_pred.npy')
        # np.array([mean_y]*len(self.y_val)).dump(mpath + 'y_val_pred.npy')


    def perform(self):

        for model_description in self.models:

            # 解析模型信息
            modelname = model_description['modelname']
            modeltype = model_description['modeltype']
            modelparams = model_description['parameters']

            mpath = self.outputfolder+self.experiment_name+'/models/'+modelname+'/'
            # create folder for model outputs
            if not os.path.exists(mpath):
                os.makedirs(mpath)
            if not os.path.exists(mpath+'results/'):
                os.makedirs(mpath+'results/')

            n_classes = self.Y.shape[1]

            # load respective model
            # 加载模型
            if modeltype == 'WAVELET':
                from models.wavelet import WaveletModel
                model = WaveletModel(modelname, n_classes, self.sampling_frequency, mpath, self.input_shape, **modelparams)
            elif modeltype == "fastai_model":
                from models.fastai_model import fastai_model
                model = fastai_model(modelname, n_classes, self.sampling_frequency, mpath, self.input_shape, **modelparams)
            elif modeltype == "your_model":
                # YOUR MODEL GOES HERE!
                from models.your_model import YourModel
                model = YourModel(modelname, n_classes, self.sampling_frequency, mpath, self.input_shape, **modelparams)
            elif modeltype == "meta_model":
                from models.meta_model import MetaModel
                model = MetaModel(modelname, n_classes, self.sampling_frequency, mpath, self.input_shape, **modelparams)
            else:
                assert(True)
                break

            # fit model
            y_train = np.hstack((self.y_train_noisy, self.y_train))
            model.fit(self.X_train, y_train, self.X_val, self.y_val)

            # predict
            train_noisy_metric = model.predict(self.X_train, self.y_train_noisy)
            train_metric = model.predict(self.X_train, self.y_train)
            val_metric = model.predict(self.X_val, self.y_val)
            test_metric = model.predict(self.X_test, self.y_test)

            te_df_result = pd.DataFrame(
                np.array(
                    [
                        train_noisy_metric['macro_auc'],
                        train_metric['macro_auc'],
                        val_metric['macro_auc'],
                        test_metric['macro_auc']
                    ]
                )
                # columns=te_df.columns,
                # index=['point', 'mean', 'lower', 'upper']
            )

            te_df_result.to_csv(self.outputfolder+self.experiment_name+'/models/'+modelname+'/results/te_results.csv')

            # predict and dump
            # model.predict(self.X_train).dump(mpath+'y_train_pred.npy')
            # model.predict(self.X_val).dump(mpath+'y_val_pred.npy')
            # model.predict(self.X_test).dump(mpath+'y_test_pred.npy')

        # modelname = 'ensemble'
        #
        # # create ensemble predictions via simple mean across model predictions (except naive predictions)
        # ensemblepath = self.outputfolder+self.experiment_name+'/models/'+modelname+'/'
        #
        # # create folder for model outputs
        # if not os.path.exists(ensemblepath):
        #     os.makedirs(ensemblepath)
        # if not os.path.exists(ensemblepath+'results/'):
        #     os.makedirs(ensemblepath+'results/')
        #
        # # load all predictions
        # ensemble_train, ensemble_val, ensemble_test = [],[],[]
        # for model_description in os.listdir(self.outputfolder+self.experiment_name+'/models/'):
        #     if not model_description in ['ensemble', 'naive']:
        #         mpath = self.outputfolder+self.experiment_name+'/models/'+model_description+'/'
        #         ensemble_train.append(np.load(mpath+'y_train_pred.npy', allow_pickle=True))
        #         ensemble_val.append(np.load(mpath+'y_val_pred.npy', allow_pickle=True))
        #         ensemble_test.append(np.load(mpath+'y_test_pred.npy', allow_pickle=True))
        #
        # # dump mean predictions
        # np.array(ensemble_train).mean(axis=0).dump(ensemblepath + 'y_train_pred.npy')
        # np.array(ensemble_test).mean(axis=0).dump(ensemblepath + 'y_test_pred.npy')
        # np.array(ensemble_val).mean(axis=0).dump(ensemblepath + 'y_val_pred.npy')

    def evaluate(self, n_bootstraping_samples=100, n_jobs=20, bootstrap_eval=False, dumped_bootstraps=True):

        # get labels
        y_train = np.load(self.outputfolder+self.experiment_name+'/data/y_train.npy', allow_pickle=True)
        y_val = np.load(self.outputfolder+self.experiment_name+'/data/y_val.npy', allow_pickle=True)
        y_test = np.load(self.outputfolder+self.experiment_name+'/data/y_test.npy', allow_pickle=True)

        # if bootstrapping then generate appropriate samples for each
        if bootstrap_eval:
            if not dumped_bootstraps:
                train_samples = np.array(utils.get_appropriate_bootstrap_samples(y_train, n_bootstraping_samples))
                test_samples = np.array(utils.get_appropriate_bootstrap_samples(y_test, n_bootstraping_samples))
                val_samples = np.array(utils.get_appropriate_bootstrap_samples(y_val, n_bootstraping_samples))
            else:
                test_samples = np.load(self.outputfolder+self.experiment_name+'/test_bootstrap_ids.npy', allow_pickle=True)
        else:
            train_samples = np.array([range(len(y_train))])
            test_samples = np.array([range(len(y_test))])
            val_samples = np.array([range(len(y_val))])

        # store samples for future evaluations
        train_samples.dump(self.outputfolder+self.experiment_name+'/train_bootstrap_ids.npy')
        test_samples.dump(self.outputfolder+self.experiment_name+'/test_bootstrap_ids.npy')
        val_samples.dump(self.outputfolder+self.experiment_name+'/val_bootstrap_ids.npy')

        # iterate over all models fitted so far
        for m in sorted(os.listdir(self.outputfolder+self.experiment_name+'/models')):
            print(m)
            mpath = self.outputfolder+self.experiment_name+'/models/'+m+'/'
            rpath = self.outputfolder+self.experiment_name+'/models/'+m+'/results/'

            # load predictions
            y_train_pred = np.load(mpath+'y_train_pred.npy', allow_pickle=True)
            y_val_pred = np.load(mpath+'y_val_pred.npy', allow_pickle=True)
            y_test_pred = np.load(mpath+'y_test_pred.npy', allow_pickle=True)

            if self.experiment_name == 'exp_ICBEB':
                # compute classwise thresholds such that recall-focused Gbeta is optimized
                thresholds = utils.find_optimal_cutoff_thresholds_for_Gbeta(y_train, y_train_pred)
            else:
                thresholds = None

            pool = multiprocessing.Pool(n_jobs)

            # tr_df = pd.concat(pool.starmap(utils.generate_results, zip(train_samples, repeat(y_train), repeat(y_train_pred), repeat(thresholds))))
            # tr_df_point = utils.generate_results(range(len(y_train)), y_train, y_train_pred, thresholds)
            # tr_df_result = pd.DataFrame(
            #     np.array([
            #         tr_df_point.mean().values, 
            #         tr_df.mean().values,
            #         tr_df.quantile(0.05).values,
            #         tr_df.quantile(0.95).values]), 
            #     columns=tr_df.columns,
            #     index=['point', 'mean', 'lower', 'upper'])

            te_df = pd.concat(pool.starmap(utils.generate_results, zip(test_samples, repeat(y_test), repeat(y_test_pred), repeat(thresholds))))
            te_df_point = utils.generate_results(range(len(y_test)), y_test, y_test_pred, thresholds)

            tr_df = pd.concat(pool.starmap(utils.generate_results, zip(train_samples, repeat(y_train), repeat(y_train_pred), repeat(thresholds))))
            tr_df_point = utils.generate_results(range(len(y_train)), y_train, y_train_pred, thresholds)

            va_df = pd.concat(pool.starmap(utils.generate_results, zip(val_samples, repeat(y_val), repeat(y_val_pred), repeat(thresholds))))
            va_df_point = utils.generate_results(range(len(y_val)), y_val, y_val_pred, thresholds)

            te_df_result = pd.DataFrame(
                np.array(
                    [
                        tr_df_point.mean().values,
                        tr_df.mean().values,
                        tr_df.quantile(0.05).values,
                        tr_df.quantile(0.95).values,

                        va_df_point.mean().values,
                        va_df.mean().values,
                        va_df.quantile(0.05).values,
                        va_df.quantile(0.95).values,

                        te_df_point.mean().values,
                        te_df.mean().values,
                        te_df.quantile(0.05).values,
                        te_df.quantile(0.95).values

                    ]
                )
                # columns=te_df.columns,
                # index=['point', 'mean', 'lower', 'upper']
            )

            # val_df = pd.concat(pool.starmap(utils.generate_results, zip(val_samples, repeat(y_val), repeat(y_val_pred), repeat(thresholds))))
            # val_df_point = utils.generate_results(range(len(y_val)), y_val, y_val_pred, thresholds)
            # val_df_result = pd.DataFrame(
            #     np.array([
            #         val_df_point.mean().values, 
            #         val_df.mean().values,
            #         val_df.quantile(0.05).values,
            #         val_df.quantile(0.95).values]), 
            #     columns=val_df.columns, 
            #     index=['point', 'mean', 'lower', 'upper'])

            pool.close()

            # dump results
            #tr_df_result.to_csv(rpath+'tr_results.csv')
            #val_df_result.to_csv(rpath+'val_results.csv')
            te_df_result.to_csv(rpath+'te_results.csv')
