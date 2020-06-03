import tensorflow as tf
tf.debugging.set_log_device_placement(True)
import logging


def allow_soft_placement():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if  len(physical_devices) == 0:
        logging.warning("Not enough GPU hardware devices available")
    else:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

allow_soft_placement()

import os
import sys
from dotenv import load_dotenv, find_dotenv
import numpy as np
import pandas as pd
from hyperopt import space_eval

sys.path.append(os.path.abspath(".."))

# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()
# load up the entries as environment variables
load_dotenv(dotenv_path)
# import pandas as pd

import random
import matplotlib.pyplot as plt
from utils.data_handler import read_pickle,save_to_pickle
from models.many_to_one_models import *
from models.hyperparameter_tuning import safeHyperopt,extract_trial_results
from hyperopt import hp, space_eval
import argparse

if __name__ == "__main__":
    #Set seeds
    seed = 100
    random.seed(seed)
    np.random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',type=str)
    parser.add_argument('--version',type=int)
    parser.add_argument('--total_trials',type=int,default = 50)
    parser.add_argument('--train_data',type=str,default='processed_data_train.pickle')
    parser.add_argument('--val_data',type=str,default='processed_data_val.pickle')
    parser.add_argument("--processed_data_path", type=str, default="../../../data/processed")
    parser.add_argument("--raw_data_path", type=str, default="../../../data/raw")
    parser.add_argument("--interim_data_path", type=str, default="../../../data/interim")


    args = parser.parse_args()

    #Import data
    train_data = read_pickle(os.path.join(args.processed_data_path,args.train_data))
    val_data = read_pickle(os.path.join(args.processed_data_path,args.val_data))

    #Verify GPU is available
    num_gpus_available = len(tf.config.experimental.list_physical_devices('GPU'))
    assert num_gpus_available == 1, 'GPU IS NOT AVAILABLE' 
    print("Num GPUs Available: ",num_gpus_available )


    #Initialize trainer class for model
    if args.model_name == "BidirectionalLSTM":
        trainer = BidirectionalLSTM(train_data,
                                    val_data,
                                    "../../../data/raw/pretrained_embeddings/glove.twitter.27B/glove.twitter.27B.200d.txt")

    elif args.model_name == "LstmCnn":
        trainer = LstmCnn(train_data,
                                    val_data,
                                    "../../../data/raw/pretrained_embeddings/glove.twitter.27B/glove.twitter.27B.200d.txt")
    else:
        print("model_name specificied does not exist")

    #Define search space. TODO: Define this dictionary through external json file
    space = {   'n_units': hp.choice('n_units',2**np.arange(6,9,dtype=int)),
                'add_recurrent_layer': True,
                'dropout': hp.choice('dropout', np.arange(.3,0.6,0.1)),
                'spatial_dropout':hp.choice('spatial_dropout',np.arange(.3,.6,.1)),
                'hidden_dense_units': hp.choice('use_dense',[0,hp.choice('hidden_dense_units',2**np.arange(4,8))]),
                'batch_size' :hp.choice('batch_size',2**np.arange(5,8,dtype=int)),
                'epochs' : 100, 
                'learning_rate':10**hp.uniform('learning_rate',-4,-2.3),
                'bidirectional':True,
                'global_max_pool':hp.choice('global_max_pool',[True,False]),
                'global_avg_pool':hp.choice('global_avg_pool',[True,False])
            }

    #Start Hyperparameter search
    trainer.search_hyperparameters(space = space,
                                   version = args.version,
                                   total_trials = args.total_trials)
