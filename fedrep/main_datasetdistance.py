# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/main_fed.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# This program implements FedRep under the specification --alg fedrep, as well as Fed-Per (--alg fedper), LG-FedAvg (--alg lg), 
# FedAvg (--alg fedavg) and FedProx (--alg prox)
#%%
import copy
import itertools
import numpy as np
import pandas as pd
import torch
from torch import nn

from utils.options import args_parser
from utils.train_utils import get_data, get_model, read_data
from models.Update import LocalUpdate, DatasetSplit
from models.test import test_img_local_all
from torch.utils.data import DataLoader, Dataset

import time
import sys

if __name__ == '__main__':
    # parse args
    sys.argv=['']
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    lens = np.ones(args.num_users)
    if 'cifar' in args.dataset or args.dataset == 'mnist':
        dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
        for idx in dict_users_train.keys():
            np.random.shuffle(dict_users_train[idx])

    for idx in range(100):
        dataset=dataset_train
        idxs=dict_users_train[idx]
    ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=True)
    
    for i,data in enumerate(ldr_train):
        print(i,data[0].shape,data[1].shape)
        break
    
# %%
