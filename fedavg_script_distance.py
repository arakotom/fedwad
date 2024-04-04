#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon  07 07:58:06 2023

@author: alain

This file can be used to produce the distance matrix between clients in the FL experiments
the npy file is going to be save and then re-used in the fedavg_expe.py file

"""




import os
import argparse

parser = argparse.ArgumentParser(description="OTDD FL")
parser.add_argument("--setting", type=int, default=0)
args = parser.parse_args()
dataset = 'cifar10'
nb_class = 5
nb_seed = 5
for n_supp in [100]:
    for n_neighbor in [0]:# 1 use a spectral clustering method that dont need it. see doc of sklearn

        setting = 'class_clustered'
        for grp_class in [2]:
            for num_users in [100,40,20]:
                for seed in range(1,nb_seed):
                    if setting == 'class_clustered':
                            for i_class in range(nb_class):
                                command = f"nohup python -u fedavg_expe.py --alg fedavg  --setting {setting:} --seed {seed:} --i_class_clustered {i_class:} --grp_class {grp_class:}"
                                command += f" --n_neighbor {n_neighbor} --n_supp {n_supp:} --dataset {dataset:}" 
                                command += f" --num_users {num_users:} > out_cc_seed{seed:}_ic{i_class}.log "
                                os.system(command)
    