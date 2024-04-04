#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon  07 09:58:06 2023

@author: alain
"""




import os
import argparse

parser = argparse.ArgumentParser(description="OTDD FL")
parser.add_argument("--setting", type=int, default=0)
args = parser.parse_args()
dataset = 'mnist'
grp_class = 1
nb_class = 5
nb_seed = 5
for n_supp in [10,100]:
    for n_neighbor in [3,5]:# 1 use a spectral clustering method that dont need it. see doc of sklearn

        if args.setting == 0:
            setting = 'class_clustered'
            for grp_class in [grp_class]:
                for num_users in [10,20,40,100]:
                    for seed in range(nb_seed):
                        if setting == 'class_clustered':
                                for i_class in range(nb_class):
                                    command = f"nohup python -u fedavg_expe.py --alg fedavg  --setting {setting:} --seed {seed:} --i_class_clustered {i_class:} --grp_class {grp_class:}"
                                    command += f" --n_neighbor {n_neighbor} --n_supp {n_supp:} --dataset {dataset:}" 
                                    command += f" --num_users {num_users:} > out_cc_{num_users}_{n_supp}_seed{seed:}_ic{i_class}.log "
                                    os.system(command)
        elif args.setting == 1:

            for grp_class in [grp_class]:#  number of classes per client
                for num_users in [10,20,40,100]:

                    for seed in range(nb_seed):
                        for setting in ['class_clustered_golden','class_full' ]: 
                            if setting == 'class_full':
                                command = f"nohup python -u fedavg_expe.py --alg fedavg --setting {setting:} --seed {seed:} --num_users {num_users:} --grp_class {grp_class:}"
                                command += f" --dataset {dataset:} > out_cf_{num_users}_{n_supp}_seed{seed:}.log "
                                os.system(command)
                            elif setting == 'class_clustered_golden':
                                for i_class in range(nb_class):
                                    command = f"nohup python -u fedavg_expe.py --alg fedavg --setting {setting:} --seed {seed:} --i_class_clustered {i_class:} --grp_class {grp_class:}"
                                    command += f" --n_neighbor {n_neighbor} --n_supp {n_supp:} --dataset {dataset:}" 
                                    command += f" --num_users {num_users:}  > out_ccg_{num_users}_{n_supp}_seed{seed:}_ic{i_class}.log "
                                    os.system(command)

        elif args.setting >= 2 :
            
            for grp_class in [grp_class]:
                for num_users in [100,40,20,10]:      
                    #for seed in range(nb_seed -1,0,-1):
                    for seed in range(nb_seed):

                        if args.setting == 2:
                            list_algo = ['fedrep','fedper']
                        elif args.setting == 3:
                            list_algo = ['fedper']
                        elif args.setting == 4:
                            list_algo = ['fedrep']
                        for alg in list_algo:

                            if args.setting == 2:
                                list_setting = ['class_full','class_clustered_golden']
                            elif args.setting >= 3:
                                list_setting = ['class_clustered']
                            for setting in list_setting:
                                if setting == 'class_full':
                                    command = f"nohup python -u fedavg_expe.py  --alg {alg} --setting {setting:} --seed {seed:} --num_users {num_users:} --grp_class {grp_class:}"
                                    command += f" --dataset {dataset:}  > out_{alg}_{num_users}_{n_supp}_{seed}.log "
                                    os.system(command)
                                elif setting == 'class_clustered' or setting == 'class_clustered_golden':
                                    for i_class in range(nb_class):
                                        command = f"nohup python -u fedavg_expe.py --alg {alg} --setting {setting:} --seed {seed:} --i_class_clustered {i_class:} --grp_class {grp_class:}"
                                        command += f" --n_neighbor {n_neighbor} --n_supp {n_supp:} --dataset {dataset:}" 
                                        command += f" --num_users {num_users:} > out_{alg}_{num_users}_{n_supp}_{seed}.log "
                                        os.system(command)



# %%
