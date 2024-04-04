#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 09:58:06 2022

@author: alain
"""




import os
import argparse

parser = argparse.ArgumentParser(description="FedOTDD")


parser.add_argument("--setting", type=int, default=0)

args = parser.parse_args()

# generate the second panel in Figure 6
# change parameter epochs, n_supp for other panels 
n_samples = 5000
epochs = 20
n_supp = 1000
size = 28 # image size
seed = 0
for source in ['MNIST','FashionMNIST','KMNIST','USPS']:
    for target in ['MNIST','FashionMNIST','KMNIST','USPS']:
        command = f"nohup python -u otdd_expe.py --seed {seed:} --s {source:} --t {target:}"
        command += f" --n_samples {n_samples:} --epochs {epochs:} --n_supp {n_supp:} --size {size:}"
        command += f" > out_dd_{source:}_{target:}.log "
        os.system(command)

