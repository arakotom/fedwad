#%%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import ot
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from  FedWaD import FedOT, InterpMeas
from sklearn.datasets import make_moons, make_blobs
import time
import argparse
import sys


if __name__ == "__main__":
    #sys.argv = ['']
    seed = 42
    np.random.seed(seed)
    parser = argparse.ArgumentParser(description="Argument Parser Example")
    
    parser.add_argument("--n_iter", type=int, default=10, help="Number of iterations")
    parser.add_argument("--n_supp", type=int, default=10, help="Number of supports")
    parser.add_argument("--n_epoch", type=int, default=20, help="Number of epochs")
    parser.add_argument("--dim", type=int, default=50, help="dim")

    parser.add_argument("--ratio_target", type=int, default=3, help="Ratio target value")
    
    args = parser.parse_args()

    n_list = [10,20,50,100,200, 500, 1000,2000]# 5000,10000]

    n_iter = args.n_iter
    n_supp = args.n_supp
    n_epoch = args.n_epoch
    dim = args.dim
    ratio_target = args.ratio_target
    filesave_results = f"results/results-speed-n_supp-{n_supp:}-n_epoch-{n_epoch}-ratio-{ratio_target:}-dim-{dim:}.npz" 

    toc_ot = np.zeros((len(n_list),n_iter))
    toc_approx = np.zeros((len(n_list),n_iter))
    toc_exact = np.ones((len(n_list),n_iter))*np.inf

    cost_ot = np.zeros((len(n_list),n_iter))
    cost_approx = np.zeros((len(n_list),n_iter))
    cost_exact = np.ones((len(n_list),n_iter))*np.inf

    supp_approx = np.zeros((len(n_list),n_iter))
    supp_exact = np.zeros((len(n_list),n_iter))
    for i,n in enumerate(n_list) :
        print(f"n={n:}")
        for j in range(n_iter):
            if dim == 2:
                mu_s = np.array([0, 0])
                cov_s = np.array([[0.1, 0], [0, 0.1]])

                mu_t = np.array([4, 4])/np.sqrt(dim)
                cov_t = np.array([[0.1, 0], [0, 0.1]])
                exact_ot = np.sqrt(np.sum((mu_t-mu_s)**2))

                xs = ot.datasets.make_2D_samples_gauss(n, mu_s, cov_s)
                xt = ot.datasets.make_2D_samples_gauss(n//ratio_target, mu_t, cov_t)
            else:
                mu_s = np.zeros(dim)
                mu_t = np.ones(dim)*4/np.sqrt(dim)
                xs =  np.random.randn(n,dim) + mu_s
                xt = np.random.randn(n,dim) + mu_t
                exact_ot = np.sqrt(np.sum((mu_t-mu_s)**2))
            start_time = time.time()
            interp_meas = InterpMeas().fit(xs,xt)
            cost_ot[i,j] = interp_meas.cost
            toc_ot[i,j] = time.time() - start_time

            fedOT = FedOT(n_supp=n_supp, n_epoch=n_epoch,t_val=0.5,get_int_list=True)
            fedOT.random_val_init = 10
            start_time = time.time()
            fedOT.fit(xs, xt,approx_interp=True)
            cost_approx[i,j] = fedOT.cost
            toc_approx[i,j] =  time.time() - start_time
            supp_approx[i,j] = fedOT.list_int_meas[-1].shape[0]


            if n <= 10000:
                fedOT_e = FedOT(n_supp=n, n_epoch=n_epoch,t_val=0.5,get_int_list=True)
                fedOT_e.random_val_init = 10
                start_time = time.time()
                fedOT_e.fit(xs, xt,approx_interp=False)
                cost_exact[i,j] = fedOT_e.cost
                toc_exact[i,j] = time.time() - start_time
                supp_exact[i,j] = fedOT_e.list_int_meas[-1].shape[0]

    print(toc_ot,toc_approx)
    np.savez(filesave_results, toc_ot=toc_ot, toc_approx=toc_approx, toc_exact=toc_exact, cost_ot=cost_ot, 
            cost_approx=cost_approx, cost_exact=cost_exact,exact_cost=exact_ot,
            supp_approx=supp_approx,supp_exact=supp_exact)


# %%
