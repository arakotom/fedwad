#%%
import numpy as np
import matplotlib.pyplot as plt

import ot
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from  FedWaD import FedOT, InterpMeas
import importlib
from sklearn.datasets import make_moons, make_blobs
from scipy.linalg import sqrtm
from otdd.otdd.pytorch.datasets import load_torchvision_data
from otdd.otdd.pytorch.distance import DatasetDistance
from myOTDD import otdd, augment_data
import time
import argparse
import pandas as pd

if __name__ == "__main__":



    parser = argparse.ArgumentParser()

    parser.add_argument('--s', type=str, default='MNIST', help="Federated")
    parser.add_argument('--t', type=str, default='USPS', help="number MNIST samples")
    
    parser.add_argument('--n_samples', type=int, default=5000, help="number of epochs")
    parser.add_argument('--epochs', type=int, default=1000, help="number of epochs")
    parser.add_argument('--n_supp', type=int, default=500, help="support size of interpolating measure")
    parser.add_argument('--size', type=int, default=28, help="size of images")
    parser.add_argument('--approx_interp', type=int, default=1, help="approx_interp")

    parser.add_argument('--n_iter', type=int, default=5,help="size of images")
    parser.add_argument('--seed', type=int, default=0, help="seed")
    parser.add_argument('--freq_save_fig', type=int, default=200, help="frequency of saving figure")
    parser.add_argument('--path_dir', type=str, default='./save/otdd/', help="path to save figures")



    args = parser.parse_args()
    
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(args)
    filename = f"otdd-s-{args.s}-t_{args.t}-epochs-{args.epochs:}-n_samples-{args.n_samples:}-size-{args.size:}-seed-{seed:}"
    filename += f"-n_supp-{args.n_supp:}-approx_interp-{args.approx_interp:}"

    n_class = 10
    n = args.n_samples
    size = args.size
    assert args.s in ['MNIST','FashionMNIST','KMNIST','USPS']
    assert args.t in ['MNIST','FashionMNIST','KMNIST','USPS']
    
    values = []
    times = []
    for k in range(args.n_iter):
    
        loaders_src  = load_torchvision_data(args.s, valid_size=0, resize = size, maxsize=n,batch_size=n)[0]
        loaders_tgt  = load_torchvision_data(args.t,  valid_size=0, resize = size, maxsize=n,batch_size=n)[0]

        aaa = loaders_src['train']
        for data,label in aaa:
            xs = data.numpy().reshape(-1,size*size).astype(np.double)
            ys = label.numpy()
        
        aaa = loaders_tgt['train']
        for data,label in aaa:
            xt = data.numpy().reshape(-1,size*size).astype(np.double)
            yt = label.numpy()
        
        start_time = time.time()
        print(yt)
        D,P = otdd(xs,ys,xt,yt,n_class,use_diag=True)
        otdd_time =  time.time() - start_time

        Xaug_s, ys = augment_data(xs,ys,n_class,use_diag=True)    
        Xaug_t, yt = augment_data(xt,yt,n_class,use_diag=True)    
        start_time = time.time()

        M_a = ot.dist(Xaug_s, Xaug_t) # dist matrix
        P_a = ot.emd([],[],M_a)
        D_a = np.sum(P_a*M_a)
        aug_otdd_time =  time.time() - start_time

        print(D,D_a)

        start_time = time.time()
        from  FedWaD import FedOT, InterpMeas
        fedOT = FedOT(n_supp=args.n_supp, n_epoch=args.epochs,verbose=False)
        if args.approx_interp:
            fedOT.fit(Xaug_s, Xaug_t,approx_interp=True)
        else:
            fedOT.fit(Xaug_s, Xaug_t,approx_interp=False)
        fedotdd_time =  time.time() - start_time

        print(D,D_a,fedOT.cost**2)

        value = [D,D_a,fedOT.cost**2]
        values.append(value)
        times.append([otdd_time,aug_otdd_time,fedotdd_time])
        val_otdd = np.array(values)
    
        m_times= np.array(times)
        m_values = np.array(values)
        np.savez(args.path_dir + filename, values=m_values, times=m_times)

        #df = pd.DataFrame({'values': values,'times':times})
        #df.to_csv(args.path_dir + filename, index=False)
