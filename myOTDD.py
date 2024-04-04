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

def get_list_mean_cov(x,y,n_class,use_diag=False):
    list_mean = []
    list_cov12 = []
    dim = x.shape[1]
    y_unique = np.unique(y)
    for i in range(n_class):
        
        ind = np.where(y==i)[0]
        
        xaux = x[ind,:]
        mean_x = np.mean(xaux,axis=0)
        list_mean.append(mean_x)  
        C = np.cov(xaux.T - mean_x.reshape(-1,1))
        C = (C + C.T)/2 
        try:
            evalues, evectors = np.linalg.eig(C)
            cov12 = evectors * np.sqrt(evalues + 1e-6) @ evectors.T
            cov12 = np.real(cov12)
        except:
            cov12 = np.diag(np.ones(C.shape[0]))
        
        
        
        if use_diag==False:
            list_cov12.append(cov12.reshape(-1))
        else:
            list_cov12.append(np.diag(cov12))    
        #list_cov12.append(cov12)
    return list_mean, list_cov12


def otdd(xs,ys,xt,yt,n_class,use_diag=False):
    """ Summary Compute the OTDD between two datasets

    Args:
        xs (_array_): input dataset
        ys (_array_): labels
        xt (_array_): input dataset
        yt (_array_): labels
        n_class (_int_):  number of classes
    
    returns:
        D (float): distance between the two datasets
        P : optimal transport matrix
    """
    #-------------------
    l_mean_s,l_cov12_s = get_list_mean_cov(xs,ys,n_class,use_diag=use_diag)
    l_mean_t,l_cov12_t = get_list_mean_cov(xt,yt,n_class,   use_diag=use_diag)
    W_y = np.zeros((n_class,n_class))

    for i in range(n_class):
        for j in range(n_class):
            W_y[i,j] += np.linalg.norm(l_mean_s[i] - l_mean_t[j])**2 
            W_y[i,j] += np.linalg.norm(l_cov12_s[i] - l_cov12_t[j])**2

    M = ot.dist(xs, xt) # dist matrix
   #print(W_y)
    
    for i in range(xs.shape[0]):
        for j in range(xt.shape[0]):
            M[i,j] += W_y[ys[i],yt[j]]

    P = ot.emd([],[],M/M.max())
    D = np.sum(P*M)
    
    return D, P 


def augment_data(x,y,n_class,use_diag=False):
    n,dim = x.shape
    list_mean, list_cov = get_list_mean_cov(x,y,n_class,use_diag=use_diag)
    #print(list_cov[0].shape)
    if use_diag == False:
        Xaug = np.zeros((n,2*dim + dim*dim))
    else:
        Xaug = np.zeros((n,2*dim + dim))
        
    for i in range(n):
        Xaug[i] = np.concatenate((x[i],list_mean[y[i]],list_cov[y[i]]))
    return Xaug, y


if __name__ == "__main__":
    
    # testing /comparing OTDD with OTDD_augmented and FedWad OTDD
    data = 'toy'
    
    if data == 'toy':
        n = 500  # nb samples
        dim = 2

        xs, ys = make_blobs(n,cluster_std=0.3)
        xt, yt = make_blobs(n,cluster_std=0.3)
        n_class = 3
        plt.figure()
        for i in range(3):
            plt.scatter(xs[ys==i, 0], xs[ys==i, 1], c='r', marker='x')
            plt.scatter(xt[yt==i, 0], xt[yt==i, 1], c='g', marker='x')

        D,P = otdd(xs,ys,xt,yt,n_class)

        Xaug_s, ys = augment_data(xs,ys,n_class)    
        Xaug_t, yt = augment_data(xt,yt,n_class)    
        M_a = ot.dist(Xaug_s, Xaug_t) # dist matrix
        P_a = ot.emd([],[],M_a)
        D_a = np.sum(P_a*M_a)
        
        from  FedWaD import FedOT, InterpMeas

        fedOT = FedOT(n_supp=200, n_epoch=10,t_val=0.5,verbose=True)
        fedOT.fit(Xaug_s, Xaug_t,approx_interp=True)
        print('OTDD:',D)
        print('OTDD_augmented:',D_a)
        print('FedWaD:',fedOT.cost**2)
    else:


        n = 1000
        loaders_src  = load_torchvision_data('MNIST', valid_size=0, resize = 28, maxsize=n,batch_size=n)[0]
        loaders_tgt  = load_torchvision_data('USPS',  valid_size=0, resize = 28, maxsize=n,batch_size=n)[0]

        aaa = loaders_src['train']
        for data,label in aaa:
            xs = data.numpy().reshape(-1,28*28).astype(np.double)
            ys = label.numpy()
        
        aaa = loaders_tgt['train']
        for data,label in aaa:
            xt = data.numpy().reshape(-1,28*28).astype(np.double)
            yt = label.numpy()
        n_class = 10
        
        D,P = otdd(xs,ys,xt,yt,n_class,use_diag=True)
        Xaug_s, ys = augment_data(xs,ys,n_class,use_diag=True)    
        Xaug_t, yt = augment_data(xt,yt,n_class,use_diag=True)    
        M_a = ot.dist(Xaug_s, Xaug_t) # dist matrix
        P_a = ot.emd([],[],M_a)
        D_a = np.sum(P_a*M_a)

        from  FedWaD import FedOT, InterpMeas

        fedOT = FedOT(n_supp=1000, n_epoch=50,t_val=0.5,verbose=True)
        fedOT.fit(Xaug_s, Xaug_t,approx_interp=True)
        print('OTDD:',D)
        print('OTDD_augmented:',D_a)
        print('FedWaD:',fedOT.cost**2)

    
    
    
# %%
