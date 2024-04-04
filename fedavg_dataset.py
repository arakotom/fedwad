# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/main_fed.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# This program implements FedRep under the specification --alg fedrep, as well as Fed-Per (--alg fedper), LG-FedAvg (--alg lg), 
# FedAvg (--alg fedavg) and FedProx (--alg prox)
#%%
import sys
sys.path.append('fedrep/')
sys.path.append('CIFAR10AE/')

import copy
import itertools
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import datasets, transforms

from utils.options import args_parser
from utils.train_utils import get_data, get_model, read_data
from models.Update import LocalUpdate, DatasetSplit
from models.test import test_img_local_all
from torch.utils.data import DataLoader, Dataset
from myOTDD import otdd, augment_data
import time
import ot
from  FedWaD import FedOT
import random

def random_segment_split(K, N):
    if N <= 0:
        return []

    segments = []
    remaining_range = K

    for i in range(N - 1):
        #segment_size = random.randint(1, remaining_range - (N - len(segments)) + 1)
        segment_size = K//N
        if i == N - 1:
            segment_size = remaining_range
        
        segments.append(segment_size)
        remaining_range -= segment_size

    segments.append(remaining_range)
    return segments

def shuffle_dict_keys(d, shuffled_dict_keys=None):
    keys = list(d.keys())
    if shuffled_dict_keys is None:
        np.random.shuffle(keys)
    else:
        keys = shuffled_dict_keys
    
    shuffled_dict = {key: d[key] for key in keys}
    return shuffled_dict, keys

def sample(dataset,classes, num_users=100):
    """_summary_
    Legacy function to sample data from dataset for each client.
    Args:
        dataset (_type_): _description_
        classes (_type_): _description_
        num_users (int, optional): _description_. Defaults to 100.

    Returns:
        _type_: _description_
    """
    N = len(classes)
    labels = dataset.targets.numpy()
    idx_class = []
    for i in range(N):
        ind = np.where(np.isin(labels,classes[i]))[0]
        idx_class.append(list(ind))
    
    
    # classes are the set of classes that are going to be used per client
    # segments are list of number of users per class 
    segments = random_segment_split(num_users, N)
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    list_users_class = []
    k = 0
    for i in range(N):
        nb_samples = len(idx_class[i])
        nb_users_in_class = segments[i]
        idx_us = random_segment_split(nb_samples, nb_users_in_class)
        last_idx = np.cumsum(idx_us)
        idx_class_shuffled = np.random.permutation(idx_class[i])
        for j in range(nb_users_in_class):
            if j == 0:
                dict_users[k] = idx_class_shuffled[:idx_us[j]]
            else:
                #print(idx_us[j-1],idx_us[j])
                dict_users[k] = idx_class_shuffled[last_idx[j-1]:last_idx[j-1]+idx_us[j]]
            #print(k,dict_users[k])
            list_users_class.append(i)
            k += 1

    return dict_users, list_users_class

def sample2(y,grp_classes, num_users=100,n_class=10, dosort=False, 
            class_of_user = None):
    assert num_users >= 2*len(grp_classes)
    # generates which grp_class each user has access to
    if class_of_user is None:
        unique_int = np.arange(len(grp_classes))
        class_of_user = np.repeat(unique_int,2)  # a least a pair of each class
        if num_users > len(grp_classes):
            class_of_user = np.concatenate((class_of_user,np.random.randint(0,len(grp_classes),(num_users-2*len(grp_classes),))))
        np.random.shuffle(class_of_user)    
    if dosort:
        # useful for visualization
        class_of_user = np.sort(class_of_user)
    # count number of users per class
    # and list all the user in class 
    nb_users_per_class = np.zeros((n_class,))
    user_in_class = {i:[] for i in range(n_class)}
    for idx in range(num_users):
        user_class = grp_classes[class_of_user[idx]]
        for c in range(n_class):
            if c in user_class:
                nb_users_per_class[c] += 1
                user_in_class[c].append(idx)
    # check by counting number of users per class . ok
    
    # for each class, create a list of users that have access to this class
    # split the index of class in nb_users_per_class and assign a split to each user 
    
    # for each class, list all sample of that class, shuffle them and split them 
    # in nb_users_per_class   
    # then of each user using that class, concatenate the samples of that classr to
    # the user's list of samples
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}    
    for i in range(n_class):
        ind = np.where(y == i)[0]
        nb_samples = len(ind)
        indices = ind[np.random.permutation(nb_samples)]

        splits_of_class = random_segment_split(nb_samples, int(nb_users_per_class[i]))
        np.random.shuffle(splits_of_class)
        last_idx = np.cumsum(splits_of_class)

        for j,idx_user in enumerate(user_in_class[i]):
            if j == 0:
                dict_users[idx_user] = np.concatenate((dict_users[idx_user],indices[0:last_idx[j]]))  
            else:
                dict_users[idx_user] = np.concatenate((dict_users[idx_user],indices[last_idx[j-1]:last_idx[j]]))
            #print(idx_user,dict_users[idx_user])
            # shuffle the samples of each user
            dict_users[idx_user] = dict_users[idx_user][np.random.permutation(dict_users[idx_user].shape[0])]

    return dict_users, class_of_user
    
def data_to_list(dataset_train, dict_users_train,args,dim=784,batch_size=10000):
    data_list = []
    if args.dataset == 'cifar10':
        autoencoder = Autoencoder()

    for idx in range(args.num_users):
        dataset=dataset_train
        idxs=dict_users_train[idx]
        ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=batch_size, shuffle=True)
    
        #for i,data in enumerate(ldr_train):
        #    data_list.append([data[0].numpy().reshape(-1,dim).astype(np.double),data[1].numpy()])
            #print(idx,np.unique(data[1].numpy()))
        data = next(iter(ldr_train))
        if args.dataset == 'mnist':
            #data_list.append([data[0].numpy().reshape(-1,dim).astype(np.double),data[1].numpy()])
            data_list.append([data[0].numpy().reshape(data[0].shape[0],-1).astype(np.double),data[1].numpy()])
        elif args.dataset == 'cifar10':
            code, decoded_imgs = autoencoder(data[0])
            label = data[1]
            data_ = code.reshape(code.shape[0],-1).detach().numpy()
            data_list.append([data_.astype(np.double),label.numpy()])
        
    return data_list

def compute_otdd(data_list,args,method='fedot'):


    M = np.zeros((args.num_users,args.num_users))
    for i in range(args.num_users):
        xs = data_list[i][0]
        ys = data_list[i][1].copy()
        print(i,np.unique(ys))
        ys_ = np.unique(ys)
        for kk in range(len(ys_)):
            ind_ = np.where(ys == ys_[kk])
            ys[ind_] = kk
        for j in range(i+1,args.num_users):
            xt = data_list[j][0]
            yt = data_list[j][1].copy()
            #print(j,np.unique(yt))

            yt_ = np.unique(yt)
            if len(yt_)>1:
                for kk in range(len(yt_)):
                    ind_ = np.where(yt == yt_[kk])
                    yt[ind_] = kk
                n_class = np.maximum(len(np.unique(ys)),len(np.unique(yt)))
                if method == "otdd":
                    D,P = otdd(xs,ys,xt,yt,n_class,use_diag=True)
                elif method == 'fedot':
                        Xaug_s, ys = augment_data(xs,ys,n_class,use_diag=True)    
                        Xaug_t, yt = augment_data(xt,yt,n_class,use_diag=True)    
                        start_time = time.time()

                        M_a = ot.dist(Xaug_s, Xaug_t) # dist matrix
                        P_a = ot.emd([],[],M_a)
                        D_a = np.sum(P_a*M_a)


                        fedOT = FedOT(n_supp=args.n_supp, n_epoch=args.n_epochs,verbose=False)
                        fedOT.fit(Xaug_s, Xaug_t,approx_interp=True)
                        D = fedOT.cost**2
            else :
                D = 1e10
            print(i,D,ys_,yt_)
            M[i,j] = D
    return (M + M.T)

def cluster_client(distance_matrix, n_clusters=5,n_neighbors=3,n_init=10,max_val = 5000):
    from sklearn.cluster import SpectralClustering
    distance_matrix = np.where(distance_matrix == 1e10, max_val, distance_matrix)

    clustering = SpectralClustering(n_clusters=n_clusters,n_neighbors=n_neighbors,
            assign_labels='kmeans',affinity='precomputed_nearest_neighbors',n_init=n_init).fit_predict(distance_matrix)
    from matplotlib import pyplot as plt
    
    plt.imshow(distance_matrix)
    print(clustering)
    return clustering

def cluster_client_precomputed(distance_matrix, n_clusters=5,n_neighbors=3,n_init=10,max_val = 5000):
    from sklearn.cluster import SpectralClustering
    distance_matrix = np.where(distance_matrix == 1e10, max_val, distance_matrix)
    # for this clustering setting we need an affinity matrix
    # with large value for similar clients and small value for dissimilar clients
    distance_matrix = (max_val - distance_matrix)/max_val
    clustering = SpectralClustering(n_clusters=n_clusters,n_neighbors=n_neighbors,
            assign_labels='kmeans',affinity='precomputed',n_init=n_init).fit_predict(distance_matrix)
    from matplotlib import pyplot as plt
    
    plt.imshow(distance_matrix)
    print(clustering)
    return clustering



if __name__ == '__main__':
    # parse args
    sys.argv=['']
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.num_users = 10
    args.n_supp = 100
    args.dataset = 'cifar10'
    args.n_epochs = 10
    # create mnist dataset 
    
    trans_mnist = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('data/mnist/', train=False, download=True, transform=trans_mnist)
    
    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)

    
    # generates classes and datasets for each client
    classes = [[0,1],[2,3],[4,5],[6,7],[8,9]]
    #dict_users_train, list_user_class  = sample2(dataset_train,classes,num_users=args.num_users)
    #dict_users_test, _ = sample2(dataset_test,classes,num_users=args.num_users)
    if args.dataset == 'mnist':
        targets_train = dataset_train.targets.numpy()
        targets_test = dataset_test.targets.numpy()
    elif args.dataset == 'cifar10':
        targets_train = np.array(dataset_train.targets)
        targets_test = np.array(dataset_test.targets)
        
    dict_users_train, list_user_class_train = sample2(targets_train,classes,
                                               num_users=args.num_users,dosort=True)
    dict_users_test, list_user_class_test = sample2(targets_test,classes,
                                               num_users=args.num_users,dosort=True,
                                               class_of_user=list_user_class_train)
    list_ind = np.array([])
    for i in range(args.num_users):
        list_ind = np.concatenate((list_ind,dict_users_train[i])) 
        
    
    for i in range(args.num_users):
        labels = np.unique(targets_train[dict_users_train[i]])
        labels_t = np.unique(targets_test[dict_users_test[i]])

        print(i,labels, labels_t)
    

    # extract matrix of data for each client, compute OTDD and cluster clients
    data_list = data_to_list(dataset_train, dict_users_train,args,dim=784,batch_size=10000)
    #distance = compute_otdd(data_list,args,method='fedot')
    #clustered_client = cluster_client(distance, n_clusters=5,n_neighbors=3)
    
    # autoencoder = Autoencoder()
    # data_list = []
    # for idx in range(args.num_users):
    #     dataset=dataset_train
    #     idxs=dict_users_train[idx]
    #     ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=10000, shuffle=True)
    #     data,label = next(iter(ldr_train))
    
    #     code, decoded_imgs = autoencoder(data)
    #     data_ = code.reshape(code.shape[0],-1).detach().numpy()
    #     data_list.append([data_.astype(np.double),label.numpy()])

    


# %%
