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
import copy
import itertools
import numpy as np
import pandas as pd
import torch
from torch import nn
import os
from utils.train_utils import get_data, get_model
from models.Update import LocalUpdate
from models.test import test_img_local_all
import argparse
from models.Update import DatasetSplit
from torch.utils.data import DataLoader, Dataset
from fedavg_dataset import data_to_list, compute_otdd, cluster_client, sample2, sample
from fedavg_dataset import cluster_client_precomputed

import time

if __name__ == '__main__':
    # parse args
            # parse args
    #sys.argv=['']
    
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=40, help="number of users: n")
    parser.add_argument('--shard_per_user', type=int, default=2, help="classes per user")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--grad_norm', action='store_true', help='use_gradnorm_avging')
    parser.add_argument('--lr_decay', type=float, default=1.0, help="learning rate decay per round")
    parser.add_argument('--local_updates', type=int, default=1000000, help="maximum number of local updates")
    parser.add_argument('--m_tr', type=int, default=10000, help="maximum number of samples/user to use for training")
    parser.add_argument('--m_ft', type=int, default=10000, help="maximum number of samples/user to use for fine-tuning")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--num_layers_keep', type=int, default=1, help='number layers to keep')
    parser.add_argument('--alg', type=str, default='fedrep', help='FL algorithm to use')
    
    # algorithm-specific hyperparameters
    parser.add_argument('--local_rep_ep', type=int, default=1, help="the number of local epochs for the representation for FedRep")
    parser.add_argument('--lr_g', type=float, default=0.1, help="global learning rate for SCAFFOLD")
    parser.add_argument('--mu', type=float, default='0.1', help='FedProx parameter mu')
    parser.add_argument('--gmf', type=float, default='0', help='FedProx parameter gmf')
    parser.add_argument('--alpha_apfl', type=float, default='0.75', help='APFL parameter alpha')
    parser.add_argument('--alpha_l2gd', type=float, default='1', help='L2GD parameter alpha')
    parser.add_argument('--lambda_l2gd', type=float, default='0.5', help='L2GD parameter lambda')
    parser.add_argument('--lr_in', type=float, default='0.0001', help='PerFedAvg inner loop step size')
    parser.add_argument('--bs_frac_in', type=float, default='0.8', help='PerFedAvg fraction of batch used for inner update')
    parser.add_argument('--lam_ditto', type=float, default='1', help='Ditto parameter lambda')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=4, help='random seed (default: 1)')
    parser.add_argument('--test_freq', type=int, default=1, help='how often to test on val set')
    parser.add_argument('--load_fed', type=str, default='n', help='define pretrained federated model path')
    parser.add_argument('--results_save', type=str, default='run', help='define fed results save folder')
    parser.add_argument('--save_every', type=int, default=50, help='how often to save models')

    # arguments for federated OT
    parser.add_argument('--setting', type=str, default='class_clustered', help='how to handle class')
    parser.add_argument('--i_class_clustered', type=int, default=0, help='which cluster to learn')
    parser.add_argument('--grp_class', type=int, default=1, help='number of support in the model')
    parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs in FedOTDD')
    parser.add_argument('--n_supp', type=int, default=10, help='number of support in FedOTDD')
    parser.add_argument('--n_cluster', type=int, default=5, help='number of cluster')
    parser.add_argument('--n_neighbor', type=int, default=1, help='number of neighbour in spectral clustering')



    
    args = parser.parse_args()


    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    print(args)

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    
    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    if args.dataset == 'mnist':
        targets_train = dataset_train.targets.numpy()
        targets_test = dataset_test.targets.numpy()
    elif args.dataset == 'cifar10':
        targets_train = np.array(dataset_train.targets)
        targets_test = np.array(dataset_test.targets)
        args.model = 'cnn'
        
    if args.setting == 'usual':
        # usual non-iid dataset + training
        # as in the original FedAvg paper and fedrep code
        dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
        num_users = len(dict_users_train)
    elif args.setting == 'class_full':
        # class-wise non-iid dataset and full training
        print('class full')
        dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
        if args.grp_class == 0:
            classes = [[0,1],[2,3],[4,5],[6,7],[8,9]]
        elif args.grp_class == 1: 
            classes = [[0,1,2],[2,3,4],[4,5,7],[6,7,8],[8,9,0]]
        else :
            # no structure
            classes = [[00]]
            pass
        if args.grp_class == 0 or args.grp_class == 1: 
            dict_users_train_full, list_users_class_train = sample2(targets_train,classes, 
                                                                    dosort=True,num_users=args.num_users)
            dict_users_test_full, list_users_class_test= sample2(targets_test,classes,num_users=args.num_users,
                                                                class_of_user=list_users_class_train)
        else:
            print("no structure")
            _, _, dict_users_train_full, dict_users_test_full = get_data(args)
            list_users_class_train, list_users_class_test = [],[]
        
        true_num_users = args.num_users

        dict_users_train = dict(dict_users_train_full)
        dict_users_test = dict(dict_users_test_full)
        num_users = len(dict_users_train)
        i_class_clustered = -1
    elif args.setting == 'class_clustered_golden':
        # cluster based on the true class

        dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
        
        if args.grp_class == 0:
            classes = [[0,1],[2,3],[4,5],[6,7],[8,9]]
        elif args.grp_class == 1: 
            classes = [[0,1,2],[2,3,4],[4,5,7],[6,7,8],[8,9,0]]
        else :
            # no structure
            classes = [[00]]
            print('no structure, no golden cluster')
            exit(0)


        dict_users_train_full, list_users_class_train = sample2(targets_train,classes, 
                                                                dosort=True,num_users=args.num_users)
        dict_users_test_full, list_users_class_test= sample2(targets_test,classes,num_users=args.num_users,
                                                            class_of_user=list_users_class_train)
        
        
        # extract clients of the cluster
        i_class_clustered = args.i_class_clustered
        ind_class  = np.where((np.array(list_users_class_train)==i_class_clustered))[0]
        num_users = len(ind_class)
        k =0 
        dict_users_train = {i: np.array([], dtype='int64') for i in range(num_users)}
        dict_users_test = {i: np.array([], dtype='int64') for i in range(num_users)}
        for index in ind_class:
            value = dict_users_train_full[index]
            dict_users_train[k] =  value
            value = dict_users_test_full[index]
            dict_users_test[k] =  value
            k += 1    
        true_num_users = args.num_users
        args.num_users = num_users
        dataset_test = dataset_train
        dict_users_test = dict_users_train        

        
    elif args.setting == 'class_clustered':
        #
        #  main setting that compute and exploits the cluster structure
        #

        dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)

        #
        # type of data and data structure
        if args.grp_class == 0:
            classes = [[0,1],[2,3],[4,5],[6,7],[8,9]]
        elif args.grp_class == 1: 
            classes = [[0,1,2],[2,3,4],[4,5,7],[6,7,8],[8,9,0]]
        else :
            # no structure
            classes = [[00]]
            pass
        if args.grp_class == 0 or args.grp_class == 1: 
            dict_users_train_full, list_users_class_train = sample2(targets_train,classes, 
                                                                    dosort=True,num_users=args.num_users)
            dict_users_test_full, list_users_class_test= sample2(targets_test,classes,num_users=args.num_users,
                                                                class_of_user=list_users_class_train)
        else:
            print("no structure")
            _, _, dict_users_train_full, dict_users_test_full = get_data(args)
            list_users_class_train, list_users_class_test = [],[]

        print(dict_users_train_full)

        # extract matrix of data for each client, compute OTDD and cluster clients
        if args.dataset == 'mnist':
            dim = 784
        elif args.dataset == 'cifar10':
            dim = 3072
        data_list = data_to_list(dataset_train, dict_users_train_full,args,dim=dim)
        namecls = '_'.join([''.join(map(str, sublist)) for sublist in classes])
        true_num_users = args.num_users
        filename = f"{args.dataset:}_distance-{namecls}-num_users-{args.num_users}-n_supp-{args.n_supp}-n_epoch-{args.n_epochs:}-seed-{args.seed}.npy"
        filename_clust = f"{args.dataset:}_distance-{namecls}-num_users-{args.num_users}-n_supp-{args.n_supp}"
        filename_clust += f"-n_epochfedot-{args.n_epochs:}-n_cluster-{args.n_cluster:}-n_neighbour-{args.n_neighbor:}-seed-{args.seed}-clust.npz"
        save_dir = f'./save/fl/{args.dataset:}/distance/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Directory '{save_dir}' created successfully!")


        if os.path.exists(save_dir + filename):
            # Load the NPY file of the distance matrix if it exists
            distance = np.load(save_dir + filename)
            print("File exists and data loaded successfully.")
            if not os.path.exists(save_dir + filename_clust):
                if args.n_neighbor > 1:
                    if args.n_neighbor >= 100:
                        n_neighbors = args.n_neighbor//args.num_users
                    else:
                        n_neighbors = args.n_neighbor
                    clustered_client = cluster_client(distance, n_clusters=args.n_cluster,n_neighbors=n_neighbors)
                else:
                    print('precomputed distance clustering')
                    clustered_client = cluster_client_precomputed(distance, n_clusters=args.n_cluster,n_neighbors=args.n_neighbor)
                np.savez(save_dir + filename_clust, clustered_client = clustered_client, true_cluster=list_users_class_train)
            else:
                res = np.load(save_dir + filename_clust)
                clustered_client = res['clustered_client']
                print("Clustering file exists and data loaded successfully.")
        else:
            # computing distance matrix
            print("Computing distance matrix.")

            distance = compute_otdd(data_list,args,method='fedot')
            np.save(save_dir + filename, distance)
            if args.n_neighbor > 1:
                # approximate sparse graph of precomputed
                if args.n_neighbor > 100:
                    n_neighbors = args.n_neighbor//args.num_users
                else:
                    n_neighbors = args.n_neighbor
                clustered_client = cluster_client(distance, n_clusters=args.n_cluster,n_neighbors=n_neighbors)
            else:
                print('precomputed distance clustering')
                clustered_client = cluster_client_precomputed(distance, n_clusters=args.n_cluster,n_neighbors=args.n_neighbor)
            np.savez(save_dir + filename_clust, clustered_client = clustered_client, true_cluster=list_users_class_train)
        
        print(clustered_client)
        i_class_clustered = args.i_class_clustered
        ind_class  = np.where((np.array(clustered_client)==i_class_clustered))[0]
        num_users = len(ind_class)
        
        k =0 
        dict_users_train = {i: np.array([], dtype='int64') for i in range(num_users)}
        dict_users_test = {i: np.array([], dtype='int64') for i in range(num_users)}

        for index in ind_class:
            value = dict_users_train_full[index]
            dict_users_train[k] =  value
            value = dict_users_test_full[index]
            dict_users_test[k] =  value
            k += 1

        args.num_users = num_users
   
    
    print(dict_users_train,list_users_class_train)
    
    print(args.alg)
    lens = np.ones(num_users)
    namecls = '-'.join([''.join(map(str, sublist)) for sublist in classes])

    # build model
    net_glob = get_model(args)
    net_glob.train()
    if args.load_fed != 'n':
        fed_model_path = './save/' + args.load_fed + '.pt'
        net_glob.load_state_dict(torch.load(fed_model_path))

    total_num_layers = len(net_glob.state_dict().keys())
    print(net_glob.state_dict().keys())
    net_keys = [*net_glob.state_dict().keys()]

    if args.alg == 'fedrep' or args.alg == 'fedper':
        if 'cifar' in  args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0,1,3,4]]
        elif 'mnist' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0,1,2]]

    
    if args.alg == 'fedavg' or args.alg == 'prox' or args.alg == 'maml':
        w_glob_keys = []
    w_glob_keys = [item for sublist in w_glob_keys for item in sublist]

    
    print(total_num_layers)
    print(w_glob_keys)
    print(net_keys)
    net_local_list = []
    w_locals = {}
    for user in range(num_users):
        w_local_dict = {}
        for key in net_glob.state_dict().keys():
            w_local_dict[key] =net_glob.state_dict()[key]
        w_locals[user] = w_local_dict
    #%%
    # training
    indd = None      # indices of embedding for sent140
    loss_train = []
    accs = []
    times = []
    accs10 = 0
    accs10_glob = 0
    start = time.time()
    for iter in range(args.epochs+1):
        w_glob = {}
        loss_locals = []
        m = max(int(args.frac * num_users), 1)
        if iter == args.epochs:
            m = num_users

        idxs_users = np.random.choice(range(num_users), m, replace=False)
        w_keys_epoch = w_glob_keys
        times_in = []
        total_len=0
        for ind, idx in enumerate(idxs_users):
            start_in = time.time()

            if args.epochs == iter:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_ft])
            else:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_tr])

            net_local = copy.deepcopy(net_glob)
            w_local = net_local.state_dict()
            if args.alg != 'fedavg' and args.alg != 'prox':
                for k in w_locals[idx].keys():
                    if k not in w_glob_keys:
                        w_local[k] = w_locals[idx][k]
            net_local.load_state_dict(w_local)
            last = iter == args.epochs

            w_local, loss, indd = local.train(net=net_local.to(args.device), idx=idx, w_glob_keys=w_glob_keys, lr=args.lr, last=last)
            loss_locals.append(copy.deepcopy(loss))
            total_len += lens[idx]
            if len(w_glob) == 0:
                w_glob = copy.deepcopy(w_local)
                for k,key in enumerate(net_glob.state_dict().keys()):
                    w_glob[key] = w_glob[key]*lens[idx]
                    w_locals[idx][key] = w_local[key]
            else:
                for k,key in enumerate(net_glob.state_dict().keys()):
                    if key in w_glob_keys:
                        w_glob[key] += w_local[key]*lens[idx]
                    else:
                        w_glob[key] += w_local[key]*lens[idx]
                    w_locals[idx][key] = w_local[key]

            times_in.append( time.time() - start_in )
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        # get weighted average for global weights
        for k in net_glob.state_dict().keys():
            w_glob[k] = torch.div(w_glob[k], total_len)

        w_local = net_glob.state_dict()
        for k in w_glob.keys():
            w_local[k] = w_glob[k]
        if args.epochs != iter:

            net_glob.load_state_dict(w_glob)

        if iter % args.test_freq==args.test_freq-1 or iter>=args.epochs-10:
            if times == []:
                times.append(max(times_in))
            else:
                times.append(times[-1] + max(times_in))
            if args.alg != 'fedavg' and args.alg != 'prox':
                #
                # Perf for local models
                acc_test, loss_test = test_img_local_all(net_glob, args, dataset_test, dict_users_test,
                                                            w_glob_keys=w_glob_keys, w_locals=w_locals,indd=indd,dataset_train=dataset_train, dict_users_train=dict_users_train, return_all=False)
                accs.append(acc_test)
                # for algs which learn a single global model, these are the local accuracies (computed using the locally updated versions of the global model at the end of each round)
                if iter != args.epochs:
                    print('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Local Test accuracy: {:.2f}'.format(
                            iter, loss_avg, loss_test, acc_test))
                else:
                    # in the final round, we sample all users, and for the algs which learn a single global model, we fine-tune the head for 10 local epochs for fair comparison with FedRep
                    print('Final Round, Train loss: {:.3f}, Test loss: {:.3f}, Local Test accuracy: {:.2f}'.format(
                            loss_avg, loss_test, acc_test))
                if iter >= args.epochs-10 and iter != args.epochs:
                    accs10 += acc_test/10

            #------------------------------------------------------------------------------
            # below prints the global accuracy of the single global model for the relevant algs
            #-----------------------------------------------------------------------------------
            if args.alg == 'fedavg' or args.alg == 'prox':
                acc_test, loss_test = test_img_local_all(net_glob, args, dataset_test, dict_users_test,
                                                        w_locals=None,indd=indd,dataset_train=dataset_train, dict_users_train=dict_users_train, return_all=False)
                if iter != args.epochs:
                    print('Round {:3d}, Global train loss: {:.3f}, Global test loss: {:.3f}, Global test accuracy: {:.2f}'.format(
                        iter, loss_avg, loss_test, acc_test))
                else:
                    print('Final Round, Global train loss: {:.3f}, Global test loss: {:.3f}, Global test accuracy: {:.2f}'.format(
                        loss_avg, loss_test, acc_test))
                # aggregating the perfomance of the global model
                accs.append(acc_test)

            if iter >= args.epochs-10 and iter != args.epochs:
                accs10_glob += acc_test/10

        if iter % args.save_every==args.save_every-1:
            model_save_path = './save/accs_'+ args.alg + '_' + args.dataset + '_' + str(args.num_users) +'_'+ str(args.shard_per_user) +'_iter' + str(iter)+ '.pt'
            torch.save(net_glob.state_dict(), model_save_path)

    
    #print('NO SAVING')
    if args.alg == 'fedavg' or args.alg == 'prox':
        print('Average global accuracy final 10 rounds: {}'.format(accs10_glob))
    end = time.time()
    print(end-start)
    print(times)
    print(accs)
    
    
    save_dir = f'./save/fl/{args.dataset:}/{args.alg}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Directory '{save_dir}' created successfully!")
        
    base_dir = f"{save_dir:}fl-{args.alg}-data-{args.dataset}-class-{namecls}-num_users-{true_num_users}"
    if args.setting == 'class_clustered':
        base_dir += f"-n_supp-{args.n_supp}-n_epochfedot-{args.n_epochs:}-n_cluster-{args.n_cluster:}-n_neighbour-{args.n_neighbor:}"
    base_dir += f'-setting-{args.setting}-i_class-{i_class_clustered:}-seed-{args.seed}.csv'

    
    
    # user_save_path = base_dir
    # accs = np.array(accs)
    # accs = pd.DataFrame(accs, columns=['accs'])
    # accs.to_csv(base_dir, index=False)

# %%
# %%
