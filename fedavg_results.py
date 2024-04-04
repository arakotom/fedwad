#%%
"""
This file is used to print the results of the experiments in the paper. 
it reads from the saved files the results of the experiments and print them in a latex format.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd

import os
import argparse

parser = argparse.ArgumentParser(description="OTDD FL")

path_file = './save/fl/'
sys.argv = ['']

parser.add_argument("--setting", type=int, default=1)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--n_supp", type=int, default=100)
parser.add_argument("--num_users", type=int, default=40)
parser.add_argument("--n_cluster", type=int, default=5)
parser.add_argument("--n_neighbor", type=int, default=1)
parser.add_argument("--grp_class", type=int, default=2)
parser.add_argument("--data", type=str, default='cifar10', help="data")

args = parser.parse_args()
dataset = args.data
n_epochs = args.n_epochs
n_supp = args.n_supp
num_users = args.num_users
n_cluster = args.n_cluster
n_neighbor = args.n_neighbor
if args.grp_class == 0:
    classes = [[0,1],[2,3],[4,5],[6,7],[8,9]]
elif args.grp_class == 1:
    classes = [[0,1,2],[2,3,4],[4,5,7],[6,7,8],[8,9,0]]
elif args.grp_class == 2:
    classes = [[0]]

namecls = '-'.join([''.join(map(str, sublist)) for sublist in classes])

if args.setting == 0:
    algo = ['fedavg','fedavg','fedavg']
    setting_list = ['class_full', 'class_clustered','class_clustered_golden']
else:
    algo = ['fedavg','fedavg','fedrep','fedrep','fedper','fedper']
    setting_list = ['class_full','class_clustered','class_full','class_clustered','class_full','class_clustered']
path_file = './save/fl/'
nb_seed = 5

list_missing = []
for num_users in [10,20,40,100]:
    for n_supp in [n_supp]:
        for n_neighbor in [1,3,5]:# 1 use a spectral clustering method that dont need it. see doc of sklearn 
            mat_result = np.zeros((nb_seed,len(algo)))
            for seed in range(nb_seed):
                for i,alg in enumerate(algo):
                    setting = setting_list[i] 
                    path_file = f'./save/fl/{dataset:}/{alg}/'

                    result = 0
                    if setting == 'class_full':
                        i_class = -1
                        filename = f"fl-{alg}-data-{args.data}-class-{namecls}-num_users-{num_users}-setting-{setting:}-i_class-{i_class:}-seed-{seed:}.csv"
                        try:
                            df = pd.read_csv(path_file + filename)
                            result = df.iloc[-1].item()
                            #print(result)
                            mat_result[seed,i] = result
                        except:
                            list_missing.append(filename)
                    elif setting == 'class_clustered_golden':
                        k = 0
                        for i_class_clustered in range(n_cluster):
                            filename = f"fl-{alg}-data-{args.data}-class-{namecls}-num_users-{num_users}-setting-{setting:}-i_class-{i_class_clustered:}-seed-{seed:}.csv"

                            try:
                                df = pd.read_csv(path_file + filename)
                                result += df.iloc[-1].item()
                                k+=1
                            except:
                                list_missing.append(filename)
                                
                        if k > 0:
                            #print(result/k)
                            mat_result[seed,i] = result/k
                        else:
                            mat_result[seed,i] = np.nan
                    elif setting == 'class_clustered':
                        namecls = '-'.join([''.join(map(str, sublist)) for sublist in classes])
                        #filename = f"mnist_distance-{namecls}-num_users-{num_users}-n_supp-{n_supp}-n_epoch-{n_epochs:}-seed-{seed}.npy"
                        #filename_clust = f"mnist_distance-{namecls}-num_users-{num_users}-n_supp-{n_supp}-n_epoch-{n_epochs:}-seed-{seed}-clust.npy"

                        k = 0
                        for i_class_clustered in range(n_cluster):
                            filename = f"fl-{alg}-data-{args.data}-class-{namecls}-num_users-{num_users}"
                            filename += f"-n_supp-{n_supp}-n_epochfedot-{n_epochs:}-n_cluster-{n_cluster:}-n_neighbour-{n_neighbor:}"
                            filename += f'-setting-{setting}-i_class-{i_class_clustered:}-seed-{seed}.csv'
                            
                            try:
                                df = pd.read_csv(path_file + filename) 
                                result += df.iloc[-1].item()
                                k +=1
                            except:
                                list_missing.append(filename)

                        if k > 0:
                            #print(result/k)
                            mat_result[seed,i] = result/k
                        else:
                            mat_result[seed,i] = np.nan
                        

            resultat = f"{num_users:} {n_supp:} {n_neighbor:}"
            mean_result = np.nanmean(mat_result,axis=0)
            std_result = np.nanstd(mat_result,axis=0)
            if n_neighbor == 1:
                print(f"{num_users:} users \\\\")
                text_clust = 'aff'
            else: 
                text_clust = f"spg-{n_neighbor:}"   
            resultat = text_clust
            for i in range(len(algo)): 
                nb_computed = len(np.where(np.isnan(mat_result[:,i]))[0])
                #resultat += f"| {mean_result[i]:2.2f} $\pm$ {std_result[i]:2.2f}  ({nb_computed:})"
                resultat += f"& {mean_result[i]:2.2f} $\pm$ {std_result[i]:2.2f} "
                if i % 2 == 1:
                    if i == len(algo)-1:
                        resultat += f"\\\\"
                    else:
                        resultat += f"&"
            
            
            
            print(resultat) 


for missing in list_missing:
    print(missing)


# %%---------------------------------------------------------------------------
#
#           PRINT results for the paper the appendix part
#           LINE algorithm and nb of users in line 
#           COLUMN setting and support size
#----------------------------------------------------------------------------
algo = ['fedavg','fedrep','fedper']
setting_list = ['class_full', 'class_clustered', 'class_clustered', 'class_clustered','class_clustered', 'class_clustered', 'class_clustered']
n_neighbor_list = [0,1,1,3,3,5,5]
n_support_list = [10,10,100,10,100,10,100 ]
list_missing = []

if args.grp_class == 0:
    classes = [[0,1],[2,3],[4,5],[6,7],[8,9]]
elif args.grp_class == 1:
    classes = [[0,1,2],[2,3,4],[4,5,7],[6,7,8],[8,9,0]]
elif args.grp_class == 2:
    classes = [[0]]

full_result = []
for alg in algo:
    text= f"{alg:}\\\\"
    print(text) 
    for num_users in [10, 20, 40, 100]:
        for n_supp in [n_supp]:
            mat_result = np.zeros((nb_seed,len(setting_list)))
            for seed in range(nb_seed):
                for i, setting in enumerate(setting_list):
                    n_neighbor = n_neighbor_list[i]
                    n_supp = n_support_list[i]
                    path_file = f'./save/fl/{dataset:}/{alg}/'

                    result = 0
                    if setting == 'class_full':
                        i_class = -1
                        filename = f"fl-{alg}-data-{args.data}-class-{namecls}-num_users-{num_users}-setting-{setting:}-i_class-{i_class:}-seed-{seed:}.csv"
                        try:
                            df = pd.read_csv(path_file + filename)
                            result = df.iloc[-1].item()
                            #print(result)
                            mat_result[seed,i] = result
                        except:
                            list_missing.append(filename)
                    elif setting == 'class_clustered_golden':
                        k = 0
                        for i_class_clustered in range(n_cluster):
                            filename = f"fl-{alg}-data-{args.data}-class-{namecls}-num_users-{num_users}-setting-{setting:}-i_class-{i_class_clustered:}-seed-{seed:}.csv"

                            try:
                                df = pd.read_csv(path_file + filename)
                                result += df.iloc[-1].item()
                                k+=1
                            except:
                                list_missing.append(filename)
                                
                        if k > 0:
                            #print(result/k)
                            mat_result[seed,i] = result/k
                        else:
                            mat_result[seed,i] = np.nan
                    elif setting == 'class_clustered':
                        namecls = '-'.join([''.join(map(str, sublist)) for sublist in classes])
                        #filename = f"mnist_distance-{namecls}-num_users-{num_users}-n_supp-{n_supp}-n_epoch-{n_epochs:}-seed-{seed}.npy"
                        #filename_clust = f"mnist_distance-{namecls}-num_users-{num_users}-n_supp-{n_supp}-n_epoch-{n_epochs:}-seed-{seed}-clust.npy"

                        k = 0
                        for i_class_clustered in range(n_cluster):
                            filename = f"fl-{alg}-data-{args.data}-class-{namecls}-num_users-{num_users}"
                            filename += f"-n_supp-{n_supp}-n_epochfedot-{n_epochs:}-n_cluster-{n_cluster:}-n_neighbour-{n_neighbor:}"
                            filename += f'-setting-{setting}-i_class-{i_class_clustered:}-seed-{seed}.csv'
                            
                            #try:
                            df = pd.read_csv(path_file + filename) 
                            result += df.iloc[-1].item()
                            k +=1
                            #except:
                                #list_missing.append(filename)

                        if k > 0:
                            #print(result/k)
                            mat_result[seed,i] = result/k
                        else:
                            mat_result[seed,i] = np.nan
                                

                
                
                
                #print(resultat) 
        #print(num_users)
        # mean_result = np.nanmean(mat_result,axis=0)
        # std_result = np.nanstd(mat_result,axis=0)
        # resultat = f"{num_users:}"           
        # for kk in range(len(setting_list)): 
        #     #nb_computed = len(np.where(np.isnan(mat_result[:,i]))[0])
        #     #resultat += f"| {mean_result[i]:2.2f} $\pm$ {std_result[i]:2.2f}  ({nb_computed:})"
        #     resultat += f"& {mean_result[kk]:2.1f} $\pm$ {std_result[kk]:2.1f} "
        #     if kk == len(setting_list)-1:
        #         resultat += f"\\\\"
                
        # print(resultat)
    
    
        mean_result = np.nanmean(mat_result,axis=0)
        std_result = np.nanstd(mat_result,axis=0)
        aux_part1 = np.round(mean_result*10)
        ind_max = np.argmax(aux_part1)
        list_max = list(np.where(aux_part1 == aux_part1[ind_max])[0])


        resultat = f"{num_users:}"           
        for kk in range(len(setting_list)): 
            if kk in list_max:
                resultat += f"& \\textbf{{{mean_result[kk]:2.1f}}} $\pm$ \\textbf{{{std_result[kk]:2.1f}}} "    
            else:
                resultat += f"& {mean_result[kk]:2.1f} $\pm$ {std_result[kk]:2.1f} "
            if kk == len(setting_list)-1:
                resultat += f"\\\\"
        print(resultat)
        full_result.append(mean_result)
full_result = np.array(full_result)
full_result_part1 = full_result
full_result_part1 = full_result_part1 - full_result_part1[:,0].reshape(-1,1)
aver_p1 = full_result_part1.mean(axis=0)
average_text = "average uplift"

for i in range(len(aver_p1)):
    #print(f"& {aver_p1[i]:2.1f} $\pm$ {full_result_part1.std(axis=0)[i]:2.1f} ")
    if i == 0:
        average_text += f"& - "
    else:
        average_text += f"& {aver_p1[i]:2.1f} $\pm$ {full_result_part1.std(axis=0)[i]:2.1f}"
# average_text += f"&"
# for i in range(len(aver_p2)):
#     #print(f"& {aver_p2[i]:2.1f} $\pm$ {full_result_part2.std(axis=0)[i]:2.1f} ")
#     if i == 0:
#         average_text += f"& - "
#     else:
#         average_text += f"& {aver_p2[i]:2.1f}  $\pm$ {full_result_part2.std(axis=0)[i]:2.1f}"
average_text += "\\\\"
print(average_text)


#for missing in list_missing:
#    print(missing)



# 
# %%---------------------------------------------------------------------------
#
#           PRINT results for the main paper 
#           LINE algorithm and nb of users in line 
#           COLUMN setting and support size = 10 + group classes 
#----------------------------------------------------------------------------

algo = ['fedavg','fedrep','fedper']
setting_list = ['class_full', 'class_clustered', 'class_clustered', 'class_clustered','class_full','class_clustered', 'class_clustered', 'class_clustered']
n_neighbor_list = [0,1,3,5,0,1,3,5]
n_support_list = [10,10,10,10,10,10,10,10 ]
n_grp_classes = [0,0,0,0,2,2,2, 2]
list_missing = []
list_classes = [[[0,1],[2,3],[4,5],[6,7],[8,9]], [[0,1,2],[2,3,4],[4,5,7],[6,7,8],[8,9,0]],
 [[0]]]

full_result = []
for alg in algo:
    text= f"{alg:}\\\\"
    print(text) 
    for num_users in [20, 40, 100]:
        mat_result = np.zeros((nb_seed,len(setting_list)))
        for seed in range(nb_seed):
            for i, setting in enumerate(setting_list):
                n_neighbor = n_neighbor_list[i]
                n_supp = n_support_list[i]
                classes = list_classes[n_grp_classes[i]]
                namecls = '-'.join([''.join(map(str, sublist)) for sublist in classes])

                path_file = f'./save/fl/{dataset:}/{alg}/'

                result = 0
                if setting == 'class_full':
                    i_class = -1
                    filename = f"fl-{alg}-data-{args.data}-class-{namecls}-num_users-{num_users}-setting-{setting:}-i_class-{i_class:}-seed-{seed:}.csv"
                    try:
                        df = pd.read_csv(path_file + filename)
                        result = df.iloc[-1].item()
                        #print(result)
                        mat_result[seed,i] = result
                    except:
                        list_missing.append(filename)
                elif setting == 'class_clustered_golden':
                    k = 0
                    for i_class_clustered in range(n_cluster):
                        filename = f"fl-{alg}-data-{args.data}-class-{namecls}-num_users-{num_users}-setting-{setting:}-i_class-{i_class_clustered:}-seed-{seed:}.csv"

                        try:
                            df = pd.read_csv(path_file + filename)
                            result += df.iloc[-1].item()
                            k+=1
                        except:
                            list_missing.append(filename)
                            
                    if k > 0:
                        #print(result/k)
                        mat_result[seed,i] = result/k
                    else:
                        mat_result[seed,i] = np.nan
                elif setting == 'class_clustered':
                    namecls = '-'.join([''.join(map(str, sublist)) for sublist in classes])
                    #filename = f"mnist_distance-{namecls}-num_users-{num_users}-n_supp-{n_supp}-n_epoch-{n_epochs:}-seed-{seed}.npy"
                    #filename_clust = f"mnist_distance-{namecls}-num_users-{num_users}-n_supp-{n_supp}-n_epoch-{n_epochs:}-seed-{seed}-clust.npy"

                    k = 0
                    for i_class_clustered in range(n_cluster):
                        filename = f"fl-{alg}-data-{args.data}-class-{namecls}-num_users-{num_users}"
                        filename += f"-n_supp-{n_supp}-n_epochfedot-{n_epochs:}-n_cluster-{n_cluster:}-n_neighbour-{n_neighbor:}"
                        filename += f'-setting-{setting}-i_class-{i_class_clustered:}-seed-{seed}.csv'
                        
                        #try:
                        df = pd.read_csv(path_file + filename) 
                        result += df.iloc[-1].item()
                        k +=1
                        #except:
                            #list_missing.append(filename)

                    if k > 0:
                        #print(result/k)
                        mat_result[seed,i] = result/k
                    else:
                        mat_result[seed,i] = np.nan
                                

                
                
                
                #print(resultat) 
        #print(num_users)
        part1 = [0,1,2,3]
        part2 = [4,5,6,7]
        mean_result = np.nanmean(mat_result,axis=0)
        std_result = np.nanstd(mat_result,axis=0)
        aux_part1 = np.round(mean_result[part1]*10)
        ind_max = np.argmax(aux_part1)
        list_max = list(np.where(aux_part1 == aux_part1[ind_max])[0])
        aux_part2 = np.round(mean_result[part2]*10)
        ind_max = np.argmax(aux_part2)
        list_max += list(np.where(aux_part2 == aux_part2[ind_max])[0] + +len(part1))

        resultat = f"{num_users:}"           
        for kk in range(len(setting_list)): 
            if kk in list_max:
                resultat += f"& \\textbf{{{mean_result[kk]:2.1f}}} $\pm$ \\textbf{{{std_result[kk]:2.1f}}} "    
            else:
                resultat += f"& {mean_result[kk]:2.1f} $\pm$ {std_result[kk]:2.1f} "
            if kk == len(setting_list)-1:
                resultat += f"\\\\"
            elif kk == 3:
                resultat += f"&"
        print(resultat)
        full_result.append(mean_result)
full_result = np.array(full_result)
full_result_part1 = full_result[:,part1] 
full_result_part2 = full_result[:,part2]
full_result_part1 = full_result_part1 - full_result_part1[:,0].reshape(-1,1)
full_result_part2 =  full_result_part2 - full_result_part2[:,0].reshape(-1,1)
aver_p1 = full_result_part1.mean(axis=0)
aver_p2 = full_result_part2.mean(axis=0)
average_text = "average uplift"

for i in range(len(aver_p1)):
    #print(f"& {aver_p1[i]:2.1f} $\pm$ {full_result_part1.std(axis=0)[i]:2.1f} ")
    if i == 0:
        average_text += f"& - "
    else:
        average_text += f"& {aver_p1[i]:2.1f} $\pm$ {full_result_part1.std(axis=0)[i]:2.1f}"
average_text += f"&"
for i in range(len(aver_p2)):
    #print(f"& {aver_p2[i]:2.1f} $\pm$ {full_result_part2.std(axis=0)[i]:2.1f} ")
    if i == 0:
        average_text += f"& - "
    else:
        average_text += f"& {aver_p2[i]:2.1f}  $\pm$ {full_result_part2.std(axis=0)[i]:2.1f}"
     
print(average_text)
#for missing in list_missing:
#    print(missing)
# %%
