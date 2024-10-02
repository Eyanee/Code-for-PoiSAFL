#The codes are based on Ubuntu 16.04 with Python 3.7 and Pytorch 1.0.1

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
# from visualdl import LogWriter

import torch

from update import LocalUpdate, test_inference, DatasetSplit
from poison_optimization_test import Outline_Poisoning, add_small_perturbation,cal_ref_distance,model_dist_norm,cal_similarity
from model import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, VGGCifar,GTSRBNet
from resnet import *
from utils1 import *
from added_funcs import poison_Mean
import csv
from torch.utils.data import DataLoader, Dataset
from options import args_parser
import os
from otherGroupingMethod import *
from otherPoisoningMethod import *
import  gc
gc.collect()
torch.cuda.empty_cache()


# For experiments with only stragglers
# For experiments with both stragglers and adversaries


if __name__ == '__main__':


    args = args_parser()

    start_time = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    path_project = os.path.abspath('..')

    exp_details(args)

    gpu_number = args.gpu_number
    device = torch.device(f'cuda:{gpu_number}' if args.gpu else 'cpu')

    # writer = LogWriter(logdir="./log/histogram_test/async_res_noattack_3")

    train_dataset, test_dataset, (user_groups, dict_common) = get_dataset(args) 

    if args.model == 'cnn':
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'GTSRB':
            global_model = GTSRBNet(args=args)
        elif args.dataset == 'cifar':

            if args.detail_model == 'simplecnn':
                global_model = CNNCifar(args=args)
            elif args.detail_model == 'vgg':
                global_model = VGGCifar()
            elif args.detail_model == 'resnet':
                global_model = ResNet18()

    elif args.model == 'MLP':
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)

    else:
        exit('Error: unrecognized model')


    global_model.to(device)

    # params = torch.load('./global_model_parameters.pth')  
    # params = torch.load('./fmnist_Trimean_100_noniid_model_parameters_new_0.1.pth')
    # params = torch.load('./cifar_iid_200_model_parameters.pth')   
    


    # global_model.load_state_dict(params)
    
    global_model.train()
    
    fi_global_model = copy.deepcopy(global_model)
    pre_global_model = copy.deepcopy(global_model)
    primitive_malicious = copy.deepcopy(global_model.state_dict())

    global_weights = global_model.state_dict()

    train_accuracy = []
    final_test_acc = []
    print_every = 1

    pre_weights = {} 
    pre_indexes = {}
    pre_grad = {}
    pre_loss = {}
    for i in range(args.staleness + 1): 
        if i != 0:
            pre_weights[i] = []
            pre_indexes[i] = []
            pre_grad[i] = []
            pre_loss[i] = []


    scheduler = {}
    

    clientStaleness = {}
    
    TARGET_STALENESS  = 1

    poisoned = False

    distance_ratio = 1
    adaptive_accuracy_threshold = 0.8 
    pinned_accuracy_threshold = 0.8


 
    for l in range(args.num_users):
        scheduler[l] = 0
        clientStaleness[l] = 0

    global_epoch = 0
    
    all_users = np.arange(args.num_users)
    m = int(args.num_users * args.attack_ratio)
    n = args.num_users - m
    attack_users = all_users[-m:]
    print("attack user num is ",m)
    
    t = int(math.ceil(n/args.staleness))
    print(" t is ",t )
    
    for i in range(args.staleness):
        if i == args.staleness +1:
            front_idx = int(t * i)
            end_idx = n-  1
        else:
            front_idx = int(t * i)
            end_idx = front_idx + t
        for j in range(front_idx, end_idx):
            clientStaleness[j] = i + 1 
    print("attack_user is", attack_users)

    for count, l in enumerate(attack_users):
        
        clientStaleness[l] = count%6+1
        
    # 恶意用户的历史存储
    MAX_STALENESS = args.staleness
    mal_parameters_list = {}
    mal_grad_list = {}
    for i in range(MAX_STALENESS):
        mal_parameters_list[i] = []
        mal_grad_list[i]= []
    
    

    
    
    std_keys = get_key_list(global_model.state_dict().keys())

    mal_rand = global_model.state_dict()
    
            
    for epoch in tqdm(range(args.epochs)):

        local_weights_delay = {}
        loss_on_public = {}
        entropy_on_public = {}
        local_index_delay = {}
        local_grad_delay = {}
        malicious_models = []
        grad_list = []

        for i in range(args.staleness + 1):
            loss_on_public[i] = []
            entropy_on_public[i] = []
            local_weights_delay[i] = []
            local_index_delay[i] = []
            local_grad_delay[i] = []


        print(f'\n | Global Training Round : {epoch + 1} | \n')

        global_model.train()

        global_weights_rep = copy.deepcopy(global_model.state_dict())
        

        # After round, each staleness group is adjusted
        local_delay_ew = copy.deepcopy(pre_weights[1])
        local_index_ew = copy.deepcopy(pre_indexes[1])
        local_delay_gd = copy.deepcopy(pre_grad[1])
        local_delay_loss = copy.deepcopy(pre_loss[1])

        for i in range(args.staleness):
            if i != 0:
                pre_weights[i] = copy.deepcopy(pre_weights[i+1])
                pre_indexes[i] = copy.deepcopy(pre_indexes[i+1])
                pre_grad[i] = copy.deepcopy(pre_grad[i + 1])
                pre_loss[i] = copy.deepcopy(pre_loss[i + 1])

        pre_weights[args.staleness] = [] # 对staleness的权重
        pre_indexes[args.staleness] = []
        pre_grad[args.staleness] = []
        pre_loss[args.staleness] = []

        ensure_1 = 0
        count = 0 
        for idx in all_users:

            if scheduler[idx] == 0:
                if idx in attack_users and args.data_poison == True:
                    local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], idx=idx,
                                              data_poison=True)

                else:
                    local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], idx=idx,
                                              data_poison=False)

                scheduler[idx] = clientStaleness[idx]  
                print("current submit client idx and staleness is ", idx ,scheduler[idx])

                

            else:

                continue

            

            w, loss,gd = local_model.update_weights(

                model=copy.deepcopy(global_model), global_round=epoch

            )
            
            ensure_1 += 1
            
            if idx in attack_users and args.model_poison == True and epoch >=0:
                mal_parameters_list[scheduler[idx] -1].append(w)
                mal_grad_list[scheduler[idx] -1].append(gd)
            
            elif  idx in attack_users and args.new_poison == True and epoch >=0:
                mal_dict = sign_attack(w, args.model_poison_scale)

                test_model = copy.deepcopy(global_model)
                test_model.load_state_dict(mal_dict)

                mal_grad = compute_gradient(mal_dict,global_model.state_dict(),std_keys,args.lr)

                

                common_acc, common_loss_sync, common_entropy_sample = test_inference(args, test_model,
                                                                                    DatasetSplit(train_dataset,
                                                                                        dict_common))
                local_weights_delay[ scheduler[idx] - 1 ].append(copy.deepcopy(mal_dict))
                local_index_delay[ scheduler[idx] - 1 ].append(idx)
                local_grad_delay[scheduler[idx] - 1].append(copy.deepcopy(mal_grad))
                loss_on_public[scheduler[idx] - 1].append(common_loss_sync)
                entropy_on_public[scheduler[idx] - 1].append(common_entropy_sample)
            else:     
                test_model = copy.deepcopy(global_model)
                test_model.load_state_dict(w)

                common_acc, common_loss_sync, common_entropy_sample = test_inference(args, test_model,
                                                                                    DatasetSplit(train_dataset,
                                                                                        dict_common))

                local_weights_delay[ scheduler[idx] - 1 ].append(copy.deepcopy(w))
                local_index_delay[ scheduler[idx] - 1 ].append(idx)
                local_grad_delay[scheduler[idx] - 1].append(copy.deepcopy(gd))
                loss_on_public[scheduler[idx] - 1].append(common_loss_sync)
                entropy_on_public[scheduler[idx] - 1].append(common_entropy_sample)
                benign_distance = model_dist_norm(w,copy.deepcopy(global_model.state_dict()))
                
                
        

                
            
        if args.model_poison == True and epoch >=0 :
            if args.poison_methods == 'LA':
                mal_len = len(mal_grad_list[0])
                malicious_dicts= LA_attack(args, mal_grad_list[0], mal_len,copy.deepcopy(global_model),std_keys)
                test_weights = copy.deepcopy(global_model.state_dict())
                test_model.load_state_dict(global_weights)
                optimizer_fed = Adam(test_model.parameters(), lr=0.01)
                for mal_vec in malicious_dicts:
                    test_model.load_state_dict(global_weights)
                    optimizer_fed.step(mal_vec)
                    mal_acc, mal_loss_sync, mal_entropy_sample = test_inference(args, test_model,
                                                                                                DatasetSplit(train_dataset,
                                                                                                    dict_common))
            
                    local_weights_delay[ 0].append(copy.deepcopy(test_model.state_dict()))
                    local_index_delay[0].append(idx)
                    loss_on_public[0].append(mal_loss_sync)
                    entropy_on_public[0].append(mal_entropy_sample)
                    #
                    local_grad_delay[0].append(copy.deepcopy(mal_vec))
                    test_model.load_state_dict(global_weights)


            else:
                

                if args.poison_methods == 'LIE':
                    test_model.load_state_dict(global_weights)
                    optimizer_fed = Adam(test_model.parameters(), lr=0.01)
                    print("len of [0] is ", len(mal_parameters_list[0]))
                    malicious_dict = LIE_attack(mal_grad_list[0])
                    optimizer_fed.step(malicious_dict)
            
                elif args.poison_methods == 'min_sum':
                    test_model.load_state_dict(global_weights)
                    optimizer_fed = Adam(test_model.parameters(), lr=0.01)
                    malicious_grads = min_sum(args, mal_grad_list[0])
                    model_grads=[]
                    start_idx = 0

                    optimizer_fed.step(malicious_grads)
                elif args.poison_methods == 'Grad':

                    test_model.load_state_dict(global_weights)
                    optimizer_fed = Adam(test_model.parameters(), lr=0.01)
                    len_mal = int(len(mal_grad_list[0])/args.attack_ratio)

                    malicious_dict = Grad_median(args,mal_grad_list[0], len_mal,std_keys)


                    optimizer_fed.step(malicious_dict)
                    
                
                for idx in range(len(mal_grad_list[0])) :

                    mal_acc, mal_loss_sync, mal_entropy_sample = test_inference(args, test_model,
                                                                                                DatasetSplit(train_dataset,
                                                                                                    dict_common))
                    malicious_dict = copy.deepcopy(test_model.state_dict())
                    
                    # print("mal_idx is", mal_acc)
                    local_weights_delay[ 0].append(malicious_dict)
                    local_index_delay[0].append(idx)
                    loss_on_public[0].append(mal_loss_sync)
                    entropy_on_public[0].append(mal_entropy_sample)
            
                    local_grad_delay[0].append(copy.deepcopy(malicious_dict))
                
                        

        for i in range(args.staleness):
            if i != 0:
                if args.update_rule == 'Sageflow':
                    if len(local_weights_delay[i]) > 0:
                        w_avg_delay, len_delay = Eflow(local_weights_delay[i], loss_on_public[i], entropy_on_public[i], epoch)
                        
                        
                        test_model.load_state_dict(w_avg_delay)
                        mal_acc, mal_loss_sync, mal_entropy_sample = test_inference(args, test_model,
                                                                                                    DatasetSplit(train_dataset,
                                                                                                        dict_common))
                    
                        pre_weights[i].append({epoch: [w_avg_delay, len_delay]})
                elif args.update_rule == 'Median':
                    if len(local_weights_delay[i]) > 0:
                        pre_weights[i].append(local_weights_delay[i])

                elif args.update_rule == 'Trimmed_mean':
                    if len(local_weights_delay[i]) > 0:
                        pre_weights[i].append(local_weights_delay[i])
                elif args.update_rule == 'norm_bounding':
                    if len(local_weights_delay[i]) > 0:
                        pre_weights[i].append(local_weights_delay[i])
                elif args.update_rule == 'AFLGuard':
                    if len(local_weights_delay[i]) > 0:
                        pre_weights[i].append(local_weights_delay[i])
                elif args.update_rule == 'Zenoplusplus':
                    if len(local_weights_delay[i]) > 0:
                        pre_weights[i].append(local_weights_delay[i])
                        pre_indexes[i].append(local_index_delay[i])
                        pre_grad[i].append(local_grad_delay[i])
                elif args.update_rule == 'FLARE':
                    if len(local_weights_delay[i]) > 0:
                        pre_weights[i].append(local_weights_delay[i])
                elif args.update_rule == 'LFR':
                    if len(local_weights_delay[i]) > 0:
                        pre_weights[i].append(local_weights_delay[i])
                        pre_indexes[i].append(local_index_delay[i])
                elif args.update_rule == 'Krum':
                    if len(local_weights_delay[i]) > 0:
                        pre_weights[i].append(local_weights_delay[i])
                else:

                    if len(local_weights_delay[i]) > 0:
                        pre_weights[i].append(local_weights_delay[i])
                        
        if args.update_rule == 'Sageflow':
            sync_weights, len_sync = Eflow(local_weights_delay[0], loss_on_public[0], entropy_on_public[0], epoch)
            # Staleness-aware grouping
            
            test_model.load_state_dict(sync_weights)
            mal_acc, mal_loss_sync, mal_entropy_sample = test_inference(args, test_model,
                                                                                        DatasetSplit(train_dataset,
                                                                                            dict_common))
            global_weights = Sag(epoch, sync_weights, len_sync, local_delay_ew,
                                                     copy.deepcopy(global_weights))
        elif args.update_rule == 'Median':
            std_keys = get_key_list(global_model.state_dict().keys())
            current_param = []
            current_param = copy.deepcopy(local_weights_delay[0])
            
            for k in local_delay_ew:
                current_param.extend(k)
            sync_weights, len_sync = pre_Median(copy.deepcopy(global_model.state_dict()), std_keys, current_param)
            global_weights = update_weights(global_model.state_dict(),sync_weights)


        elif args.update_rule == 'Trimmed_mean': 
            std_keys = get_key_list(global_model.state_dict().keys())
            current_param = []
            current_param = copy.deepcopy(local_weights_delay[0])
            
            for k in local_delay_ew:
                current_param.extend(k)
            sync_weights, len_sync = pre_Trimmed_mean(args,copy.deepcopy(global_model.state_dict()), std_keys, current_param)
            global_weights = update_weights(global_model.state_dict(),sync_weights)
        elif args.update_rule == 'norm_bounding':
            avg_weights = norm_clipping(global_model, local_weights_delay[0],local_delay_ew,std_keys,  args.lr)
            global_weights = update_weights(global_model.state_dict(), avg_weights)
        elif args.update_rule == 'AFLGuard':
            current_param = []
            global_test_model = LocalUpdate(args=args, dataset=train_dataset, idxs=dict_common, idx=idx,
                                              data_poison=False, delay = False)
            current_param = copy.deepcopy(local_weights_delay[0])
            
            for k in local_delay_ew:
                current_param.extend(k)
            # print("len current param", len(current_param))
            global_weights = AFLGuard(current_param, global_model, global_test_model, epoch, std_keys, args.lr, lamda = 1.8 )
        
        elif args.update_rule == 'Zenoplusplus':
            # current_grad = copy.deepcopy(local_grad_delay[0])
            current_param = copy.deepcopy(local_weights_delay[0])
            current_index = copy.deepcopy(local_index_delay[0])
            global_test_model = LocalUpdate(args=args, dataset=train_dataset, idxs=dict_common, idx=idx,
                                              data_poison=False)
            for item in local_delay_ew:
                current_param.extend(item)
            for item  in local_index_ew:
                current_index.extend(item)
            global_param_update = update_weights_zeno(args,copy.deepcopy(global_model),epoch,DatasetSplit(train_dataset,
                                                                                                        dict_common))
            accept_list = Zenoplusplus(args, copy.deepcopy(global_model.state_dict()),current_param,global_param_update,std_keys, current_index)
            if len(accept_list)== 0:
                global_weights = global_model.state_dict()
            else:
                global_weights , avg_weights= Fedavg(args, epoch, accept_list, global_model)

        elif args.update_rule == 'FLARE':
            update_params = copy.deepcopy(local_weights_delay[0])
            for item  in local_delay_ew:
                update_params.extend(item)
            global_weights = FLARE(args, global_model, update_params,DatasetSplit(train_dataset,dict_common) )
        
        elif args.update_rule == 'LFR':
            update_params = copy.deepcopy(local_weights_delay[0])
            for item  in local_delay_ew:
                update_params.extend(item)
            update_indexes = copy.deepcopy(local_index_delay[0])
            for item  in local_index_ew:
                update_indexes.extend(item)
            global_weights = LFR(args,global_model,update_params,update_indexes,DatasetSplit(train_dataset,dict_common))

        elif args.update_rule == 'Krum':
            update_params = copy.deepcopy(local_weights_delay[0])
            for item  in local_delay_ew:
                update_params.extend(item)
            benign_number = len(update_params) - m
            global_weights = Krum(update_params,std_keys,benign_number )
        else:
            # Fedavg
            all_weights = copy.deepcopy(local_weights_delay[0])
            # all_weights.extend(local_delay_ew)
            for item in local_delay_ew:
                all_weights.extend(item)
            global_weights , avg_weights= Fedavg(args, epoch, all_weights, global_model)

    

            

        # Update global weights
        pre_global_model.load_state_dict(global_model.state_dict())
        global_model.load_state_dict(global_weights)

        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            if c in attack_users and args.inverse_poison == True:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[c], data_poison=False,  idx=c)
            else:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[c], data_poison=False, idx=c)

            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_test_acc = sum(list_acc) / len(list_acc)
        train_accuracy.append(train_test_acc)

        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')

            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))

        test_acc, test_loss, _ = test_inference(args, global_model, test_dataset)
        final_test_acc.append(test_acc)
        print('Test Accuracy: {:.2f}% \n'.format(100 * test_acc))

        # Schedular Update
        for l in all_users:
            if(scheduler[l] > 0):
                scheduler[l] = (scheduler[l] - 1)   
                

            


    print(f' \n Results after {args.epochs} global rounds of training:')

    print("|---- Avg testing Accuracy across each device's data: {:.2f}%".format(100 * train_accuracy[-1]))

    for i in range(len(train_accuracy)):
        print("|----{}th round Training Accuracy : {:.2f}%".format(i, 100 * train_accuracy[i]))

    print("|----Final Test Accuracy: {:.2f}%".format(100 * test_acc))

    for i in range(len(final_test_acc)):
        print("|----{}th round Final Test Accuracy : {:.2f}%".format(i, 100 * final_test_acc[i]))


    exp_details(args)
    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))


    if args.data_poison == True:
        attack_type = 'data'
    elif args.model_poison == True:
        attack_type = 'model'
        model_scale = '_scale_' + str(args.model_poison_scale)
        attack_type += model_scale
    else:
        attack_type = 'no_attack'

    file_n = f'accuracy_{args.update_rule}__{args.poison_methods}_{attack_type}_poison_eth_{args.eth}_delta_{args.delta}_{args.frac}_{args.seed}_{args.lam}.csv'

    f = open(file_n, 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    for i in range((len(final_test_acc))):
        wr.writerow([i + 1, final_test_acc[i] * 100])

    f.close()










