#The codes are based on Ubuntu 16.04 with Python 3.7 and Pytorch 1.0.1

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch

from update import LocalUpdate, test_inference, DatasetSplit
from poison_optimization_fmnist import Outline_Poisoning, add_small_perturbation,cal_ref_distance,model_dist_norm,cal_similarity
from model import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, VGGCifar,GTSRBNet
from resnet import *
from utils1 import *
import csv
from torch.utils.data import DataLoader, Dataset
from options import args_parser
import os
from otherGroupingMethod import *
from otherPoisoningMethod import *
import  gc
gc.collect()
torch.cuda.empty_cache()



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

    for i in range(args.staleness):
        if i == args.staleness +1:
            front_idx = int(t * i)
            end_idx = n-  1
        else:
            front_idx = int(t * i)
            end_idx = front_idx + t
        for j in range(front_idx, end_idx):
            clientStaleness[j] = i + 1 


    for l in attack_users:
        clientStaleness[l] = TARGET_STALENESS   
        
    MAX_STALENESS = args.staleness
    mal_parameters_list = {}
    for i in range(MAX_STALENESS):
        mal_parameters_list[i] = {}
    
    mal_grad_list = []

    
    
    std_keys = get_key_list(global_model.state_dict().keys())

    mal_rand = global_model.state_dict()
    
            
    for epoch in tqdm(range(args.epochs)):

        local_weights_delay = {}
        loss_on_public = {}
        entropy_on_public = {}
        local_index_delay = {}
        local_grad_delay = {}
        malicious_models = []
        mal_grad_list = []

        for i in range(args.staleness + 1):
            loss_on_public[i] = []
            entropy_on_public[i] = []
            local_weights_delay[i] = []
            local_index_delay[i] = []
            local_grad_delay[i] = []


        print(f'\n | Global Training Round : {epoch + 1} | \n')

        global_model.train()

        global_weights_rep = copy.deepcopy(global_model.state_dict())
        

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

        pre_weights[args.staleness] = []
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

            else:

                continue

            w, loss,gd = local_model.update_weights(

                model=copy.deepcopy(global_model), global_round=epoch

            )
            gd_test = compute_gradient(w,global_model.state_dict(),std_keys,args.lr)

            ensure_1 += 1 
            if idx in attack_users and args.model_poison == True and epoch >=args.staleness - MAX_STALENESS:
                mal_parameters_list[0][idx] = w 

                mal_grad_list.append(gd)

            elif  idx in attack_users and args.new_poison == True and epoch >=MAX_STALENESS:
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
                local_grad_delay[scheduler[idx] - 1].append(copy.deepcopy(gd_test))
                loss_on_public[scheduler[idx] - 1].append(common_loss_sync)
                entropy_on_public[scheduler[idx] - 1].append(common_entropy_sample)
                benign_distance = model_dist_norm(w,copy.deepcopy(global_model.state_dict()))

                

               

                
            
        if args.model_poison == True and epoch >=MAX_STALENESS:
            if args.poison_methods == 'ourpoisonMethod':
                malicious_models = list(mal_parameters_list[0].values())

                pinned_accuracy_threshold = 0.4 # 
                adaptive_accuracy_threshold = pinned_accuracy_threshold
                if epoch == MAX_STALENESS:
                    distance_threshold = cal_ref_distance(malicious_models,copy.deepcopy(global_model), 0.8) 
                    mal_rand = add_small_perturbation(global_model, args, pinned_accuracy_threshold, train_dataset, distance_threshold,perturbation_range=(-0.5, 0.5))
                    test_model.load_state_dict(mal_rand)
                    mal_acc, mal_loss_sync, mal_entropy_sample = test_inference(args, test_model,
                                                                                        DatasetSplit(train_dataset,
                                                                                            dict_common))   


                distance_threshold = cal_ref_distance(malicious_models,copy.deepcopy(global_model), 0.8) 
                malicious_dict = Outline_Poisoning(args, copy.deepcopy(global_model), malicious_models, 
                                                                train_dataset, distance_threshold, pinned_accuracy_threshold,copy.deepcopy(mal_rand))
                
            
                test_model.load_state_dict(malicious_dict)
                mal_acc, mal_loss_sync, mal_entropy_sample = test_inference(args, test_model,
                                                                                            DatasetSplit(train_dataset,
                                                                                                        dict_common))
                
                mal_grad = compute_gradient(malicious_dict,global_model.state_dict(),std_keys,args.lr)
                norm_2 = torch.norm(mal_grad, p =2 )
                mal_distance = model_dist_norm(malicious_dict,copy.deepcopy(global_model.state_dict()))
                for idx in attack_users:
                        
                    local_weights_delay[ scheduler[idx] - 1 ].append(copy.deepcopy(malicious_dict))
                    local_index_delay[ scheduler[idx] - 1 ].append(idx)
                    loss_on_public[scheduler[idx] - 1].append(mal_loss_sync)
                    entropy_on_public[scheduler[idx] - 1].append(mal_entropy_sample)
                    #
                    local_grad_delay[scheduler[idx] - 1].append(copy.deepcopy(mal_grad))
        
            elif args.poison_methods == 'LA':
                    malicious_dicts= LA_attack(args, list(mal_parameters_list[0].values()), m,copy.deepcopy(global_model),std_keys)
                    global_weights = copy.deepcopy(global_model.state_dict())
                    # test_model.load_state_dict(global_weights)
                    # optimizer_fed = Adam(test_model.parameters(), lr=0.01)
                    for mal_vec in malicious_dicts:
                        
                        mal_dict = restoreWeight(global_weights, std_keys, mal_vec)
                        test_model.load_state_dict(mal_dict)
                        mal_acc, mal_loss_sync, mal_entropy_sample = test_inference(args, test_model,
                                                                                                    DatasetSplit(train_dataset,
                                                                                                        dict_common))
                

                        
                        local_weights_delay[ scheduler[idx] - 1 ].append(copy.deepcopy(test_model.state_dict()))
                        local_index_delay[ scheduler[idx] - 1 ].append(idx)
                        loss_on_public[scheduler[idx] - 1].append(mal_loss_sync)
                        entropy_on_public[scheduler[idx] - 1].append(mal_entropy_sample)
                        #
                        local_grad_delay[scheduler[idx] - 1].append(copy.deepcopy(mal_dict))
                        test_model.load_state_dict(global_weights)  # 重新清零


            else:
                # for idx in attack_users  :
                if  scheduler[idx] == clientStaleness[idx]:
                    if args.poison_methods == 'LIE':
                        print("len of [max] is ", len(list(mal_parameters_list[MAX_STALENESS - 1].values())))
                        print("len of [0] is ", len(list(mal_parameters_list[0].values())))
                        malicious_dict = LIE_attack(list(mal_parameters_list[0].values()))
                        malicious_grads=malicious_dict
                        test_model.load_state_dict(malicious_dict)
                    
                    elif args.poison_methods == 'min_max':
                        print("len of [0] is ", len(list(mal_parameters_list[0].values())))
                        malicious_dict = min_max(args, list(mal_parameters_list[0].values()))
                    elif args.poison_methods == 'min_sum':
                        test_model.load_state_dict(global_weights)
                        optimizer_fed = Adam(test_model.parameters(), lr=0.01)
                        malicious_grads = min_sum(args, mal_grad_list)
                        model_grads=[]
                        start_idx = 0

                        optimizer_fed.step(malicious_grads)
                    elif args.poison_methods == 'Grad':

                        print("grad")
                        # benign_grads =  torch.stack(benign_params)
                        test_model.load_state_dict(global_weights)
                        optimizer_fed = Adam(test_model.parameters(), lr=0.01)
                        malicious_dict = Grad_median(args,mal_grad_list, m,std_keys)

                        optimizer_fed.step(malicious_dict)
 
                    mal_acc, mal_loss_sync, mal_entropy_sample = test_inference(args, test_model,
                                                                                                DatasetSplit(train_dataset,
                                                                                                    dict_common))
                    malicious_dict = copy.deepcopy(test_model.state_dict())
                    # mal_grad = compute_gradient(malicious_dict,global_model.state_dict(),std_keys,args.lr)
                    print("mal_acc is", mal_acc)
                    print("mal_loss is", mal_loss_sync)
                    print("mal_entropy is", mal_entropy_sample)
                    for idx in attack_users  :
                        local_weights_delay[ scheduler[idx] - 1 ].append(malicious_dict)
                        local_index_delay[ scheduler[idx] - 1 ].append(idx)
                        loss_on_public[scheduler[idx] - 1].append(mal_loss_sync)
                        entropy_on_public[scheduler[idx] - 1].append(mal_entropy_sample)
                
                        local_grad_delay[scheduler[idx] - 1].append(copy.deepcopy(malicious_dict))
                    
                        

        for i in range(args.staleness):
            if i != 0:
                if args.update_rule == 'Sageflow':
                    # Averaging delayed local weights via entropy-based filtering and loss-wegithed averaging
                    if len(local_weights_delay[i]) > 0:
                        w_avg_delay, len_delay = Eflow(local_weights_delay[i], loss_on_public[i], entropy_on_public[i], epoch)
                        
                        
                        test_model.load_state_dict(w_avg_delay)
                        mal_acc, mal_loss_sync, mal_entropy_sample = test_inference(args, test_model,
                                                                                                    DatasetSplit(train_dataset,
                                                                                                        dict_common))
                        print("benign_acc is", mal_acc)
                        print("benign_loss is", mal_loss_sync)
                        print("benign_entropy is", mal_entropy_sample)
                        
                        pre_weights[i].append({epoch: [w_avg_delay, len_delay]})
                        # print("num1 of attacker is ",num_attacker_1)
                elif args.update_rule == 'Median':
                    if len(local_weights_delay[i]) > 0:

                        pre_weights[i].append(local_weights_delay[i])

                elif args.update_rule == 'Trimmed_mean':
                    if len(local_weights_delay[i]) > 0:
                        # std_keys = get_key_list(global_model.state_dict().keys())
                        # w_avg_delay, len_delay = pre_Trimmed_mean(copy.deepcopy(global_model.state_dict()), std_keys, local_weights_delay[i])
                        # pre_weights[i].append({epoch: [w_avg_delay, len_delay]})
                        pre_weights[i].append(local_weights_delay[i])
                elif args.update_rule == 'norm_bounding':
                    if len(local_weights_delay[i]) > 0:
                        pre_weights[i].append(local_weights_delay[i])
                elif args.update_rule == 'Zeno':
                    if len(local_weights_delay[i]) > 0:
                        pre_weights[i] = np.concatenate((pre_weights[i], local_weights_delay[i]), axis=0)
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
                        w_avg_delay = average_weights(local_weights_delay[i])
                        len_delay = len(local_weights_delay[i])   
                        pre_weights[i].append({epoch: [w_avg_delay, len_delay]})
                        
        if args.update_rule == 'Sageflow':
            sync_weights, len_sync = Eflow(local_weights_delay[0], loss_on_public[0], entropy_on_public[0], epoch)
            # Staleness-aware grouping
            
            test_model.load_state_dict(sync_weights)
            mal_acc, mal_loss_sync, mal_entropy_sample = test_inference(args, test_model,
                                                                                        DatasetSplit(train_dataset,
                                                                                            dict_common))
            print("sync_acc is", mal_acc)
            print("sync_loss is", mal_loss_sync)
            print("sync_entropy is", mal_entropy_sample)

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
            print("Trimmed mean")
            global_weights = update_weights(global_model.state_dict(),sync_weights)
        elif args.update_rule == 'norm_bounding':
            avg_weights = norm_clipping(global_model, local_weights_delay[0],local_delay_ew,std_keys,  args.lr)
            global_weights = update_weights(global_model.state_dict(), avg_weights)
        elif args.update_rule == 'Zeno':
            print("Zeno")
            local_delay_ew = np.concatenate((local_delay_ew, local_weights_delay[0]), axis=0)

            global_weights = Zeno(local_delay_ew, local_delay_gd,  args,
                                    copy.deepcopy(global_model),
                                    DatasetSplit(train_dataset, dict_common), epoch)
        elif args.update_rule == 'AFLGuard':
            current_param = []
            global_test_model = LocalUpdate(args=args, dataset=train_dataset, idxs=dict_common, idx=idx,
                                              data_poison=False, delay = False)


            current_param = copy.deepcopy(local_weights_delay[0])
            
            for k in local_delay_ew:
                current_param.extend(k)
            global_weights = AFLGuard(current_param, global_model, global_test_model, epoch, std_keys, args.lr, lamda = 5.0 )
        
        elif args.update_rule == 'Zenoplusplus':
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
            # global_grad = compute_gradient(global_param_update,global_model.state_dict(),std_keys,args.lr)
            
            accept_list = Zenoplusplus(args, copy.deepcopy(global_model.state_dict()),current_param,global_param_update,std_keys, current_index)
            print("len accept list", len(accept_list) )
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

            sync_weights = average_weights(local_weights_delay[0])
            len_sync = len(local_weights_delay[0])
            # Staleness-aware grouping
            
            test_model.load_state_dict(sync_weights)
            global_weights = Sag(epoch, sync_weights, len_sync, local_delay_ew,
                                                     copy.deepcopy(global_weights))
    

            

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
                
        # Mal_parameters_list Update
        
        if idx in attack_users and args.model_poison == True and epoch >=args.staleness - MAX_STALENESS:
            for i in range(MAX_STALENESS-1 , 0 , -1):
                mal_parameters_list[i] = copy.copy(mal_parameters_list[i-1])
            mal_parameters_list[0] = {}
            


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










