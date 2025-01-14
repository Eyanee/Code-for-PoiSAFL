
import copy
import math
import torch
import logging
from torch.utils.data import DataLoader, Dataset
from torch import nn
import numpy as np
import torch.nn.functional as F


def poison_Mean(para_updates, avg_update, args, user_number ,benign_user_number): 
    # n_attackers = args.num_users - args.benign_user_number
    dev_type = args.dev_type

    if dev_type == 'sign':
        deviation = torch.sign(avg_update)
    elif dev_type == 'unit_vec':
        deviation = avg_update / torch.norm(avg_update)  # unit vector, dir opp to good dir
    elif dev_type == 'std':
        deviation = torch.std(para_updates, 0)

    lamda = torch.Tensor([args.threshold]).cuda(0)  # compute_lambda_our(all_updates, model_re, n_attackers) #args.threshold, type=float, default=20.0
    prev_loss = -1
    step = lamda / 2
    lamda_succ = 0

    # print("user_number is ", user_number)
    # print("para_updates length is ", para_updates.size())
    i = 0
    while torch.abs(lamda_succ - lamda) > args.threshold_diff: #threshold_diff, type=float, default=1e-5
        mal_update = avg_update - lamda * deviation  
        for client_num in range(user_number):
            if client_num >= benign_user_number: 
                para_updates[client_num] = mal_update.clone().detach() 

        agg_grads = grads_Mean(para_updates, args) 

        loss = torch.norm(agg_grads - avg_update) 

        if prev_loss < loss:
            lamda_succ = lamda
            lamda = lamda + step / 2
        else:
            lamda = lamda - step / 2

        step = step / 2
        prev_loss = loss
        i = i + 1
    mal_update = avg_update - lamda_succ * deviation

    return mal_update

def grads_Mean(para_updates, args):
    mean_grads = torch.mean(para_updates, 0) 
    return mean_grads


def scale_attack(global_grad, ori_grad, scale_weight, current_number_of_adversaries):
   

    clip_rate = (scale_weight/ current_number_of_adversaries) * -1
    print(f"Scaling by  {clip_rate}")
    mod_grad = copy.deepcopy(ori_grad)

    for key in mod_grad.keys():

        target_value = mod_grad[key]
        value = global_grad[key]
        new_value = target_value + (value - target_value) * clip_rate 

        mod_grad[key].copy_(new_value)
        
    distance = model_dist_norm(ori_grad, mod_grad)
    return mod_grad


def scale_attack_mod(args, model, train_dataset, global_grad, ori_grad, current_number_of_adversaries):
    scale_weight = args.scale_weight
    clip_rate = (scale_weight/ current_number_of_adversaries)  ### clip_rate可以修改成跟修改参数相关的因素的值  attacker_num/ staleness/
    print(f"Scaling by  {clip_rate}")
    mod_grad = copy.deepcopy(ori_grad)

    device = f'cuda:{args.gpu_number}' if args.gpu else 'cpu'
    testloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    batch_entropy = []

    avg_batch_entropy = 2
    while avg_batch_entropy >= 1 or avg_batch_entropy <= 0.75:
        if avg_batch_entropy < 0.75:
            clip_rate = clip_rate * 2
        else:
            clip_rate = clip_rate/2
        print(f"step scaling by  {clip_rate}")
        new_grad = modifyGradient(clip_rate, ori_grad, global_grad)
        batch_entropy = []
        model.load_state_dict(new_grad)
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            output, out = model(images)
            Information = F.softmax(out, dim=1) * F.log_softmax(out, dim=1)
            entropy  = -1.0 * Information.sum(dim=1) # size [64]
            average_entropy = entropy.mean().item()
            batch_entropy.append(average_entropy)
  
        
        avg_batch_entropy = sum(batch_entropy)/len(batch_entropy)

        

    return new_grad


def modifyGradient(clip_rate, ori_grad, global_grad):
    mod_grad = copy.deepcopy(ori_grad)
    for key in mod_grad.keys(): ## model.state_dict()

        target_value = mod_grad[key]
        value = global_grad[key]
        new_value = target_value + (value - target_value) * clip_rate 

        mod_grad[key].copy_(new_value)
    return mod_grad

def model_dist_norm(ori_params, mod_params):
    squared_sum = 0
    for name in ori_params:
        squared_sum += torch.sum(torch.pow(ori_params[name] - mod_params[name], 2))
    distance = math.sqrt(squared_sum)
    return distance


