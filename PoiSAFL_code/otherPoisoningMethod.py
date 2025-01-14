
import torch
import numpy as np
import copy

def get_key_list(std_keys):
    res = list()
    for key in std_keys:
        res.append(key)

    return res

def modifyWeight(std_keys, local_weights):  
    param_updates = list()
    param_update = list()
    for update_item in local_weights:
        param_new = []
        for key in std_keys:
            param_new.append(copy.deepcopy(update_item[key]))

        param_update = [] 
        for j in range(len(param_new)):
            sub_res = torch.sub(param_new[j], 0).reshape(-1)
            param_update = sub_res if len(param_update) == 0 else torch.cat((param_update, sub_res), 0)

        param_updates = param_update.clone().unsqueeze(0) if len(param_updates) == 0 else torch.cat((param_updates, param_update.clone().unsqueeze(0)), dim=0)  # 先用unsqueeze(0)增加维度

    return param_updates


def restoreWeight(std_dict, std_keys, update_weights):
    update_dict = copy.deepcopy(std_dict)
    front_idx = 0
    end_idx = 0

    for k in std_keys:
        tmp_len = len(list(std_dict[k].reshape(-1)))
        end_idx = front_idx + tmp_len
        tmp_tensor = update_weights[front_idx:end_idx].view(std_dict[k].shape)
        update_dict[k] = copy.deepcopy(tmp_tensor)
        front_idx = end_idx
    return update_dict



def LIE_attack(benign_users):
    if len(benign_users) == 0:
        return
    std_dict = copy.deepcopy(benign_users[0])

    users_grads = torch.stack(benign_users,dim = 0)

    params_mean = torch.mean(users_grads, axis=0) 
    params_stdev = torch.var(users_grads, axis=0) ** 0.5 

    
    mal_param = params_mean + 0.74 * params_stdev


    return mal_param




def min_max(args, all_updates, dev_type='std'):
    gpu_number = args.gpu_number
    device = torch.device(f'cuda:{gpu_number}' if args.gpu else 'cpu')

    std_keys = all_updates[0].keys()
    std_dict = copy.deepcopy(all_updates[0])
    param_updates = modifyWeight(std_keys, all_updates)

    model_re = torch.mean(param_updates, axis=0) 

    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(param_updates, 0)
        
    max_distance = 0
    max_d = 0
    lamda_succ = lamda = torch.Tensor([11.0]).to(device)
    lamda = torch.Tensor([10.0]).to(device)
    lamda_fail = lamda
    threshold_diff = 1e-3
    for grad_i in param_updates:
        for grad_j in param_updates:
            distance = torch.norm(grad_i - grad_j)**2
        max_distance = max(max_distance, distance)

    

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        for grad in param_updates:
            distance = torch.norm(grad - mal_update)**2
            max_d = max(max_d, distance)
        if max_d <= max_distance:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
            break
        else:

            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    mal_update = (model_re - lamda_succ * deviation)
    
    mal_update = restoreWeight(std_dict, std_keys, mal_update)

    return mal_update


def min_sum(args, param_updates, dev_type='unit_vec'):
    
    gpu_number = args.gpu_number
    device = torch.device(f'cuda:{gpu_number}' if args.gpu else 'cpu')
    all_updates = torch.stack(param_updates,dim = 0)

    model_re = torch.mean(all_updates, axis=0) 

    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)
    
    lamda = torch.Tensor([50.0]).float().to(device)
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0
    
    distances = []
    for update in all_updates:
        distance = torch.norm((all_updates - update), dim=1) ** 2
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
    
    scores = torch.sum(distances, dim=1)
    min_score = torch.min(scores)
    del distances

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        distance = torch.norm((all_updates - mal_update), dim=1) ** 2
        score = torch.sum(distance)
        
        if score <= min_score:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    mal_update = (model_re - lamda_succ * deviation)
    
    return mal_update



def Grad_median(args, param_updates, n_attackers, std_keys, dev_type='unit_vec', threshold=1.0):
    gpu_number = args.gpu_number
    device = torch.device(f'cuda:{gpu_number}' if args.gpu else 'cpu')
    
    std_dict = copy.deepcopy(param_updates[0])
    all_updates = torch.stack(param_updates,dim = 0)

    model_re = torch.mean(all_updates, axis=0)



    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re) 
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(param_updates, 0)

    lamda = torch.Tensor([threshold]).to(device)
    threshold_diff = 1e-5
    prev_loss = -1
    lamda_fail = lamda
    lamda_succ = 0
    iters = 0 
    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_updates), 0)

        agg_grads = torch.median(mal_updates, 0)[0]
        
        loss = torch.norm(agg_grads - model_re)
        
        if prev_loss < loss:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2
        prev_loss = loss
        
    mal_update = (model_re - lamda_succ * deviation)
    

    return mal_update

def compute_gradient(model_1, model_2, std_keys,  lr):
    grad = list()
    for key in std_keys:
        param1 = model_1[key]
        param2 = model_2[key]
        tmp = (param1 - param2)
        grad = tmp.view(-1) if len(grad)== 0 else torch.cat((grad,tmp.view(-1)),0)
    return grad

def restoregradients(std_dict, std_keys, update_weights):
    update_dict = copy.deepcopy(std_dict)
    front_idx = 0
    end_idx = 0

    for k in std_keys:
        tmp_len = len(list(std_dict[k].reshape(-1)))
        end_idx = front_idx + tmp_len
        # print("update_weights shape", type(update_weights))
        # print("front idx and end idx", front_idx, end_idx)
        tmp_tensor = update_weights[front_idx:end_idx].view(std_dict[k].shape)
        update_dict[k] = copy.deepcopy(tmp_tensor) +  update_dict[k]
        front_idx = end_idx
    return update_dict

def getAllGraidients(args, params,global_model,std_keys):
    grads = list()
    for param in params:
        grad  = compute_gradient(param,global_model.state_dict(),std_keys,args.lr)
        grads.append(grad)
    grads =torch.stack(grads,dim  = 0)

    return grads

### LA attack
    ## on Trimmed Mean and Mean full knowledge
def modifyLA(std_keys, param_updates,global_model,lr):
    res_list = []
    for param in param_updates:
        param_mod = compute_gradient(param,global_model.state_dict(),std_keys,lr)
        res_list.append(param_mod)
    return torch.stack(res_list,dim = 0)
        
    
def LA_attack(args, param_updates, n_attackers,global_model,std_keys):

    gpu_number = args.gpu_number
    device = torch.device(f'cuda:{gpu_number}' if args.gpu else 'cpu')


    all_updates = torch.stack(param_updates,dim = 0)


    model_re = torch.mean(all_updates, 0)
    model_std = torch.std(all_updates, 0)
    deviation = torch.sign(model_re)
    
    max_vector_low = model_re + 3 * model_std 
    max_vector_hig = model_re + 4 * model_std
    min_vector_low = model_re - 4 * model_std
    min_vector_hig = model_re - 3 * model_std

    max_range = torch.cat((max_vector_low[:,None], max_vector_hig[:,None]), dim=1)
    min_range = torch.cat((min_vector_low[:,None], min_vector_hig[:,None]), dim=1)

    rand = torch.from_numpy(np.random.uniform(0, 1, [len(deviation), n_attackers])).type(torch.FloatTensor).to(device)

    max_rand = torch.stack([max_range[:, 0]] * rand.shape[1]).T + rand * torch.stack([max_range[:, 1] - max_range[:, 0]] * rand.shape[1]).T
    min_rand = torch.stack([min_range[:, 0]] * rand.shape[1]).T + rand * torch.stack([min_range[:, 1] - min_range[:, 0]] * rand.shape[1]).T

    mal_vec = (torch.stack([(deviation > 0).type(torch.FloatTensor)] * max_rand.shape[1]).T.to(device) * max_rand + torch.stack(
        [(deviation > 0).type(torch.FloatTensor)] * min_rand.shape[1]).T.to(device) * min_rand).T

    print("mal_vec shape is ",mal_vec.shape)


    
    return mal_vec


