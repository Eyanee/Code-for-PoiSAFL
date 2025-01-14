import torch
import numpy as np
from torch import nn
import heapq
import copy
import math
from update import  LocalUpdate
from utils1 import average_weights
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset



def get_key_list(std_keys):
    res = list()
    for key in std_keys:
        res.append(key)

    return res

def modifyWeight(std_keys, local_weights):  
    """
    local_weights : state_dict 字典值
    """
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



def restoregradients(std_dict, std_keys, update_weights):

    update_dict = copy.deepcopy(std_dict)
    front_idx = 0
    end_idx = 0


    for k in std_keys:
        tmp_len = len(list(std_dict[k].reshape(-1)))
        end_idx = front_idx + tmp_len

        tmp_tensor = update_weights[front_idx:end_idx].view(std_dict[k].shape)
        update_dict[k] = copy.deepcopy(tmp_tensor) +  update_dict[k]
        front_idx = end_idx
    return update_dict


def preGrouping(std_keys, local_weights_delay, local_delay_ew):
    for l in local_delay_ew:
        local_weights_delay.extend(l)
        
    param_updates = modifyWeight(std_keys, local_weights_delay)
    
    return param_updates

def Trimmed_mean(para_updates, n_attackers):
    sorted_updates = torch.sort(para_updates, 0)[0]
    agg_para_update = torch.mean(sorted_updates[n_attackers:-n_attackers], 0) if n_attackers else torch.mean(sorted_updates, 0)
    return agg_para_update

def pre_Trimmed_mean(args, std_dict, std_keys, current_epoch_updates):
    weight_updates = modifyWeight(std_keys, current_epoch_updates)
    n_attackers = int(0.2 * len(weight_updates))
    Median_avg = Trimmed_mean(weight_updates, n_attackers)
    Median_avg = restoreWeight(std_dict, std_keys, Median_avg)
    num = n_attackers*2
    return Median_avg, len(weight_updates) - num

def Median(para_updates): #
    agg_para_update = torch.median(para_updates, 0,keepdim=True)
    # print("agg_para is ", agg_para_update[0].squeeze(0))
    return agg_para_update[0].squeeze(0)

def pre_Median(std_dict, std_keys, current_epoch_updates):
    weight_updates = modifyWeight(std_keys, current_epoch_updates)
    length = len(weight_updates)
    Median_avg = Median(weight_updates)
    Median_avg = restoreWeight(std_dict, std_keys, Median_avg)
    return Median_avg, length

def Mean(para_updates, args):  #
    agg_para_update = torch.mean(para_updates, dim=0)
    return agg_para_update

def compute_AFA_mean(params):
    avg_params = copy.deepcopy(params[0])
    all_params = []
    num = len(params)
    for key in avg_params.keys():
        for i in range(num):
            all_params.append(params[i][key])
        all_params_n = torch.stack(all_params, dim=0)
        all_params_n = all_params_n.to(torch.float32)  
        avg_params[key] = torch.mean(all_params_n, dim = 0)
        all_params = []
        # print("len all_params is ",all_params)
    return avg_params


def compute_similarity(params, avg_params):
    all_similariy = []
    for key in params:
        tmp_similarity = nn.functional.cosine_similarity(params[key].view(-1), avg_params[key].view(-1),dim=0)

        all_similariy.append(tmp_similarity)
    return sum(all_similariy)





def preGroupingIndex(local_index_delay, local_index_ew):
    index = local_index_delay
    for idx in local_index_ew:
        index.extend(idx)
    print(index)
    return index

"""


"""
def compute_L2_norm(params):
    squared_sum = 0
    distance = 0

    for key in params.keys():
        distance = distance + torch.norm(params[key])

    return distance

def compute_gradient(model_1, model_2, std_keys,  lr):
    grad = list()
    for key in std_keys:
        param1 = model_1[key]
        param2 = model_2[key]
        tmp = (param1 - param2)
        grad = tmp.view(-1) if len(grad)== 0 else torch.cat((grad,tmp.view(-1)),0)
    print("grad,shape is",grad.shape)
    return grad
    


def test_inference_clone(args, model, test_dataset):

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    device = f'cuda:{args.gpu_number}' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    batch_losses = []
    batch_entropy = []
    batch_grad = []

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        model.zero_grad()
        for param in model.parameters():
            param.requires_grad_(True)


        output, out,PLR = model(images)
        Information = F.softmax(out, dim=1) * F.log_softmax(out, dim=1)
        
        entropy  = -1.0 * Information.sum(dim=1) # size [64]
        average_entropy = entropy.mean().item()
        

        batch_loss = criterion(output, labels)
        batch_loss.backward()

        batch_losses.append(batch_loss.item())

        _, pred_labels = torch.max(output,1)
        pred_labels = pred_labels.view(-1)
        pred_dec = torch.eq(pred_labels, labels)
        current_acc = torch.sum(pred_dec).item() + 1e-8

        batch_entropy.append(average_entropy)

        correct += current_acc
        total += len(labels)
        
        grad = torch.cat([p.grad.view(-1) for p in model.parameters()])
        batch_grad.append(grad)  

        model.zero_grad()

        for param in model.parameters():
            param.requires_grad_(False)



    accuracy  = correct/total

        

    return accuracy, sum(batch_losses)/len(batch_losses), sum(batch_entropy)/len(batch_entropy)

def Zeno(weights,  args, model, cmm_dataset, current_epoch):

    common_acc, common_loss, _= test_inference_clone(args, model, cmm_dataset)
    
    test_model = copy.deepcopy(model)
    loss_list = []
    for param in weights:
        mod_params = copy.deepcopy(model.state_dict())
        for key in param.keys():
            mod_params[key] = torch.subtract(mod_params[key],param[key]*0.01)
        test_model.load_state_dict(mod_params)
        test_acc,test_loss, _  = test_inference_clone(args, model, cmm_dataset)
        loss_list.append(test_loss)
        
    print("loss_list is", loss_list)
    print("common_loss is", common_loss)
    
    fai = 0.01
    w = model.state_dict()
    score = []

    for i in range(0, len(weights)):
        length = compute_L2_norm(weights[i])
        print("length is", length)
        tmp = common_loss - loss_list[i] - fai * length
        print("score is ",tmp)
        score.append(tmp.__float__())
    min_value = min(score)
    score = np.array(score) - min_value
    score = score / sum(score)

    

    re_model = copy.deepcopy(w)
    for key in w.keys():
        for i in range(0, len(score)):
            if i == 0:
                re_model[key] = score[i] * weights[i][key]
            else:
                re_model[key] += score[i] * weights[i][key]

    alpha = 0.5
    if args.dataset == 'cifar10' or 'cifar100':
        alpha = 0.5 ** (current_epoch // 300)
    elif args.dataset =='fmnist':
        alpha = 0.8
    elif args.dataset =='mnist':
        alpha = 0.5 ** (current_epoch // 15)

    for key in w.keys():       
        re_model[key] = re_model[key] * (alpha) + w[key] * (1 - alpha)
    
    return re_model

def pre_Zeno(current_epoch_updates, args, loss, cmm_dataset, global_model):
    # weight_updates = modifyWeight(std_keys, current_epoch_updates)
    Zeno_avg = Zeno(current_epoch_updates, loss, args, global_model, cmm_dataset)
    # Median_avg = restoreWeight(std_dict, std_keys, Median_avg)
    return Zeno_avg



def update_weights_zeno(args, model, global_round,test_dataset):
    model.train()
    epoch_loss = []
    epoch_grad = []
    device = f'cuda:{args.gpu_number}' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    if args.optimizer == 'sgd':

        lr = args.lr
        lr = lr * (0.5) ** (global_round // args.lrdecay)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    elif args.optimizer == 'adam':

        lr = args.lr
        lr = lr * (0.5) ** (global_round // args.lrdecay)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for iter in range(args.local_ep):
        batch_loss = []
        batch_grad = []
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
                
            model.zero_grad()
            for param in model.parameters():
                param.requires_grad_(True)
            log_probs, _ ,PLR= model(images)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()
            
            grad = torch.cat([p.grad.view(-1) for p in model.parameters()])
            # print(grad)
            batch_grad.append(grad)               

            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
        
    
    return model.state_dict()
# Zeno++
def scale_updates(param_updates,c):
    for key in param_updates.keys():
        param_updates[key] = c * param_updates[key]
    return param_updates

"""


"""
def Zenoplusplus(args, global_state_dict, param_updates,global_update_param, std_keys, indexes):
    print("index is ", indexes)
    zeno_rho = 0.001 #
    zeno_epsilon = 0.02 # 0.02

    accept_list = []

    global_update = compute_gradient(global_update_param,global_state_dict,std_keys,args.lr)
    global_param_square =  torch.norm(global_update)
    print("global_param_square norm is   ",global_param_square)
    for idx, param_update in enumerate(param_updates):
        param_update_gd = compute_gradient(param_update,global_state_dict,std_keys,args.lr)
        user_param_square = torch.norm(param_update_gd)
        
        c = (global_param_square / user_param_square).double()
        user_param_square= user_param_square* c
        user_param =  param_update_gd *  c
        
        # compute score
        zeno_innerprod = 0
        zeno_square = global_param_square
        zeno_innerprod = torch.dot(user_param.double(), global_update.double())
        score = args.lr * (zeno_innerprod) - zeno_rho * (zeno_square) + args.lr * zeno_epsilon
        print("score :", score)
        if score >= 0:
            modified_updates = restoregradients(global_state_dict, std_keys,  param_update_gd)
            # param_updates[index]
            accept_list.append(modified_updates)
    return accept_list



def compute_mmd(x, y, sigma=1.0):
    xx = torch.matmul(x, x.t())
    yy = torch.matmul(y, y.t())
    xy = torch.matmul(x, y.t())

    k_xx = torch.exp(-torch.sum((x.unsqueeze(1) - x.unsqueeze(0)) ** 2, dim=2) / (2 * sigma ** 2)).mean()
    k_yy = torch.exp(-torch.sum((y.unsqueeze(1) - y.unsqueeze(0)) ** 2, dim=2) / (2 * sigma ** 2)).mean()
    k_xy = torch.exp(-torch.sum((x.unsqueeze(1) - y.unsqueeze(0)) ** 2, dim=2) / (2 * sigma ** 2)).mean()

    mmd = k_xx + k_yy - 2 * k_xy
    return mmd


"""


"""
def FLARE(args, global_model, param_updates, common_dataset):

    
    device = f'cuda:{args.gpu_number}' if args.gpu else 'cpu'
    test_model = copy.deepcopy(global_model)
    test_dict = copy.deepcopy(global_model.state_dict())
    user_len = len(param_updates)
    test_model.eval()

    trainloader = DataLoader(common_dataset, batch_size=64, shuffle=False) 
    user_PLR = []

    for param in param_updates:
        test_model.load_state_dict(param)
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            output,out, PLR = test_model(images)
    
        user_PLR.append(PLR)
    print("user_PLR len size", len(user_PLR))

    mmd_set = np.zeros((user_len, user_len))
    mmd_indicator = np.zeros((user_len, user_len))
    count_indicator = np.zeros(user_len)

    for i in range(user_len):
        for j in range(i+1, user_len):
            mmd_value = compute_mmd(user_PLR[i], user_PLR[j])
            # print("mmd_value", mmd_value)
            mmd_set[i,j]= mmd_value
            mmd_set[j,i]= mmd_value
    
    ## 
    k = int(user_len * 0.5)
    for idx, row in  enumerate(mmd_set):

        sorted_row = np.sort(row)  
        kth_largest = sorted_row[k - 1]    
        for jdx, element in enumerate(row):
            if element >= kth_largest:
                mmd_indicator[idx, jdx] = 1
                count_indicator[jdx] = count_indicator[jdx] + 1 

    count_tensor = torch.Tensor(count_indicator)
    print("counter_tensor is ",count_tensor)
    count_res = F.softmax(count_tensor, dim=-1)
    # print("count_res", count_indicator)
    print("count_res", count_res)
    # print("count_sum is ", torch.sum(count_res))
    for key in test_dict.keys():
        for i in range(0, len(count_res)):
            if i == 0:
                test_dict[key] = count_res[i] * param_updates[i][key]
            else:
                test_dict[key] += count_res[i] * param_updates[i][key]


        
    return test_dict

def cosine_similarity(model1, model2):

    cos_sim_list = []
  
    for key in model1.keys():
        param1 = model1[key]
        param2 = model2[key]

        cos_sim = torch.cosine_similarity(param1, param2, dim=0, eps=1e-8)
        cos_sim_list.append(torch.mean(cos_sim))
    return cos_sim_list


def normalize_update(update1, update2):

    scaled_model = {}
    for key in update1.keys():
        param1 = update1[key]
        param2 = update2[key]
        norm1 = torch.norm(param1)
        norm2 = torch.norm(param2)
        if norm1 != norm2:
            scale_factor = norm2 / norm1
            scaled_param = torch.mul(param1, scale_factor)
        else:
            scaled_param = param1.clone()
        scaled_model[key] = scaled_param

    
    return scaled_model

def update_weights(global_params, param_update):
    alpha = 0.8
    return_params = copy.deepcopy(global_params)
    for key in param_update.keys():
        return_params[key] =  return_params[key] * (1 - alpha) + param_update[key] * (alpha)

    return return_params


def get_param_flatterned(std_keys, param):
    param_update = []
    for k in std_keys:
        param_update = param[k].view(-1) if len(param_update) == 0 else torch.cat((param_update, param[k].view(-1)), 0)
    return param_update

def AFLGuard(param_updates, global_model, global_test_model, epoch, std_keys, lr, lamda):
    
    update_params = []
    global_weights = copy.deepcopy(global_model.state_dict())
    w, loss,gd = global_test_model.update_weights(

                model=copy.deepcopy(global_model), global_round=epoch

            )
    param_g = compute_gradient(w,copy.deepcopy(global_model.state_dict()),std_keys,lr)
    norm_2 = torch.norm(param_g, p =2)
    for idx, param in enumerate(param_updates):
        
        param_i =compute_gradient(param,copy.deepcopy(global_model.state_dict()),std_keys,lr)
        
        norm_1 = torch.norm(torch.subtract(param_i, param_g))
        # norm_1 = torch.norm(param_i,p =2)
        print("norm_1 is ", norm_1)
        print("norm_2 is ", norm_2)
        print("norm_2 * lamda is",norm_2 * lamda )

        if norm_1 <= norm_2 * lamda:
            print("satisfying", idx)
            # global_weights = copy.deepcopy(global_model.state_dict())
            update_param = restoregradients(global_weights,std_keys,param_i)
            update_params.append(update_param)
            
        else:
            print("do  not satisfying", idx)
        
    update_res =average_weights(update_params)
    global_weights = update_weights(global_model.state_dict(),update_res)
    global_model.load_state_dict(global_weights)
    return global_model.state_dict()


def norm_clipping(global_model, local_weights_delay ,local_delay_ew,std_keys,lr):
    params_mod = list()
    params = copy.deepcopy(local_weights_delay)
    for item in local_delay_ew:
        params.extend(item)
    for item in params:
        params_mod.append(compute_gradient(item, global_model.state_dict(),std_keys,lr ))
    

    number_to_consider = int(len(params_mod)* 0.8) 
    weight_updates = torch.stack(params_mod,dim= 0)
    norm_res = torch.norm(weight_updates, p =2 ,dim = 1)
    sorted_norm, sorted_idx = torch.sort(norm_res)
    used_idx = sorted_idx[:number_to_consider]
    avg_grad =  torch.mean(weight_updates[used_idx,: ],dim = 0)
    weight_res = restoregradients(copy.deepcopy(global_model.state_dict()),std_keys,avg_grad)

    return weight_res


def LFR(args, global_model, param_updates, indexes, common_dataset):

    loss_list = []
    test_model = copy.deepcopy(global_model)
    w_avg = average_weights(param_updates)
    w_origin = update_global(args, copy.deepcopy(global_model.state_dict()),w_avg)
    test_model.load_state_dict(copy.deepcopy(w_origin))
    acc_origin, loss_origin, _ = test_inference_clone(args, test_model, common_dataset)
    for idx in range(len(param_updates)):
        
        temp_list = param_updates[:idx] + param_updates[idx+1:]  
        w_avg = average_weights(temp_list)
        w_temp = update_global(args, copy.deepcopy(global_model.state_dict()),w_avg)
        test_model.load_state_dict(copy.deepcopy(w_temp))
        acc_temp, loss_temp, _ = test_inference_clone(args, test_model, common_dataset)
        loss_diff = abs(loss_temp - loss_origin)
        loss_list.append(loss_diff)
    
    loss_sorted = sorted(loss_list, reverse=True)
    threshold_idx = math.floor(len(loss_list) *0.2)
    threshold_value = loss_sorted[threshold_idx]

    
    new_list = get_list(loss_list,threshold_value,threshold_idx)  
    print("list is ",new_list)
    total_sum = sum(new_list)  
    new_list = [x / total_sum for x in new_list]  
    print("weight is ",new_list)

    for key in w_avg.keys():
        for idx, param in enumerate(param_updates):
            if idx == 0:
                w_avg[key] = param[key] * new_list[idx]

            else:
                w_avg[key] += param[key] * new_list[idx]
  
    
    return update_global(args, global_model.state_dict(), w_avg)


def Krum(params,std_keys, benign_user_number):  # 
    update_dict = copy.deepcopy(params[0])
    para_updates = modifyWeight(std_keys,params)



    clients_l2 = [[] for _ in range(len(para_updates))]

    for index1, client_logits1 in enumerate(para_updates):
        for index2, client_logits2 in enumerate(para_updates):
            if (index1 == index2): 
                continue
            l2 = torch.dist(client_logits1, client_logits2, p=2) 
            clients_l2[index1].append(l2) 

    clients_l2_filter = [[] for _ in range(len(para_updates))]
    for index, client_l2 in enumerate(clients_l2):
        list.sort(client_l2)
        client_l2_minN = sum(client_l2[0:benign_user_number - 2]) 
        # print(client_l2_minN)
        clients_l2_filter[index].append(client_l2_minN)

    selected_client_index = clients_l2_filter.index(min(clients_l2_filter))
    agg_para_update = para_updates[selected_client_index]

    
    return_params = restoreWeight(update_dict,std_keys, agg_para_update)

    return return_params



def get_list(old_list, threshold_value, num):
    new_list = [1]*len(old_list)
    count = 0
    for idx, item in enumerate(old_list):
        if count > num:
            break
        if item >= threshold_value:
            new_list[idx] = 0
            count = count +1
    return new_list


def update_global(args, global_weights,update_param):
    if args.dataset =='cifar':
        alpha = 0.8
    elif args.dataset =='fmnist':
        alpha = 0.8

    elif args.dataset =='mnist':
        alpha = 0.1
    else:
        alpha = 0.8
    w_semi = copy.deepcopy(global_weights)
    for key in w_semi.keys():
        w_semi[key] = w_semi[key] * (1 - alpha) + update_param[key] * (alpha)
    return w_semi

def flatten_parameters(model):
    return np.concatenate([param.cpu().detach().numpy().flatten() for param in model.parameters()])

def is_nested_list(lst):  

    for item in lst:  

        if isinstance(item, list):  

            return True  

    return False 

class DatasetSplit_clone(Dataset):

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

