import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import copy
import math
from update import test_inference
from otherGroupingMethod import get_key_list
from p_Optimizer import MyPOptimizer
import torch.nn.init as init

initial_w_rand = None

def cal_similarity(ori_params, mod_params):
    std_keys = get_key_list(ori_params.keys())
    params1 = torch.cat([ori_params[k].view(-1) for k in std_keys])
    params2 = torch.cat([mod_params[k].view(-1) for k in std_keys])

    cos_similarity = F.cosine_similarity(params1, params2, dim=0)

    return cos_similarity

def  model_dist_norm(ori_params, mod_params):
    distance1 = 0
    pdlist = nn.PairwiseDistance(p=2)
    for key in ori_params.keys():
        if key.endswith('num_batches_tracked'):  
            continue  
        t1 = ori_params[key]
        t2 = mod_params[key]
        if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
            if ori_params[key].ndimension() == 1:
                t1 = ori_params[key].unsqueeze(0)
                t2 = mod_params[key].unsqueeze(0)

            else:
                t1 = ori_params[key]
                t2 = mod_params[key]
            temp1 = torch.sum(pdlist(t1 ,t2))
            output = torch.sum(temp1)
            distance1 += output

    return distance1


def get_distance_list(ori_state_dict, mod_state_dict):
    distance_list = []
    pdlist = nn.PairwiseDistance(p=2)

    for key in ori_state_dict.keys():

        if ori_state_dict[key].ndimension() == 1:
            t1 = ori_state_dict[key].unsqueeze(0)
            t2 = mod_state_dict[key].unsqueeze(0)
        else:
            t1 = ori_state_dict[key]
            t2 = mod_state_dict[key]
        temp1 = torch.sum(pdlist(t1 ,t2))
        output = torch.sum(temp1)
        distance_list.append(output)

    return distance_list


def Outline_Poisoning(args, global_model, malicious_models, train_dataset, distance_threshold, pinned_accuracy_threshold,w_rand):
   
    w_poison, optimization_res = phased_optimization(args, global_model, w_rand, train_dataset, distance_threshold,  0.8)

    return w_poison 



def cal_Norm(model_dict):
    res = 0
    for key in model_dict.keys():
        res += torch.norm(model_dict[key].double())
    return res

def cal_ref_distance(malicious_models, global_model, distance_ratio):
    
    distance_res = computeTargetDistance(malicious_models, global_model, distance_ratio)

    

    return distance_res


def phased_optimization(args, global_model, w_rand, train_dataset, distance_threshold, pinned_accuracy_threshold):

    # parameter determination
    round = 0
    MAX_ROUND = 3
    entropy_threshold = 1.5
    
    teacher_model = copy.deepcopy(global_model)

    student_model = copy.deepcopy(global_model)
    test_model = copy.deepcopy(global_model)

    teacher_model.load_state_dict(w_rand)
    student_model.load_state_dict(w_rand)
    teacher_model.eval()
    
    distillation_res = True
    round = 0

    while round < MAX_ROUND:
        
        test_acc, test_loss, test_entropy = test_inference(args, copy.deepcopy(student_model), train_dataset)

        student_model.load_state_dict(w_rand)
        test_distance = model_dist_norm( global_model.state_dict(),w_rand)
        test_distance_1 = model_dist_norm( global_model.state_dict(),student_model.state_dict())
        w_semi = Avg(global_model.state_dict(),w_rand)

        test_acc_1, loss_1,entropy_1 = test_inference(args, test_model, train_dataset)
        test_simliarity = cal_similarity( global_model.state_dict(),w_rand)

        if test_distance <= distance_threshold and test_acc_1 <= pinned_accuracy_threshold and test_entropy  <= entropy_threshold and test_loss <=3:
            return w_rand, True
        elif test_distance > distance_threshold:
            w_rand = adaptive_scaling(w_rand, global_model.state_dict(), distance_threshold, test_distance)
            student_model.load_state_dict(w_rand)
            round = round - 1
              
        elif (test_entropy > entropy_threshold or test_loss > 3) and distillation_res == True:
            if test_loss > 1:
                print("0")
                distillation_res, w_rand = self_distillation(args,teacher_model, student_model, train_dataset, entropy_threshold, global_model, pinned_accuracy_threshold, distance_threshold, distillation_round = 5)
            else:
                print("1")
                distillation_res, w_rand = self_distillation(args,teacher_model, student_model, train_dataset, entropy_threshold, global_model, pinned_accuracy_threshold, distance_threshold, distillation_round = 5)
 
        else:
            test_distance_2 = model_dist_norm( global_model.state_dict(),student_model.state_dict())
            print("test distance 2 is ", test_distance_2)
            distillation_res, w_rand = self_distillation(args,  teacher_model, student_model, train_dataset, entropy_threshold, global_model, pinned_accuracy_threshold, distance_threshold, distillation_round = 5)
            
        student_model.load_state_dict(w_rand)
        round = round +1

    test_acc, test_loss, test_entropy = test_inference(args, student_model, train_dataset)
    test_distance = model_dist_norm(student_model.state_dict(), global_model.state_dict())
    w_semi = Avg(global_model.state_dict(),student_model.state_dict())
    test_model.load_state_dict(w_semi)
    test_acc_1, loss_1,entropy_1= test_inference(args, test_model, train_dataset)
    test_simliarity = cal_similarity(student_model.state_dict(), global_model.state_dict())

    if test_entropy> 1:
        distillation_res, w_rand = self_distillation(args,  teacher_model, student_model, train_dataset, entropy_threshold, global_model, pinned_accuracy_threshold, distance_threshold, distillation_round = 1)

    return w_rand, False


def adaptive_scaling(w_rand, ref_model_dict, distance_threshold, test_distance):
    use_case = 5
  
    if use_case == 5:
        pdlist= nn.PairwiseDistance(p=2)
        cal_distance = test_distance
        while cal_distance > distance_threshold:
            ratio = math.sqrt((distance_threshold / cal_distance)) * 0.98 # 
            keys = reversed(get_key_list(ref_model_dict))
            for key in keys:
                w_rand[key] = torch.sub(w_rand[key], ref_model_dict[key]) * ratio + ref_model_dict[key]
            cal_distance = model_dist_norm(w_rand, ref_model_dict)
        return_distance = model_dist_norm(w_rand, ref_model_dict)
        return w_rand
    
    return w_rand

def Avg(ref_state_dict, mal_state_dict):
    res_dict = copy.deepcopy(ref_state_dict)
    for key in ref_state_dict.keys():
        res_dict[key] = 2 * ref_state_dict[key] + 2 * mal_state_dict[key]
        
    return res_dict

def self_distillation(args, teacher_model, student_model, train_dataset, entropy_threshold, ref_model, accuracy_threshold, distance_threshold,  distillation_round):
    
    lr = args.lr
    device = f'cuda:{args.gpu_number}' if args.gpu else 'cpu'
    trainloader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    # optimizer = torch.optim.Adam(student_model.parameters(), lr=lr, weight_decay=1e-4)
    optimizer = MyPOptimizer(student_model.parameters(),lr=lr)
    criterion1 = nn.NLLLoss().to(device)
    
    test_model = copy.deepcopy(ref_model)

    teacher_model.to(device)
    student_model.to(device)
    
    num_epochs = distillation_round
    temperature = 50 ### 增大
    alpha = 0.88
    beta = 0.12
    previous_loss = 0

    student_model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        acc, loss, avg_entropy = test_inference(args, copy.deepcopy(student_model), train_dataset)

        w_semi = Avg(ref_model.state_dict(),student_model.state_dict())
        test_model.load_state_dict(w_semi)
        acc_1, loss_1,entropy_1 = test_inference(args, test_model, train_dataset)
        compute_distance = model_dist_norm( ref_model.state_dict(),student_model.state_dict())
        compute_norm  = cal_Norm(student_model.state_dict())
        cos_sim = cal_similarity(ref_model.state_dict(),student_model.state_dict())

        # loss =1
        if epoch != 0:
            # 防止循环太多次
            if abs(loss - previous_loss) < 0.005:
                print("exit early")
                return False, student_model.state_dict()
        previous_loss =  loss


        if avg_entropy <= entropy_threshold and acc_1<= accuracy_threshold and loss <= 3:
        # if loss <=2 and avg_entropy <= entropy_threshold :
            return True, student_model.state_dict()
        elif avg_entropy <= entropy_threshold and loss > 3:
            print("change alpha")
            # alpha = 0.7
            # beta = 0.3
            alpha = 0.75
            beta = 0.25
        # # elif 
        # elif acc_1<= accuracy_threshold:
        #     print("restore alpha")

        #     alpha = 4
        #     beta = 1
        else:
            print("distillation for acc")

            alpha = 0.5 
            beta = 0.5
        

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            student_model.zero_grad()
            for param in student_model.parameters():
                param.requires_grad_(True)

            teacher_labels = list()
            _ , teacher_outputs,PLR = teacher_model(images)
            for item in teacher_outputs:
                pred_label = int(torch.max(item, 0)[1])
                teacher_labels.append(pred_label)

            pred_is = torch.tensor(teacher_labels)
            pred_is = pred_is.to(device)
            stu_out, student_outputs,PLR = student_model(images)
            _ , teacher_outputs,PLR= teacher_model(images)
            l_loss = criterion1(stu_out, pred_is)
            t_loss = criterion1(stu_out,  labels) 
            loss =  alpha * l_loss + beta * t_loss

            # loss = l_loss
            
            loss.backward()
            optimizer.step(list(student_model.parameters()))

    return True, student_model.state_dict()


def sign_flip(orginal_dict):
    for key in orginal_dict.keys():
        orginal_dict[key] = -1 * orginal_dict[key]
    return orginal_dict

def add_small_perturbation(original_model, args, pinned_accuracy, train_dataset, distance_threshold, perturbation_range):

    # std_keys = original_model.state_dict().keys()
    correct,total = 0.0,0.0
    test_model= copy.deepcopy(original_model)
    device = f'cuda:{args.gpu_number}' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    optimizer = torch.optim.Adam(test_model.parameters(), lr=args.lr, weight_decay=1e-4)
    # optimizer = MyPOptimizer(test_model.parameters(),lr=args.lr)
    for round in range(1):
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            test_model.zero_grad()

            for param in test_model.parameters():
                param.requires_grad_(True)


            output, out,PLR= test_model(images)
            _, pred_labels = torch.max(output,1)
            pred_labels = pred_labels.view(-1)
            pred_dec = torch.eq(pred_labels, labels)
            current_acc = torch.sum(pred_dec).item() + 1e-8
            correct += current_acc
            total += len(labels)
            # categaries = output.shape[1]
            loss = -1 *  criterion(output, labels)
            loss.backward()
            optimizer.step()

        accuracy  = correct/total
        print("batch acc is",   accuracy )

    return test_model.state_dict()



def cal_inner_product(local_dict, previous_local_dict, mal_dict):
    keys = get_key_list(local_dict.keys())
    res = 0
    for key in keys:
        mal_tensor = mal_dict[key].to(local_dict[key].dtype)
        diff1 = (mal_tensor - previous_local_dict[key]).view(-1)
        diff2 = (local_dict[key] - previous_local_dict[key]).view(-1)
        temp = torch.dot(diff1, diff2)
        res = res + temp
    
    return res




def dict2gradient(model_dict, std_keys):

    for idx, key in enumerate(std_keys):
        if idx == 0:
            grads = model_dict[key].view(-1)
        grads = torch.cat((grads, model_dict[key].view(-1)), dim = 0)

    print("grads shape is ", grads.shape)

    return grads


def gradient2dict(weights, std_keys, std_dict):
    update_dict = {}
    front_idx = 0
    end_idx = 0
    for k in std_keys:
        tmp_len = len(list(std_dict[k].reshape(-1)))
        end_idx = front_idx + tmp_len
        tmp_tensor = weights[front_idx:end_idx].view(std_dict[k].shape)
        update_dict[k] = tmp_tensor.clone()
        front_idx = end_idx
    return update_dict


def computeTargetDistance(model_dicts, global_model, ratio):
    res_distance = []

    print("len of model dicts is ", len(model_dicts))

    for model_dict in model_dicts:
        tmp_distance = model_dist_norm(model_dict, global_model.state_dict())
        tmp_similarity = cal_similarity(model_dict, global_model.state_dict())
        tmp_norm =cal_Norm(model_dict)
        res_distance.append(tmp_distance)
        print("compute distance is ", tmp_distance)
        print("compute norm is ", tmp_norm)
    res_distance.sort()

    max_idx = int(len(model_dicts))-1

    target_distance = res_distance[max_idx] * 0.8

    return target_distance


def modelAvg(benign_model_dicts, num_attacker, malicious_model):

    keys = benign_model_dicts[0].keys()
    avg_dict = copy.deepcopy(benign_model_dicts[0])

    for key in keys:
        tmp_param = []

        for model_dict in benign_model_dicts:
            tmp_param = model_dict[key].clone().unsqueeze(0) if len(tmp_param) == 0 else torch.cat((tmp_param, model_dict[key].clone().unsqueeze(0)), 0)

        if num_attacker != 0:
            for i in range(num_attacker):
                tmp_param = torch.cat((tmp_param, malicious_model.state_dict()[key].clone().unsqueeze(0)), 0)
        
        avg_param = torch.mean(tmp_param, dim = 0)
        avg_dict[key] = avg_param

    return avg_dict