import copy
import torch
from torchvision import datasets, transforms

import sys

from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid,gtsrb_iid
from sampling_withcommon import mnist_noniidcmm, cifar_noniidcmm,gtsrb_noniidcmm
from update import LocalUpdate, test_inference, DatasetSplit
from math import exp
import numpy as np
from numpy import linalg
from options import args_parser
import math
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor,Resize,Normalize
import pdb
from adam import Adam

def get_dataset(args):
    print("*******enter")
    if args.dataset == 'cifar':
        print("*******cifar")
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=apply_transform)

        if args.iid == 1:

            user_groups = cifar_iid(train_dataset, args.num_users)

        elif args.iid == 0:
            # print("noniid here")
            user_groups = cifar_noniidcmm(train_dataset, args.num_users, args.num_commondata)

        else:
            if args.unequal:
                raise NotImplementedError()
            else:
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'GTSRB':
        print("*******************GTSRB")
        data_dir = '../data/GTSRB'
        data_transforms = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
        ])
        # Resize, normalize and jitter image brightness
        data_jitter_brightness = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ColorJitter(brightness=5),
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
        ])

        # Resize, normalize and jitter image saturation
        data_jitter_saturation = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ColorJitter(saturation=5),
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
        ])

        # Resize, normalize and jitter image contrast
        data_jitter_contrast = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ColorJitter(contrast=5),
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
        ])

        # Resize, normalize and jitter image hues
        data_jitter_hue = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ColorJitter(hue=0.4),
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
        ])

        # Resize, normalize and rotate image
        data_rotate = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
        ])

        # Resize, normalize and flip image horizontally and vertically
        data_hvflip = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(1),
            transforms.RandomVerticalFlip(1),
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
        ])

        # Resize, normalize and flip image horizontally
        data_hflip = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(1),
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
        ])

        # Resize, normalize and flip image vertically
        data_vflip = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomVerticalFlip(1),
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
        ])

        # Resize, normalize and shear image
        data_shear = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomAffine(degrees = 15,shear=2),
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
        ])

        # Resize, normalize and translate image
        data_translate = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomAffine(degrees = 15,translate=(0.1,0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
        ])

        # Resize, normalize and crop image 
        data_center = transforms.Compose([
            transforms.Resize((36, 36)),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
        ])

        # Resize, normalize and convert image to grayscale
        data_grayscale = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
        ])
                                      
        train_dataset = datasets.GTSRB(data_dir, split = 'train', download=True,
                                                  transform=data_transforms)
        test_dataset = datasets.GTSRB(data_dir, split = 'test', download=True,
                                                  transform=data_transforms)
        print("len testdataset", len(test_dataset))
        print("len traindataset", len(train_dataset))
        if args.iid == 1:

            user_groups = gtsrb_iid(train_dataset, args.num_users)

        elif args.iid == 0:

            user_groups = gtsrb_noniidcmm(train_dataset, args.num_users, args.num_commondata, args.alpha)

    elif args.dataset == 'mnist' or 'fmnist':
        print("*******mnist")
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        if args.dataset == 'mnist':
            data_dir = '../data/mnist'
            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                           transform=apply_transform)
            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)
        else:
            data_dir = '../data/fmnist'
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                                  transform=apply_transform)
            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                                 transform=apply_transform)

        if args.iid == 1:

            user_groups = mnist_iid(train_dataset, args.num_users)

        elif args.iid == 0:

            user_groups = mnist_noniidcmm(train_dataset, args.num_users, args.num_commondata, args.alpha)
    
   

    return train_dataset, test_dataset, user_groups


# Federated averaging

def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(0, len(w)):
            if i == 0:
                w_avg[key] = torch.div(copy.deepcopy(w[0][key]), len(w))
            else:
                w_avg[key] += torch.div(copy.deepcopy(w[i][key]), len(w))
        # w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def compute_gradient_norm(w1, w2):
    w_avg = copy.deepcopy(w1)
    norm = 0
    for key in w_avg.keys():

        w_avg[key] = w1[key] - w2[key]
        norm += torch.norm(w_avg[key])
    return norm

# Loss_weighted_average
def weighted_average(w, beta):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(0, len(w)):
            if i == 0:
                w_avg[key] = w[i][key] * beta[0]
            else:
                w_avg[key] += w[i][key] * beta[i]
    return w_avg




# Staleness-aware grouping

def Sag(current_epoch, current_average, current_length, epoch_weights, global_weights):
    args = args_parser()
   
    alpha = []
    weights_d = []
    num_device = []

    alpha_for_attack = np.ones(args.staleness+1)

    comm = current_length
    print("curent length is ",current_length)
    print("the length of epoch_weights is ", len(epoch_weights))
    for i in epoch_weights:
        key = list(i.keys())[0]
        alpha.append(key)
        weights_d.append(i[key][0])
        num_device.append(i[key][1])
        comm = comm + i[key][1]

    # For empty staleness groups
    # You can ignore this part
    #########################################################################################
    if current_average is not None:
        w_semi = copy.deepcopy(current_average)

    else:
        for weigts_delay in weights_d:
            if weigts_delay is not None:
                w_semi = copy.deepcopy(weigts_delay)
                break
            

    if len(weights_d) > 0 and current_epoch >=args.staleness:
        if current_average is None:
            alpha_for_attack[0] = 0
        print("the length of weights_d is ", len(weights_d))

    #Staleness-based weight *#*****
    print("alpha 1 is",alpha)
    alphas = 1.0  /  ((current_epoch - np.array(alpha) + 1) )


    if len(alphas) == 0:
        alphas = np.array([alpha_for_attack[0]])
    else:
        alphas = np.concatenate((np.array([1]), alphas), axis=0)

    print("alphas 3 is",alphas)

    sum_alphas = sum(alphas)
    alphas = alphas / sum_alphas
    print("alphas is ",alphas)

    for key in w_semi.keys():
        for i in range(0, len(weights_d) + 1):
            if i == 0:
                w_semi[key] = w_semi[key] * (alphas[0])
            else:
                if weights_d[i-1] is None:
                    continue
                else:
                    w_semi[key] += weights_d[i - 1][key] * alphas[i]

    # w_semi_copy = copy.deepcopy(w_semi)
    for key in w_semi.keys():
        if args.dataset =='cifar':
            alpha = 0.8 ** (current_epoch // 300)
        elif args.dataset =='fmnist':
            alpha = 0.8

        elif args.dataset =='mnist':
            alpha = 0.5 ** (current_epoch // 15)
        elif args.dataset =='GTSRB':
            alpha = 0.8

        w_semi[key] = w_semi[key] * (alpha) + global_weights[key] * (1 - alpha)

    return w_semi


def Fedavg(args, current_epoch, all_weights, global_model):
    
    print("len all weights is", len(all_weights))
    # print("all_weights",all_weights)
    avg_weights = average_weights(all_weights)

    w_semi = copy.deepcopy(global_model.state_dict())
    for key in w_semi.keys():
        if args.dataset =='cifar':
            alpha = 0.8
        elif args.dataset =='fmnist':
            alpha = 0.8

        elif args.dataset =='mnist':
            alpha = 0.8
        else:
            alpha = 0.8
        w_semi[key] = w_semi[key] * (1 - alpha) + avg_weights[key] * (alpha)

    return w_semi, avg_weights


def communication_w(w, w_pre):

    w_com = copy.deepcopy(w)

    for key in w_com.keys():
        w_com[key] = w[key] - w_pre[key]

    return w_com

def receive_w(w, w_pre):

    w_com = copy.deepcopy(w)

    for key in w_com.keys():
        w_com[key] = w[key] + w_pre[key]

    return w_com




# Entropy based filtering and loss weighted averaging

def Eflow(w, loss, entropy, current_epoch, num_device=[]):

    args=args_parser()
    w_avg = copy.deepcopy(w[0])
    num_attack = 0
    alpha = []

    for j in range(0, len(loss)):
        # print("j  ", j)
        # print("entropy[j] " ,entropy[j] )
        if entropy[j] >= args.eth:
            norm_q = 0
            num_attack += 1
        else:
            norm_q = 1
        # print("norm_q is  ",norm_q)
        if len(num_device) == 0:
            # alpha.append(norm_q / loss[j] ** args.delta)
            alpha.append(norm_q)


    sum_alpha = sum(alpha)

    if sum_alpha <= 0.001:
        for k in range(0, len(alpha)):
            w_avg = None

    else:
        for k in range(0, len(alpha)):
            alpha[k] = alpha[k] / sum_alpha

        for key in w_avg.keys():
            for i in range(0, len(w)):
                if i == 0:
                    w_avg[key] = w_avg[key] * alpha[i]

                else:
                    w_avg[key] += w[i][key] * alpha[i]

    return w_avg, len(loss) -num_attack




def sign_attack(w,scale):
    w_avg = copy.deepcopy(w)
    for key in w_avg.keys():
        w_avg[key] = -1* w[key] * scale
    return w_avg




import tarfile  

import os  

  
  



def exp_details(args):
    print('\nExperimental details:')
    print(f'    Dataset     : {args.dataset}')
    print(f'    Model     : {args.model}')
    print(f'    detailed Model     : {args.detail_model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid == 1:
        print('    IID')
    elif args.iid == 2:
        print('    Non-IID with common data')


    else:
        print('    Non-IID')
    if args.unequal:
        print('    Unbalanced')
    else:
        print('    balanced')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')

    print(f'    Attack ratio : {args.attack_ratio}')
    if args.data_poison == True:
        print('     Data poison attack is done!')
    elif args.model_poison == True:
        print('     Model attack is done!')
    else:
        print('     None of attack is done!\n')

    return




