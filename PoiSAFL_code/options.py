import argparse


# Reference
# The original backbone code comes from
# https://github.com/AshwinRJ/Federated-Learning-PyTorch/tree/master/src


import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # Training parameter
    parser.add_argument('--epochs', type=int, default=300,
                        help="number of training rounds")
    parser.add_argument('--num_users', type=int, default=60,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.5,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=5,
                        help="number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lrdecay',type=float, default=2000, help="Learning rate decay every nth epoch")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--staleness', type=int, default=2,
                        help='maximum staleness)')
    parser.add_argument('--update_rule', type=str, default='Sageflow',
                        help='choose update rule of server')


    # The amount of Public data1
    parser.add_argument('--num_commondata', type=float, default=100,
                        help='number of public data which server has')
    
    # The alpha of non-iid distribution
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='alpha for dirichlet distribution')
    
    
    # For Adversarial attack
    parser.add_argument('--attack_ratio', type=float, default=0.1,
                        help='attack ratio')
    parser.add_argument('--data_poison', type=str2bool, default=True,
                        help='True: data poisoning attack, False: no attack')
    parser.add_argument('--model_poison', type=str2bool, default=False,
                        help='True: model poisoning attack, False: no attack')
    parser.add_argument('--model_poison_scale', type=float, default=10,
                        help='scale of model poisoning attack (0.1 or 10)')
    parser.add_argument('--inverse_poison', type=str2bool, default=False,
                        help='True: data poisoning attack, False: no attack')

    # Hyperparameters of Sageflow
    parser.add_argument('--eth', type=float, default=1,
                        help='Eth of Eflow')
    parser.add_argument('--delta', type=float, default=1,
                        help='Delta of Eflow')
    parser.add_argument('--lam', type=float, default=0.5,
                        help='lambda of Sag')


    # Other arguments
    parser.add_argument('--dataset', type=str, default='cifar', help="name \
                        of dataset: choose mnist or fmnist or cifar")
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--gpu', default=True, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--gpu_number', default=6, help="GPU number to use")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                            of classes")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")

    # Data distribution setting
    parser.add_argument('--iid', type=int, default=1,
                        help='Set to 1 for IID. Set to 0 for non-IID')

    # Detailed settings
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--detail_model', type=str, default='resnet', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                            use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                            of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                            mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                            strided convolutions")
    
    # new poisoning attack
    parser.add_argument('--dev_type', type=str, default='unit_vec',
                        help="sign, unit_vec, std")
    parser.add_argument('--threshold', type=float, default=20.0,
                        help="sign, unit_vec, std")
    parser.add_argument('--threshold_diff', type=float, default=1e-5,
                        help="sign, unit_vec, std")
    parser.add_argument('--new_poison', type=str2bool, default=False,
                        help='True: new poisoning attack, False: no attack')
    
    parser.add_argument('--poison_epoch', type=int, default=19,
                        help='the number of epoch when started attack')
    parser.add_argument('--poison_methods', type=str, default="ourpoisonMethod",
                        help='the number of epoch when started attack')
    
    # scale attack
    parser.add_argument('--scale_weight', type=int, default=100,
                        help='scale attack L = scale_weight / num_attacker * (ori- global) + ori ')

    args = parser.parse_args()
    return args


def str2bool(v):
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False