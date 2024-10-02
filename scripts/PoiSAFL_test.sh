#!/bin/bash
cd ../PoiSAFL_code

# For mnist and fmnist, epoch 150
# For cifar10, epoch: 1200

python Async_PoiSAFL.py --epoch 50 --update_rule Sageflow  --poison_methods ourpoisonMethod  --local_ep 5 --lrdecay 2000 --data_poison False  --lr 0.01  --new_poison False  --model_poison True --dataset GTSRB --frac 0.2 --attack_ratio 0.2 --gpu_number 0 --iid 1 --model_poison_scale 1.0 --eth 1 --delta 0.5 --lam 0.5 --seed 2021 --staleness 6 --scale_weight 40 --alpha 0.1