from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from utils.mask_network_utils import get_mask_network
from utils.data_utils import get_dataloader

# attack loader
from attack.attack import attack_loader
from pruner.kfac_MAD_pruner import KFACMADPruner

from utils.mask_parameter_generator_utils import MaskParameterGenerator
from models.operator.mask import *


# fetch args
parser = argparse.ArgumentParser()


# model parameter
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--network', default='vgg', type=str)
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--device', default='cuda:0', type=str)
parser.add_argument('--batch_size', default=128, type=float)

# attack parameter
parser.add_argument('--attack', default='pgd', type=str)
parser.add_argument('--eps', default=0.03, type=float)
parser.add_argument('--steps', default=10, type=int)
args = parser.parse_args()


# init dataloader
trainloader, testloader = get_dataloader(dataset=args.dataset,
                                         train_batch_size=args.batch_size,
                                         test_batch_size=256,
                                         is_test=False)

# init model
net = get_mask_network(network=args.network,
                    depth=args.depth,
                    dataset=args.dataset,
                    device=args.device)
net = net.to(args.device)


# Load Plain Network
print('==> Loading Plain checkpoint..')
assert os.path.isdir('checkpoint/pretrain'), 'Error: no checkpoint directory found!'

''' ------------------------------------------------------------------------------------------------------------- '''
if not os.path.isdir('pickle'):
    os.mkdir('pickle')
checkpoint = torch.load('checkpoint/pretrain/%s/%s_adv_%s%s_best.t7' % (args.dataset, args.dataset, args.network, args.depth), map_location=args.device)
pickle_path = './pickle/%s_adv_%s%s_saliency.pickle' % (args.dataset, args.network, args.depth)
''' ------------------------------------------------------------------------------------------------------------- '''
print(pickle_path)
net.load_state_dict(checkpoint['net'], strict=False)

# Attack loader
attack = attack_loader(net=net, attack=args.attack, eps=args.eps, steps=args.steps, dataset=args.dataset, device=args.device)

# init criterion
criterion = nn.CrossEntropyLoss()

pruner = KFACMADPruner(net, attack, args.device, dataset=args.dataset)
mask_model = MaskParameterGenerator(net)


# [KFAC Masking Adversarial Damage (MAD)]
def optimizing_mask():

    total = 0
    correct = 0

    adv_delta_L_avg = []

    desc = ('[Mask Optimizing for total Dataset] R Acc: %.3f%% (%d/%d)' %
            (0, correct, total))
    prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        net.eval()
        adv_x = attack(inputs, targets)
        inputs, adv_x, targets = inputs.to(args.device), adv_x.to(args.device), targets.to(args.device)

        # Adv mask optimizer [KFAC Masking Adversarial Damage (MAD)]
        mask_optimizer = optim.Adam(mask_model.mask_parameters(), lr=0.1)
        _, adv_delta_L_list, _, a_outputs\
            = pruner._optimize_mask(adv_x, targets, mask_optimizer, mask_epoch=20, debug_acc=False, is_compute_delta_L=True)

        # performance validation
        _, a_predicted = a_outputs.max(1)
        a_num = a_predicted.eq(targets).sum().item()

        total += targets.size(0)
        correct += a_num

        # averaging
        if len(adv_delta_L_avg) == 0:

            adv_delta_L_avg = adv_delta_L_list

            adv_delta_L_avg = [x / len(trainloader) for x in adv_delta_L_avg]

        else:
            for index, l in enumerate(adv_delta_L_list):
                adv_delta_L_avg[index] += l / len(trainloader)


        desc = ('[Mask Optimizing for total Dataset] R Acc: %.1f%% (%d/%d)' %
                (100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

    # pickle dictionary by converting torch.tensor.cuda to cpu
    pickle_dict = {}
    pickle_dict['adv_delta_L_avg'] = [x.cpu() for x in adv_delta_L_avg]

    # save
    import pickle
    with open(pickle_path, 'wb') as f:
        pickle.dump(pickle_dict, f, pickle.HIGHEST_PROTOCOL)

onehot_dict = optimizing_mask()







