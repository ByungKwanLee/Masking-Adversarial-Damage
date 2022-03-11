'''Train CIFAR10/CIFAR100 with PyTorch.'''
from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn

from tqdm import tqdm
from utils.network_utils import get_network
from utils.data_utils import get_dataloader

# attack loader
from attack.attack import attack_loader

# warning ignore
import warnings
warnings.filterwarnings("ignore")
from utils.utils import str2bool


# fetch args
parser = argparse.ArgumentParser()

# model parameter
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--network', default='vgg', type=str)
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--baseline', default='adv', type=str)
parser.add_argument('--device', default='cuda:0', type=str)

# mad parameter
parser.add_argument('--percnt', default=0.9, type=float)
parser.add_argument('--pruning_mode', default='el', type=str)
parser.add_argument('--largest', default='false', type=str2bool)

# attack parameter
parser.add_argument('--attack', default='pgd', type=str)
parser.add_argument('--eps', default=0.03, type=float)
parser.add_argument('--steps', default=30, type=int)
args = parser.parse_args()


''' ------------------------------------------------------------------------------------------------------------- '''
if args.baseline == 'mad':
    if args.pruning_mode == 'rd':
        checkpoint_name = 'checkpoint/pretrain/%s/%s_%s_mad%s_%s%s_best.t7' % (args.dataset, args.pruning_mode, args.dataset, str(int(100*args.percnt)), args.network, args.depth)
        print("This test : {}".format(checkpoint_name))
    else:
        checkpoint_name = 'checkpoint/pretrain/%s/%s_%s_%s_mad%s_%s%s_best.t7' % (args.dataset, args.largest, args.pruning_mode, args.dataset, str(int(100*args.percnt)), args.network, args.depth)
        print("This test : {}".format(checkpoint_name))
else:
    checkpoint_name = 'checkpoint/pretrain/%s/%s_%s_%s%s_best.t7' % (args.dataset, args.dataset, args.baseline, args.network, args.depth)
    print("This test : {}".format(checkpoint_name))
''' ------------------------------------------------------------------------------------------------------------- '''

# init dataloader
_, testloader = get_dataloader(dataset=args.dataset,
                                         train_batch_size=1,
                                         test_batch_size=128)

# init model
net = get_network(network=args.network,
                  depth=args.depth,
                  dataset=args.dataset,
                  device=args.device)
net = net.to(args.device)


# Load Plain Network
print('==> Loading Plain checkpoint..')
assert os.path.isdir('checkpoint/pretrain'), 'Error: no checkpoint directory found!'


checkpoint = torch.load(checkpoint_name, map_location=args.device)
net.load_state_dict(checkpoint['net'], strict=False)


# init criterion
criterion = nn.CrossEntropyLoss()

# compute prune ratio
from models.operator.mask import compute_prune_ratio
p_ratio, param_size = compute_prune_ratio(net, is_param=True)
print("Prune Ratio : {:.2f}, Param size : {:.2f}".format( p_ratio, param_size))

def test():
    net.eval()
    test_loss = 0

    attack_score = []
    attack_module = {}
    for attack_name in ['Plain', 'fgsm', 'pgd', 'cw_Linf', 'apgd', 'auto']:
        args.attack = attack_name
        attack_module[attack_name] = attack_loader(net=net, attack=attack_name,
                                                   eps=args.eps, steps=args.steps,
                                                   dataset=args.dataset, device=args.device) \
            if attack_name != 'Plain' else None

    for key in attack_module:
        total = 0
        correct = 0
        prog_bar = tqdm(enumerate(testloader), total=len(testloader), leave=True)
        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            if key != 'Plain':
                inputs = attack_module[key](inputs, targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            desc = ('[Test/%s] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (key, test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)

        attack_score.append(100. * correct / total)

    print('\n----------------Summary----------------')
    print(args.steps, ' steps attack')
    for key, score in zip(attack_module, attack_score):
        print(str(key), ' : ', str(score) + '(%)')
    print('---------------------------------------\n')

if __name__ == '__main__':
    test()






