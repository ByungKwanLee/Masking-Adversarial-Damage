from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from utils.mask_network_utils import get_mask_network
from utils.data_utils import get_dataloader
from utils.common_utils import PresetLRScheduler

# attack loader
from attack.attack import attack_loader
from pruner.kfac_MAD_pruner import KFACMADPruner

from utils.mask_parameter_generator_utils import MaskParameterGenerator
from utils.utils import *
from models.operator.mask import *
from utils.utils import str2bool

# fetch args
parser = argparse.ArgumentParser()

# model parameter
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--network', default='vgg', type=str)
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--device', default='cuda:0', type=str)

# mad parameter
parser.add_argument('--percnt', default=0.9, type=float)
parser.add_argument('--pruning_mode', default='element', type=str)
parser.add_argument('--largest', default='false', type=str2bool)

# learning parameter
parser.add_argument('--learning_rate', default=0.1, type=float)
parser.add_argument('--weight_decay', default=0.0002, type=float)
parser.add_argument('--batch_size', default=128, type=float)
parser.add_argument('--epoch', default=60, type=int)

# attack parameter
parser.add_argument('--attack', default='pgd', type=str)
parser.add_argument('--eps', default=0.03, type=float)
parser.add_argument('--steps', default=10, type=int)
args = parser.parse_args()


# init dataloader
trainloader, testloader = get_dataloader(dataset=args.dataset,
                                         train_batch_size=args.batch_size,
                                         test_batch_size=256)

# init model
net = get_mask_network(network=args.network,
                    depth=args.depth,
                    dataset=args.dataset,
                    device=args.device)
net = net.to(args.device)


# Load Plain Network
print('==> Loading Adv checkpoint..')
assert os.path.isdir('checkpoint/pretrain'), 'Error: no checkpoint directory found!'

''' ------------------------------------------------------------------------------------------------------------- '''
if not os.path.isdir('pickle'):
    os.mkdir('pickle')
checkpoint = torch.load('checkpoint/pretrain/%s/%s_adv_%s%s_best.t7' % (args.dataset, args.dataset, args.network, args.depth), map_location=args.device)
pickle_path = './pickle/%s_adv_%s%s_saliency.pickle' % (args.dataset, args.network, args.depth)
print('checkpoint/pretrain/%s/%s_adv_%s%s_best.t7' % (args.dataset, args.dataset, args.network, args.depth))
''' ------------------------------------------------------------------------------------------------------------- '''
net.load_state_dict(checkpoint['net'], strict=False)

# Attack loader
if args.dataset=='tiny':
    print('Fast FGSM training')
    attack = attack_loader(net=net, attack='fgsm_train', eps=args.eps, steps=args.steps, dataset=args.dataset, device=args.device)
else:
    print('Low PGD training')
    attack = attack_loader(net=net, attack=args.attack, eps=args.eps, steps=args.steps, dataset=args.dataset, device=args.device)

# init criterion
criterion = nn.CrossEntropyLoss()

start_epoch = 0
best_acc = 0

pruner = KFACMADPruner(net, attack, args.device, dataset=args.dataset)
mask_model = MaskParameterGenerator(net)

# [KFAC Masking Adversarial Damage (MAD)]
def pruning_model():
    # load
    import pickle

    with open(pickle_path, 'rb') as f:

        pickle_dict = pickle.load(f)
        adv_delta_L_avg = pickle_dict['adv_delta_L_avg']

        if args.pruning_mode == 'element':
            onehot_dict = pruner._global_remove_weight(adv_delta_L_avg, percnt=args.percnt, largest=args.largest)
        elif args.pruning_mode == 'random':
            onehot_dict = pruner._global_random_pruning(adv_delta_L_avg, percnt=args.percnt)

        print("{} Successfully Pruned !!".format(pickle_path))
        print("Pruning Ratio Per Layer is : {}".format(pruner._compute_prune_ratio_per_layer()))
        return onehot_dict


def train(epoch, onehot_dict, mode):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    p_ratio = compute_prune_ratio(net)

    lr_scheduler(optimizer, epoch)
    desc = ('[Train/%s/LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d), Prune: %.2f' %
            (mode, lr_scheduler.get_lr(optimizer), 0, 0, correct, total, p_ratio))

    prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        net.train()
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        loss, outputs = pruner._optimize_weight_with_delta_L(inputs, targets, optimizer, onehot_dict,
                                                                    pruning_mode=args.pruning_mode)
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = ('[Train/%s/LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d), Prune: %.2f' %
                (mode, lr_scheduler.get_lr(optimizer), train_loss / (batch_idx + 1), 100. * correct / total, correct, total, p_ratio))
        prog_bar.set_description(desc, refresh=True)


def test(epoch, is_attack=False):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    print("Pruning Ratio Per Layer is : {}".format(pruner._compute_prune_ratio_per_layer()))
    desc = ('[Test/LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (lr_scheduler.get_lr(optimizer), test_loss/(0+1), 0, correct, total))

    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs = attack(inputs, targets) if is_attack else inputs
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = ('[Test/LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (lr_scheduler.get_lr(optimizer), test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

    if is_attack:
        return

    # Save checkpoint.
    acc = 100.*correct/total

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'loss': loss,
            'args': args
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if not os.path.isdir('checkpoint/pretrain'):
            os.mkdir('checkpoint/pretrain')

        if args.pruning_mode=='element':
            torch.save(state, './checkpoint/pretrain/%s/%s_el_%s_mad%d_%s%s_best.t7' % (args.dataset, str(args.largest), args.dataset,
                                                                               int(args.percnt*100),
                                                                         args.network,
                                                                         args.depth))
            print('./checkpoint/pretrain/%s/%s_el_%s_mad%d_%s%s_best.t7' % (args.dataset, str(args.largest), args.dataset,
                                                                    int(args.percnt*100),
                                                                         args.network,
                                                                         args.depth))


        elif args.pruning_mode=='random':
            torch.save(state, './checkpoint/pretrain/%s/rd_%s_mad%d_%s%s_best.t7' % (args.dataset, args.dataset,
                                                                               int(args.percnt*100),
                                                                         args.network,
                                                                         args.depth))
            print('./checkpoint/pretrain/%s/rd_%s_mad%d_%s%s_best.t7' % (args.dataset, args.dataset,
                                                                    int(args.percnt*100),
                                                                         args.network,
                                                                         args.depth))
        best_acc = acc



onehot_dict = pruning_model()

# MAD init optimizer and lr scheduler
optimizer = optim.SGD(mask_model.non_mask_parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
lr_schedule = {0: args.learning_rate,
               int(args.epoch*0.5): args.learning_rate*0.1,
               int(args.epoch*0.75): args.learning_rate*0.01}
lr_scheduler = PresetLRScheduler(lr_schedule)

print("--------------MAD--------------")
for epoch in range(start_epoch, args.epoch):
    train(epoch, onehot_dict)
    test(epoch, is_attack=False)
    test(epoch, is_attack=True)






