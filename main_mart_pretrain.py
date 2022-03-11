from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from utils.network_utils import get_network
from utils.data_utils import get_dataloader
from utils.common_utils import PresetLRScheduler

# attack loader
from attack.attack import attack_loader

# fetch args
parser = argparse.ArgumentParser()

# model parameter
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--network', default='vgg', type=str)
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--device', default='cuda:0', type=str)

# learning parameter
parser.add_argument('--learning_rate', default=1e-3, type=float)
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
net = get_network(network=args.network,
                  depth=args.depth,
                  dataset=args.dataset,
                  device=args.device)
net = net.to(args.device)


# Load Adv Network
print('==> Loading Adv checkpoint..')
assert os.path.isdir('checkpoint/pretrain'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('checkpoint/pretrain/%s/%s_adv_%s%s_best.t7' % (args.dataset, args.dataset, args.network, args.depth), map_location=args.device)
print('checkpoint/pretrain/%s/%s_adv_%s%s_best.t7' % (args.dataset, args.dataset, args.network, args.depth))
net.load_state_dict(checkpoint['net'])

# Attack loader
if args.dataset=='tiny':
    print('Fast FGSM training')
    attack = attack_loader(net=net, attack='fgsm_train', eps=args.eps, steps=args.steps, dataset=args.dataset, device=args.device)
else:
    print('Low PGD training')
    attack = attack_loader(net=net, attack=args.attack, eps=args.eps, steps=args.steps, dataset=args.dataset, device=args.device)


# init optimizer and lr scheduler
optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
lr_schedule = {0: args.learning_rate,
               int(args.epoch*0.5): args.learning_rate*0.1,
               int(args.epoch*0.75): args.learning_rate*0.01}
lr_scheduler = PresetLRScheduler(lr_schedule)
# lr_scheduler = #StairCaseLRScheduler(0, args.decay_every, args.decay_ratio)

# init criterion
criterion = nn.CrossEntropyLoss()

start_epoch = 0
best_acc = 0
if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint/pretrain'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('checkpoint/pretrain/%s/%s_adv_%s%s_bn_best.t7' % (args.dataset, args.dataset, args.network, args.depth))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print('==> Loaded checkpoint at epoch: %d, acc: %.2f%%' % (start_epoch, best_acc))


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    lr_scheduler(optimizer, epoch)
    desc = ('[Train/LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (lr_scheduler.get_lr(optimizer), 0, 0, correct, total))

    prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        optimizer.zero_grad()

        # MART Loss
        loss, logits = mart_loss_orig(net, inputs, targets, optimizer, device=args.device)

        loss.backward()
        optimizer.step()


        train_loss += loss.item()
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = ('[Train/LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (lr_scheduler.get_lr(optimizer), train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)



def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    desc = ('[Test/LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (lr_scheduler.get_lr(optimizer), test_loss/(0+1), 0, correct, total))

    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs = attack(inputs, targets)
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
        torch.save(state, './checkpoint/pretrain/%s/%s_mart_%s%s_best.t7' % (args.dataset, args.dataset,
                                                                     args.network,
                                                                     args.depth))
        print('./checkpoint/pretrain/%s/%s_mart_%s%s_best.t7' % (args.dataset, args.dataset,
                                                                     args.network,
                                                                     args.depth))
        best_acc = acc




def mart_loss_orig(model,
              x_natural,
              y,
              optim,
              device,
              step_size=0.03/10*2.3 if args.dataset != 'tiny' else 0.03 * 1.25,
              epsilon=0.03,
              perturb_steps=10 if args.dataset != 'tiny' else 1,
              beta=5,
              distance='l_inf'):
    kl = torch.nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)
    if args.dataset != 'tiny':
        # generate adversarial example
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
    else:
        # generate adversarial example
        x_adv = x_natural.detach() + torch.empty_like(x_natural).uniform_(-epsilon, epsilon).to(device).detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = torch.autograd.Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optim.zero_grad()

    logits = model(x_natural)

    logits_adv = model(x_adv)

    adv_probs = F.softmax(logits_adv, dim=1)

    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

    nat_probs = F.softmax(logits, dim=1)

    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    loss = loss_adv + float(beta) * loss_robust

    return loss, logits_adv



for epoch in range(start_epoch, args.epoch):
    train(epoch)
    test(epoch)






