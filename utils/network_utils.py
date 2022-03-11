import torch
from models.vgg import VGG
from models.resnet import resnet
from models.wide import wide_resnet


def get_network(network, depth, dataset, device=None):

    if dataset == 'cifar10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).to(device)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).to(device)
    elif dataset == 'svhn': # later, it should be updated
        mean = torch.tensor([0.43090966, 0.4302428, 0.44634357]).to(device)
        std = torch.tensor([0.19759192, 0.20029082, 0.19811132]).to(device)
    elif dataset == 'cifar100':
        mean = torch.tensor([0.5071, 0.4867, 0.4408]).to(device)
        std = torch.tensor([0.2675, 0.2565, 0.2761]).to(device)
    elif dataset == 'tiny':
        mean = torch.tensor([0.48024578664982126, 0.44807218089384643, 0.3975477478649648]).to(device)
        std = torch.tensor([0.2769864069088257, 0.26906448510256, 0.282081906210584]).to(device)

    if network == 'vgg':
        return VGG(depth=depth, dataset=dataset, mean=mean, std=std)
    elif network == 'resnet':
        return resnet(depth=depth, dataset=dataset, mean=mean, std=std)
    elif network == 'wide':
        return wide_resnet(depth=depth, widen_factor=10, dataset=dataset, mean=mean, std=std)
    else:
        raise NotImplementedError


def stablize_bn(net, trainloader, device='cuda'):
    """Iterate over the dataset for stabilizing the
    BatchNorm statistics.
    """
    net = net.train()
    for batch, (inputs, _) in enumerate(trainloader):
        inputs = inputs.to(device)
        net(inputs)