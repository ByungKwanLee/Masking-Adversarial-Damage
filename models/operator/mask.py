import torch
from torch.nn import Conv2d, Linear, Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import math

initial_mask = 1
name = 'softplus'
thresh = 2

def f(x, name=name):
    if name == 'softplus':
        return F.softplus(x)
    elif name == 'sigmoid':
        return torch.sigmoid(x)
    elif name == 'exp':
        return torch.exp(x)
    elif name == 'cov':
        return 1 / 2 * (torch.tanh(x) + 1)
    elif name == 'identity':
        return x
    elif name == 'tanh':
        return 0.01 * torch.tanh(x)


def f_inv(x, name=name):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x).float()
    if name == 'softplus':
        x = (0.0001) * (x == 0).float() + x * (x != 0).float()
        return torch.log(torch.exp(x) - 1)
    elif name == 'sigmoid':
        x = x * (x < 1).float() * (x > 0).float() + 0.999 * (x==1).float() + 0.001 * (x==0).float()
        return torch.log(x / (1-x))
    elif name == 'exp':
        x = 0.001 * (x == 0).float() + x * (x != 0).float()
        return torch.log(x)
    elif name == 'cov':
        return torch.atanh(2 * 0.99 * (x == 1).float() + 0.001 * (x == 0).float() + x * (0<x).float()*(x<1).float() - 1)
    elif name == 'identity':
        return x
    elif name == 'tanh':
        x = -0.999 * (x == -1).float() + 0.999 * (x == 1).float() + x * (-1 < x).float() * (x < 1).float()
        return torch.atanh(x / 0.01)


def operator(w, m):
    return w * f(m)



def compute_prune_ratio(net, is_param=False):
    count = 0
    w_shape = 0
    for name, param in net.named_parameters():
        if not 'mask' in name:
            count += (param == 0).sum().item()
            w_shape += param.shape.numel()

    if is_param:
        return int(count) / w_shape, w_shape
    else:
        return int(count) / w_shape



def clamping_mask_network(modules):
    for m in modules:
        m.mask_weight.data.clamp_(min=f_inv(0), max=f_inv(1))
        if m.bias is not None:
            m.mask_bias.data.clamp_(min=f_inv(0), max=f_inv(1))

def reinitialize_mask_network(modules):
    for m in modules:
        m.mask_weight.data = f_inv(initial_mask) * torch.ones_like(m.mask_weight.data)
        if m.bias is not None:
            m.mask_bias.data = f_inv(initial_mask) * torch.ones_like(m.mask_bias.data)

def index2mask(index, modules, device):
    mask_dict = {}
    index = index.to(device)
    i=0
    for m in modules:
        mask_weight = torch.ones_like(m.mask_weight)
        j = i + mask_weight.shape[0]
        mask_weight = f_inv(index[i:j]).view(mask_weight.shape)
        i=j
        mask_dict[m] = [mask_weight]

    return mask_dict

class Conv2d_mask(Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride = 1,
                 padding = 0,
                 dilation = 1,
                 groups = 1,
                 bias = True,
                 padding_mode = 'zeros',
                 ):
        super(Conv2d_mask, self).__init__(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
                padding_mode)

        self.bias_bool = bias
        self.mask_weight = Parameter(f_inv(initial_mask) * torch.ones(self.weight.shape))
        self.mask_bias   = Parameter(f_inv(initial_mask)*torch.ones_like(self.bias)) if bias else None

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            operator(weight, self.mask_weight), operator(self.bias, self.mask_bias) if self.bias_bool else self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, operator(weight, self.mask_weight), operator(self.bias, self.mask_bias) if self.bias_bool else self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class Linear_mask(Linear):
    def __init__(self, in_features, out_features, bias = True):
        super(Linear_mask, self).__init__(in_features, out_features, bias)
        self.mask_weight = Parameter(f_inv(initial_mask)*torch.ones(self.weight.shape))

        self.bias_bool = bias
        self.mask_bias = Parameter(f_inv(initial_mask)*torch.ones_like(self.bias)) if bias else None

    def forward(self, input):
        return F.linear(input, operator(self.weight, self.mask_weight), operator(self.bias, self.mask_bias) if self.bias_bool else self.bias)
