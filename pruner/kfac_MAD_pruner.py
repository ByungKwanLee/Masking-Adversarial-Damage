from collections import OrderedDict
from utils.kfac_utils import (ComputeCovA,
                              ComputeCovG,
                              fetch_mat_weights,
                              fetch_mat_mask_weights,)

from utils.utils import *
from models.operator.mask import *


class KFACMADPruner:

    def __init__(self,
                 model,
                 attack,
                 device,
                 batch_averaged=True,
                 dataset = None
                 ):
        self.iter = 0
        self.device = device
        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()
        self.batch_averaged = batch_averaged
        self.mask_known_modules = {'Conv2d_mask', 'Linear_mask'}
        self.modules = []
        self.model = model
        self.attack = attack
        self.steps = 0
        self.dataset = dataset

        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}
        self.W_pruned = {}
        self.S_l = None


    def _save_input(self, module, input):
        aa = self.CovAHandler(input[0].data, module)
        # Initialize buffers
        if self.steps == 0:
            self.m_aa[module] = torch.diag(aa.new(aa.size(0)).fill_(0))
        self.m_aa[module] += aa

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        gg = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)
        # Initialize buffers
        if self.steps == 0:
            self.m_gg[module] = torch.diag(gg.new(gg.size(0)).fill_(0))
        self.m_gg[module] += gg


    def _mask_prepare_model(self):
        count = 0
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.mask_known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)
                count += 1


    def _compute_minibatch_fisher(self, inputs, targets, device='cuda', fisher_type=True):

        self.model = self.model.eval()
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = self.model(inputs)
        if fisher_type:
            sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs.cpu().data, dim=1),
                                          1).squeeze().to(device)
            loss_sample = F.cross_entropy(outputs, sampled_y)
            loss_sample.backward()
        else:
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
        self.steps = 1

    def _rm_mask_hooks(self):
        for m in self.model.modules():
            classname = m.__class__.__name__
            if classname in self.mask_known_modules:
                m._backward_hooks = OrderedDict()
                m._forward_pre_hooks = OrderedDict()

    def _clear_buffer(self):
        self.m_aa = {}
        self.m_gg = {}
        self.d_a = {}
        self.d_g = {}
        self.Q_a = {}
        self.Q_g = {}
        self.modules = []
        if self.S_l is not None:
            self.S_l = {}

        self.m_a = {}
        self.m_g = {}
        self.steps = 0

    def _optimize_mask(self, inputs, targets, optim, mask_epoch, debug_acc=True, is_compute_delta_L=False):

        self.model = self.model.eval()
        self._mask_prepare_model()
        self._rm_mask_hooks()
        reinitialize_mask_network(self.modules)

        if debug_acc:
            self.debug(inputs, targets, str='No update/Mask')

        loss = 0
        for _ in range(mask_epoch):
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, targets)

            # optimizing mask
            optim.zero_grad()
            self.model.zero_grad()
            loss.backward()
            optim.step()
            clamping_mask_network(self.modules)

        if debug_acc:
            self.debug(inputs, targets, str='Update/Mask')

        self._clear_buffer()
        if is_compute_delta_L:
            self._mask_prepare_model()
            self._compute_minibatch_fisher(inputs, targets, self.device, False)
            self._rm_mask_hooks()
            delta_L_list, mask_list = self._compute_delta_L() # weight delta L
            outputs = self.model(inputs)
            reinitialize_mask_network(self.modules)
            self._clear_buffer()
            return loss, delta_L_list, mask_list, outputs
        else:
            return loss


    def _optimize_weight_with_delta_L(self, inputs, targets, optim,
                                             onehot_dict=None, pruning_mode='element'):

        self._mask_prepare_model()
        self._rm_mask_hooks()

        self.model = self.model.train()
        optim.zero_grad()

        loss, logit = self.loss_function(model=self.model,
                                    x_natural=inputs,
                                    y=targets,
                                    optim=optim,
                                    device=self.device,
                                    step_size=0.03/10*2.3 if self.dataset != 'tiny' else 0.03 * 1.25,
                                    epsilon=0.03,
                                    perturb_steps=10 if self.dataset != 'tiny' else 1,
                                    beta=0.5)
        loss.backward()

        if onehot_dict is not None:
            if (pruning_mode == 'element') or (pruning_mode == 'random'):
                for m in self.modules:
                    onehot = onehot_dict[m]
                    if m.bias is not None:
                        m.bias.grad.data *= (1 - onehot)[:, -1].view(m.bias.shape)
                        m.bias.data *= (1 - onehot)[:, -1].view(m.bias.shape)

                        m.weight.grad.data *= (1 - onehot)[:, :-1].view(m.weight.shape)
                        m.weight.data *= (1 - onehot)[:, :-1].view(m.weight.shape)
                    else:
                        m.weight.grad.data *= (1 - onehot).view(m.weight.shape)
                        m.weight.data *= (1 - onehot).view(m.weight.shape)
        optim.step()
        self._clear_buffer()

        return loss, logit

    def _global_remove_weight(self, adv_delta_L_avg, percnt, largest):
        print("--------------GLOBAL PRUNING--------------")
        self._mask_prepare_model()
        self._rm_mask_hooks()
        onehot_dict = {}

        # convolution layer pruning
        concat_conv, linear, size = list_to_concat(adv_delta_L_avg, self.device)
        val, ind = concat_conv.sort(descending=largest)
        sort_ind = ind[:int(percnt * val.size(0))]
        conv_onehot = index2onehot(sort_ind, size=ind.shape[0]).to(self.device)


        # linear layer pruning
        val, ind = linear.sort(descending=largest, dim=1)
        linear_onehot = torch.zeros_like(linear).to(self.device)
        for i in range(linear.shape[0]):
            one = torch.zeros(linear.size(1)).to(self.device)
            one[:int(percnt * linear.size(1))] = 1
            linear_onehot.data[i, ind[i]] = one

        initial = 0
        for index, m in enumerate(self.modules):

            if 'Linear' in m._get_name():

                onehot_dict[m] = linear_onehot

                m.weight.data *= (1 - onehot_dict[m][:, :-1]).view(m.weight.shape)
                m.bias.data *= (1 - onehot_dict[m][:, -1]).view(m.bias.shape)

                m.mask_weight.data = f_inv(1) * torch.ones_like(m.mask_weight)
                m.mask_bias.data = f_inv(1) * torch.ones_like(m.mask_bias)

                initial += size[index].numel()
                break


            if m.bias is not None:

                onehot_dict[m] = conv_onehot[initial:initial + size[index].numel()].view(size[index])

                m.weight.data *= (1 - onehot_dict[m][:, :-1]).view(m.weight.shape)
                m.bias.data *= (1 - onehot_dict[m][:, -1]).view(m.bias.shape)

                m.mask_weight.data = f_inv(1) * torch.ones_like(m.mask_weight)
                m.mask_bias.data = f_inv(1) * torch.ones_like(m.mask_bias)

                initial += size[index].numel()
            else:
                onehot_dict[m] = conv_onehot[initial:initial+size[index].numel()].view(size[index])

                m.weight.data *= (1 - onehot_dict[m]).view(m.weight.shape)
                m.mask_weight.data = f_inv(1) * torch.ones_like(m.mask_weight)

                initial += size[index].numel()

        assert initial == conv_onehot.shape.numel() + linear_onehot.shape.numel()

        self._clear_buffer()
        return onehot_dict

    def _global_random_pruning(self, adv_delta_L_avg, percnt):
        print("--------------Random PRUNING--------------")
        self._mask_prepare_model()
        self._rm_mask_hooks()
        onehot_dict = {}

        # convolution layer pruning
        concat_conv, linear, size = list_to_concat(adv_delta_L_avg, self.device)
        sort_ind = torch.randperm(concat_conv.shape[0])[:int(percnt * concat_conv.size(0))]
        conv_onehot = index2onehot(sort_ind, size=concat_conv.shape[0]).to(self.device)

        # linear layer pruning
        linear_onehot = torch.zeros_like(linear).to(self.device)
        for i in range(linear.shape[0]):
            sort_ind = torch.randperm(linear_onehot.shape[1])[:int(percnt * linear_onehot.shape[1])]
            onehot = index2onehot(sort_ind, size=linear_onehot.shape[1]).to(self.device)
            linear_onehot.data[i] = onehot

        initial = 0
        for index, m in enumerate(self.modules):

            if 'Linear' in m._get_name():
                onehot_dict[m] = linear_onehot

                m.weight.data *= (1 - onehot_dict[m][:, :-1]).view(m.weight.shape)
                m.bias.data *= (1 - onehot_dict[m][:, -1]).view(m.bias.shape)

                m.mask_weight.data = f_inv(1) * torch.ones_like(m.mask_weight)
                m.mask_bias.data = f_inv(1) * torch.ones_like(m.mask_bias)

                initial += size[index].numel()
                break

            if m.bias is not None:

                onehot_dict[m] = conv_onehot[initial:initial + size[index].numel()].view(size[index])

                m.weight.data *= (1 - onehot_dict[m][:, :-1]).view(m.weight.shape)
                m.bias.data *= (1 - onehot_dict[m][:, -1]).view(m.bias.shape)

                m.mask_weight.data = f_inv(1) * torch.ones_like(m.mask_weight)
                m.mask_bias.data = f_inv(1) * torch.ones_like(m.mask_bias)

                initial += size[index].numel()
            else:
                onehot_dict[m] = conv_onehot[initial:initial + size[index].numel()].view(size[index])

                m.weight.data *= (1 - onehot_dict[m]).view(m.weight.shape)
                m.mask_weight.data = f_inv(1) * torch.ones_like(m.mask_weight)

                initial += size[index].numel()

        assert initial == conv_onehot.shape.numel() + linear_onehot.shape.numel()

        self._clear_buffer()
        return onehot_dict


    def _compute_prune_ratio_per_layer(self):
        self._mask_prepare_model()
        self._rm_mask_hooks()

        layer_prune_ratio = []
        for m in self.modules:

            count = 0
            shape = 0

            if m.bias is not None:
                count += (m.weight == 0).sum().item()
                shape += m.weight.shape.numel()

                count += (m.bias == 0).sum().item()
                shape += m.bias.shape.numel()
            else:
                count += (m.weight == 0).sum().item()
                shape += m.weight.shape.numel()

            layer_prune_ratio.append(round(int(count)/shape, 3))

        self._clear_buffer()
        return layer_prune_ratio




    def debug(self, inputs, targets, str=''):
        self.model = self.model.eval()
        _, predicted = self.model(inputs).max(1)
        total = targets.size(0)
        correct = predicted.eq(targets).sum().item()
        print("[{}] Acc is {}".format(str, correct / total))



    def _compute_delta_L(self):

        delta_L_list = []
        mask_list = []
        for idx, m in enumerate(self.modules):

            m_aa, m_gg = self.m_aa[m], self.m_gg[m]

            w = fetch_mat_weights(m)
            mask = fetch_mat_mask_weights(m)
            w_mask = w - operator(w, mask)

            double_grad_L = torch.empty_like(w_mask)

            # 1/2 * Δ𝑤^𝑇 *𝐻 * Δ𝑤
            for i in range(m_gg.shape[0]):
                block1 = 0.5 * m_gg[i, i] * w_mask.t()[:, i].view(-1, 1)
                block2 = w_mask[i].view(1, -1) @ m_aa
                block = block1 @ block2
                double_grad_L[i, :] = block.diag()

            delta_L = double_grad_L
            delta_L_list.append(delta_L.detach())
            mask_list.append(f(mask).detach())

        return delta_L_list, mask_list


    @staticmethod
    def loss_function(model,
                  x_natural,
                  y,
                  optim,
                  device,
                  step_size,
                  epsilon,
                  perturb_steps,
                  beta,
                  distance='l_inf'):
        kl = torch.nn.KLDivLoss(reduction='none')
        model.eval()
        batch_size = len(x_natural)
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
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