import torch
import torchattacks
from torchattacks.attack import Attack

class FGSM_train(Attack):

    def __init__(self, model, eps=0.007):
        super().__init__("FGSM_train", model)
        self.eps = eps
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        adv_images = images.clone().detach()

        # Starting at a uniformly random point
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = torch.nn.CrossEntropyLoss()

        adv_images.requires_grad = True
        outputs = self.model(adv_images)

        # Calculate loss
        if self._targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(cost, adv_images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images_ = adv_images.detach() + 1.25 * self.eps*grad.sign()
        delta = torch.clamp(adv_images_ - images, min=-self.eps, max=self.eps)
        return torch.clamp(images + delta, min=0, max=1).detach()

class CW_Linf(Attack):

    def __init__(self, model, eps, c=0.1, kappa=0, steps=1000, lr=0.01):
        super().__init__("CW_Linf", model)
        self.eps = eps
        self.alpha = eps/steps * 2.3
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        adv_images = images.clone().detach()


        # Starting at a uniformly random point
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()


        for step in range(self.steps):

            adv_images.requires_grad = True

            outputs = self.model(adv_images)
            f_loss = self.f(outputs, labels).sum()
            cost = f_loss

            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            adv_images = adv_images.detach() - self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

    def tanh_space(self, x):
        return 1/2*(torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        return self.atanh(x*2-1)

    def atanh(self, x):
        return 0.5*torch.log((1+x)/(1-x))

    # f-function in the paper
    def f(self, outputs, labels):
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())

        if self._targeted:
            return torch.clamp((i-j), min=-self.kappa)
        else:
            return torch.clamp((j-i), min=-self.kappa)

def attack_loader(net, attack, eps, steps, dataset, device):

    if dataset == 'cifar10':
        n_channel = 3
        n_classes = 10
        img_size = 32
    elif dataset == 'cifar100':
        n_channel = 3
        n_classes = 100
        img_size = 32
    elif dataset == 'svhn':
        n_channel = 3
        n_classes = 10
        img_size = 32
    elif dataset == 'tiny':
        n_channel = 3
        n_classes = 200
        img_size = 64

    # torch attacks
    if attack == "fgsm":
        return torchattacks.FGSM(model=net, eps=eps)

    elif attack == "fgsm_train":
        return FGSM_train(model=net, eps=eps)

    elif attack == "bim":
        return torchattacks.BIM(model=net, eps=eps, alpha=1/255)

    elif attack == "pgd":
        return torchattacks.PGD(model=net, eps=eps,
                                alpha=eps/steps*2.3, steps=steps, random_start=True)

    elif attack == "cw_linf":
        return CW_Linf(model=net, eps=eps, lr=0.1, steps=30)

    elif attack == "apgd":
        return torchattacks.APGD(model=net, eps=eps, loss='ce', steps=30)

    elif attack == "auto":
        return torchattacks.AutoAttack(model=net, eps=eps, n_classes=n_classes)
