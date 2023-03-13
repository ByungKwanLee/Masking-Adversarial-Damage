![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)
# CVPR 2022
[![Generic badge](https://img.shields.io/badge/Library-Pytorch-green.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ByungKwanLee/Masking-Adversarial-Damage/blob/master/LICENSE) 

# Title: [Masking Adversarial Damage: Finding Adversarial Saliency for Robust and Sparse Network](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_Masking_Adversarial_Damage_Finding_Adversarial_Saliency_for_Robust_and_Sparse_CVPR_2022_paper.pdf)



---


#### Authors: [Byung-Kwan Lee*](https://scholar.google.co.kr/citations?user=rl0JXCQAAAAJ&hl=en), [Junho Kim*](https://scholar.google.com/citations?user=ZxE16ZUAAAAJ&hl=en), and [Yong Man Ro](https://scholar.google.co.kr/citations?user=IPzfF7cAAAAJ&hl=en) (*: equally contributed)
#### Affiliation: School of Electrical Engineering, Korea Advanced Institute of Science and Technology (KAIST)
#### Email: `leebk@kaist.ac.kr`, `arkimjh@kaist.ac.kr`, `ymro@kaist.ac.kr`


---

This is official PyTorch Implementation code for the paper of "Masking Adversarial Damage: Finding Adversarial Saliency 
for Robust and Sparse Network" accepted in CVPR 2022. To bridge adversarial robustness and model compression, we propose a
novel adversarial pruning method, Masking Adversarial Damage (MAD) that employs second-order information of adversarial loss.
By using it, we can accurately estimate adversarial saliency for model parameters and determine which parameters can be 
pruned without weakening adversarial robustness.

<p align="center">
<img src="figure/adversarial saliency.png" width="760" height="200">
</p>


Furthermore, we reveal that model parameters of initial layer are highly sensitive to the adversarial examples and show that compressed feature representation retains semantic information for the target objects.

<p align="center">
<img src="figure/semantic information.png" width="465" height="350">
</p>

Through extensive experiments on public datasets, we demonstrate that MAD effectively prunes adversarially trained 
networks without loosing adversarial robustness and shows better performance than previous adversarial pruning methods.
For more detail, you can refer to our paper that will be accessible to public soon!.

<p align="center">
<img src="figure/pruning ratio.png"  width="720" height="300">
</p>

Adversarial attacks can potentially cause negative impacts on various DNN applications due to high computation and its 
fragility. By pruning model parameters without weakening adversarial robustness, our work contributes important societal 
impacts in this research area. Furthermore, in our promising observation that model parameters of initial layer are highly 
sensitive to adversarial loss, we hope to progress in another future direction of utilizing such property to enhance adversarial robustness.

In conclusion, in order to achieve adversarial robustness and model compression concurrently, we propose a novel adversarial pruning method, 
Masking Adversarial Damage (MAD). By exploiting second-order information with mask optimization and Block-wise K-FAC, 
we can precisely estimate adversarial saliency of the whole parameters. Through extensive validations, we corroborate 
pruning model parameters in order of low adversarial saliency retains adversarial robustness while alleviating less performance 
degradation compared with previous adversarial pruning methods.

---

## Datasets
* [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (32x32, 10 classes)
* [SVHN](http://ufldl.stanford.edu/housenumbers/) (32x32, 10 classes)
* [Tiny-ImageNet](https://www.kaggle.com/c/tiny-imagenet/overview) (64x64, 200 classes)

---

## Networks

* [VGG-16](https://arxiv.org/pdf/1409.1556) (models/vgg.py)
* [ResNet-18](https://arxiv.org/pdf/1512.03385) (models/resnet.py)
* [WideResNet-28-10](https://arxiv.org/abs/1605.07146) (models/wide.py)


---

## Masking Adversarial Damage (MAD)
#### Step 1. Finding Adversarial Saliency
* Run `compute_saliency.py` *(Procedure of saving a pickle file for adversarial saliency to all model parameters. Then, you should need a folder (e.g., `pickle` folder) in which the pickle file is saved)*

```bash
# model parameter
parser.add_argument('--dataset', default='cifar10', type=str) # 'svhn', 'tiny'
parser.add_argument('--network', default='vgg', type=str) # 'resnet', 'wide'
parser.add_argument('--depth', default=16, type=int) # 18 (ResNet), 28 (WideResNet)
parser.add_argument('--device', default='cuda:0', type=str)
parser.add_argument('--batch_size', default=128, type=float)

# attack parameter
parser.add_argument('--attack', default='pgd', type=str)
parser.add_argument('--eps', default=0.03, type=float)
parser.add_argument('--steps', default=10, type=int)
```

Among codes for running `compute_saliency`, the following code represents the major contribution of our work that is the procedure
of computing adversarial saliency realized with Block-wise K-FAC. Note that it is important to consider the factors of 
`block1` and `block2` below for Block-wise K-FAC that dramatically reduces computation. 

```python
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
```



#### Step 2. Pruning Low Advesarial Saliency
* Run `main_mad_pretrain.py` *(Necessary to load a pickle file generated in Step 1)*
```bash
# model parameter
parser.add_argument('--dataset', default='cifar10', type=str) # 'svhn', 'tiny'
parser.add_argument('--network', default='vgg', type=str) # 'resnet', 'wide'
parser.add_argument('--depth', default=16, type=int) # 18 (ResNet), 28 (WideResNet)
parser.add_argument('--device', default='cuda:0', type=str)

# mad parameter
parser.add_argument('--percnt', default=0.9, type=float) # 0.99 (Sparsity)
parser.add_argument('--pruning_mode', default='element', type=str) # 'random' (randomly pruning)
parser.add_argument('--largest', default='false', type=str2bool) # 'true' (pruning high adversarial saliency)

# learning parameter
parser.add_argument('--learning_rate', default=0.1, type=float)
parser.add_argument('--weight_decay', default=0.0002, type=float)
parser.add_argument('--batch_size', default=128, type=float)
parser.add_argument('--epoch', default=60, type=int)

# attack parameter
parser.add_argument('--attack', default='pgd', type=str)
parser.add_argument('--eps', default=0.03, type=float)
parser.add_argument('--steps', default=10, type=int)
```

---

## Adversarial Training (+ Recent Adversarial Defenses)

* [AT](https://arxiv.org/abs/1706.06083) (main_adv_pretrain.py)
* [TRADES](https://arxiv.org/abs/1901.08573) (main_trades_pretrain.py)
* [MART](https://openreview.net/forum?id=rklOg6EFwS) (main_mart_pretrain.py)
* [FAST](https://openreview.net/forum?id=BJx040EFvH) for Tiny-ImageNet (refer to **FGSM_train** class in *attack/attack.py*)

### *Running Adversarial Training*

To easily make an adversarially trained model, we first train a standard model by [1]
and perform adversarial training (AT) by [2], starting from the trained standard model. To execute recent adversarial defenses, AT model created by [2]
would be helpful to train TRADES or MART through [3-1] or [3-2].

* **[1] Plain**  (Plain Training)
    - Run `main_pretrain.py`
  
    ```bash
      # model parameter
      parser.add_argument('--dataset', default='cifar10', type=str) # 'svhn', 'tiny'
      parser.add_argument('--network', default='vgg', type=str) # 'resnet', 'wide'
      parser.add_argument('--depth', default=16, type=int) # 18 (ResNet), 28 (WideResNet)
      parser.add_argument('--epoch', default=200, type=int)
      parser.add_argument('--device', default='cuda:0', type=str)
      
      # learning parameter
      parser.add_argument('--learning_rate', default=0.1, type=float)
      parser.add_argument('--weight_decay', default=0.0002, type=float)
      parser.add_argument('--batch_size', default=128, type=float)
    ```
  
* **[2] AT**     ([PGD Adversarial Training](https://openreview.net/forum?id=rJzIBfZAb))
    - Run `main_adv_pretrain.py`

    ```bash
      # model parameter
      parser.add_argument('--dataset', default='cifar10', type=str) # 'svhn', 'tiny'
      parser.add_argument('--network', default='vgg', type=str) # 'resnet', 'wide'
      parser.add_argument('--depth', default=16, type=int) # 18 (ResNet), 28 (WideResNet)
      parser.add_argument('--device', default='cuda:0', type=str)
      
      # learning parameter
      parser.add_argument('--learning_rate', default=0.1, type=float)
      parser.add_argument('--weight_decay', default=0.0002, type=float)
      parser.add_argument('--batch_size', default=128, type=float)
      parser.add_argument('--epoch', default=60, type=int)
      
      # attack parameter
      parser.add_argument('--attack', default='pgd', type=str)
      parser.add_argument('--eps', default=0.03, type=float)
      parser.add_argument('--steps', default=10, type=int)
    ```

  
* **[3-1] TRADES**  ([Recent defense method](http://proceedings.mlr.press/v97/zhang19p.html))
    - Run `main_trades_pretrain.py`
  
    ```bash
      # model parameter  
      parser.add_argument('--dataset', default='cifar10', type=str) # 'svhn', 'tiny'
      parser.add_argument('--network', default='vgg', type=str) # 'resnet', 'wide'
      parser.add_argument('--depth', default=16, type=int) # 18 (ResNet), 28 (WideResNet)
      parser.add_argument('--device', default='cuda:0', type=str)
      
      # learning parameter
      parser.add_argument('--learning_rate', default=1e-3, type=float)
      parser.add_argument('--weight_decay', default=0.0002, type=float)
      parser.add_argument('--batch_size', default=128, type=float)
      parser.add_argument('--epoch', default=10, type=int)
      
      # attack parameter
      parser.add_argument('--attack', default='pgd', type=str)
      parser.add_argument('--eps', default=0.03, type=float)
      parser.add_argument('--steps', default=10, type=int)
    ```


* **[3-2] MART**  ([Recent defense method](https://openreview.net/forum?id=rklOg6EFwS))
    - Run `main_mart_pretrain.py`
  
    ```bash
      # model parameter
      parser.add_argument('--dataset', default='cifar10', type=str) # 'svhn', 'tiny'
      parser.add_argument('--network', default='vgg', type=str) # 'resnet', 'wide'
      parser.add_argument('--depth', default=16, type=int) # 18 (ResNet), 28 (WideResNet)
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
    ```

---


## Adversarial Attacks (by [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch))
* Fast Gradient Sign Method ([FGSM](https://arxiv.org/abs/1412.6572))
* Projected Gradient Descent ([PGD](https://arxiv.org/abs/1706.06083))
* Carlini & Wagner ([CW](https://arxiv.org/abs/1608.04644))
* AutoPGD ([AP](https://arxiv.org/abs/2003.01690))
* AutoAttack ([AA](https://arxiv.org/abs/2003.01690))

This implementation details for the adversarial attacks are described in *attack/attack.py*.

  ```bash
  # torchattacks
  if attack == "fgsm":
      return torchattacks.FGSM(model=net, eps=eps)

  elif attack == "fgsm_train":
      return FGSM_train(model=net, eps=eps)

  elif attack == "pgd":
      return torchattacks.PGD(model=net, eps=eps, alpha=eps/steps*2.3, steps=steps, random_start=True)

  elif attack == "cw_linf":
      return CW_Linf(model=net, eps=eps, lr=0.1, steps=30)

  elif attack == "apgd":
      return torchattacks.APGD(model=net, eps=eps, loss='ce', steps=30)

  elif attack == "auto":
      return torchattacks.AutoAttack(model=net, eps=eps, n_classes=n_classes)
  ```


### *Testing Adversarial Robustness*

* **Mearsuring the robustness in an adversarial trained model**
  - Run `test.py` 

  ```bash
    # model parameter
    parser.add_argument('--dataset', default='cifar10', type=str) # 'svhn', 'tiny'
    parser.add_argument('--network', default='vgg', type=str) # 'resnet', 'wide'
    parser.add_argument('--depth', default=16, type=int) # 18 (ResNet), 28 (WideResNet)
    parser.add_argument('--baseline', default='adv', type=str) # 'trades', 'mart', 'mad'
    parser.add_argument('--device', default='cuda:0', type=str)
    
    # mad parameter
    parser.add_argument('--percnt', default=0.9, type=float) # 0.99 (Sparsity)
    parser.add_argument('--pruning_mode', default='el', type=str) # 'rd' (random)
    parser.add_argument('--largest', default='false', type=str2bool) # 'true' (pruning high adversarial saliency)
    
    # attack parameter
    parser.add_argument('--attack', default='pgd', type=str)
    parser.add_argument('--eps', default=0.03, type=float)
    parser.add_argument('--steps', default=30, type=int)
  ```


---
