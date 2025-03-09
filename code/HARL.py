import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def HARL(model,
         x_natural,
         y,
         num_classes,
         step_size=0.003,
         epsilon=0.031,
         perturb_steps=10,
         alpha=0.1,
         beta=0.1,
         gamma=0.1,
         distance='l_inf'):
    """
    HARL: Hierarchical Adversarial Robustness Loss
    Combines adversarial example generation (PGD) and hierarchical equalization loss into a unified loss function.

    :param model: PyTorch model.
    :param x_natural: Clean input samples [batch_size, ...].
    :param y: Ground truth labels [batch_size].
    :param num_classes: Number of classes in the classification task.
    :param step_size: Step size for PGD attack.
    :param epsilon: Perturbation size for PGD attack.
    :param perturb_steps: Number of steps for PGD attack.
    :param alpha: Weight for balancing the losses across classes.
    :param beta: Weight for balancing the hierarchical equalization.
    :param gamma: Weight for adjusting the focus on rare classes.
    :param distance: Distance metric for the attack ('l_inf' supported).
    :return: Computed HARL value.
    """
    model.eval()
    
    # Step 1: Generate adversarial examples using PGD
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(x_natural.device).detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    
    # Step 2: Compute Hierarchical Equalization Loss
    outputs = model(x_adv)  # Model outputs for adversarial examples
    batch_size, num_classes_actual = outputs.size()
    assert num_classes_actual == num_classes, "Mismatch in number of classes"

    # Compute cross-entropy loss
    pixel_loss = F.cross_entropy(outputs, y, reduction='none')

    # Compute class-wise losses
    class_losses = torch.zeros(num_classes).to(outputs.device)
    for cls in range(num_classes):
        mask = (y == cls).float()
        class_loss = (pixel_loss * mask).sum() / (mask.sum() + 1e-10)
        class_losses[cls] = class_loss

    # Compute average and normalized class losses
    avg_class_loss = class_losses.mean()
    normalized_class_losses = class_losses / (class_losses.sum() + 1e-10)

    # Loss components
    balanced_loss = alpha * avg_class_loss
    hierarchical_loss = beta * ((class_losses - avg_class_loss) ** 2).mean()
    rare_class_loss = gamma * (normalized_class_losses ** 2).sum()

    # Total loss
    total_loss = balanced_loss + hierarchical_loss + rare_class_loss
    return total_loss
