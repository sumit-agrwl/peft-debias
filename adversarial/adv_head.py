import torch
from torch import nn
from torch.nn import functional as F

from torch.autograd import Function
from torch.nn import Module

def get_rev_grad_func(grad_rev_strength):
    class _RevGradFunc(Function):
        @staticmethod
        def forward(ctx, input_):
            ctx.save_for_backward(input_)
            output = input_
            return output

        @staticmethod
        def backward(ctx, grad_output):  # pragma: no cover
            grad_input = None
            if ctx.needs_input_grad[0]:
                grad_input = - grad_rev_strength * grad_output
            return grad_input
    revgrad = _RevGradFunc.apply
    return revgrad

class RevGrad(Module):
    def __init__(self, grad_rev_strength, *args, **kwargs):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """

        super().__init__(*args, **kwargs)
        self.grad_rev_strength = grad_rev_strength
        self.rev_grad_func = get_rev_grad_func(grad_rev_strength)

    def forward(self, input_):
        output = self.rev_grad_func(input_)
        return output

class AdversarialClassifierHead(nn.Module):
    def __init__(self, feat_dim, attr_dim, adv_grad_rev_strength, adv_dropout=None, hidden_layer_num=1):
        super().__init__()
        input_dim = feat_dim

        mlp = []
        for i in range(hidden_layer_num):
            if adv_dropout is not None:
                mlp.append(nn.Dropout(adv_dropout))
            mlp.append(nn.Linear(input_dim, input_dim))
            mlp.append(nn.Tanh())

        mlp.append(nn.Linear(input_dim, attr_dim))
        self.mlp = nn.Sequential(*mlp)
        self.adv_grad_rev_strength = adv_grad_rev_strength
        self.rev_grad = RevGrad(self.adv_grad_rev_strength)

    def forward(self, hidden, rev_grad):
        if rev_grad:
            hidden = self.rev_grad(hidden)
        
        pred = self.mlp(hidden)
        return pred

    def compute_loss(self, *args, **kwargs):
        return self.compute_loss_ce_equal_odds(*args, **kwargs)


    def compute_loss_ce_equal_odds(self, attr_pred, attr_gt, cls_weights=None):
        if cls_weights is not None:
            adv_loss = F.cross_entropy(attr_pred, attr_gt, weight=cls_weights, reduction='none', ignore_index=-1)
        else:
            adv_loss = F.cross_entropy(attr_pred, attr_gt, reduction='none', ignore_index=-1)
        return adv_loss.mean()


