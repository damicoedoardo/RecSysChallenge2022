import torch
import torch.nn.functional as F


def bpr_loss(x_u: torch.Tensor, x_i: torch.Tensor, x_j: torch.Tensor):
    # compute dot products
    x_ui = torch.einsum("nf,nf->n", x_u, x_i)
    x_uj = torch.einsum("nf,nf->n", x_u, x_j)
    x_uij = x_ui - x_uj
    loss = -F.logsigmoid(x_uij).sum()
    return loss
