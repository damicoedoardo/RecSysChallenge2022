import torch
import torch.nn.functional as F


def bpr_loss(x_u: torch.Tensor, x_i: torch.Tensor, x_j: torch.Tensor):
    # compute dot products
    x_ui = torch.einsum("nf,nf->n", x_u, x_i)
    x_uj = torch.einsum("nf,nf->n", x_u, x_j)
    x_uij = x_ui - x_uj
    loss = -F.logsigmoid(x_uij).sum()
    return loss


class CosineContrastiveLossOriginal(torch.nn.Module):
    def __init__(self, margin=0, negative_weight=None):
        """
        :param margin: float, margin in CosineContrastiveLoss
        :param num_negs: int, number of negative samples
        :param negative_weight:, float, the weight set to the negative samples. When negative_weight=None, it
            equals to num_negs
        """
        super(CosineContrastiveLoss, self).__init__()
        self._margin = margin
        self._negative_weight = negative_weight

    def forward(self, y_pred, y_true):
        """
        :param y_pred: prdicted values of shape (batch_size, 1 + num_negs)
        :param y_true: true labels of shape (batch_size, 1 + num_negs)
        """
        pos_logits = y_pred[:, 0]
        pos_loss = torch.relu(1 - pos_logits)
        neg_logits = y_pred[:, 1:]
        neg_loss = torch.relu(neg_logits - self._margin)
        if self._negative_weight:
            loss = pos_loss + neg_loss.mean(dim=-1) * self._negative_weight
        else:
            loss = pos_loss + neg_loss.sum(dim=-1)
        return loss.mean()


class CosineContrastiveLoss(torch.nn.Module):
    def __init__(self, margin: float = 0, negative_weight: float = 0.5):
        """Cosine contrastive loss
        Args:
            margin (int, optional): margin. Defaults to 0.
            negative_weight (_type_, optional): negative weight. Defaults to None.
        """
        super(CosineContrastiveLoss, self).__init__()
        self._margin = margin
        self._negative_weight = negative_weight

    def forward(self, x_u: torch.Tensor, x_i: torch.Tensor, x_j: torch.Tensor):
        """_summary_

        Args:
            x_u (torch.Tensor): user embedding tensor shape [batch_size, embedding_size]
            x_i (torch.Tensor): positive item embedding tensor shape [batch_size, embedding_size]
            x_j (torch.Tensor): negative item embedding tensor shape [batch_size, num_neg_samples, embedding_size]

        Returns:
            _type_: _description_
        """
        # compute positve part of loss function
        x_ui = torch.einsum("bf,bf->b", x_u, x_i)
        # print(x_ui.mean())

        # the relu here should be not necessary...
        pos_loss = torch.relu(1 - x_ui)
        # pos_loss = torch.relu(-x_ui)

        # pos_loss = x_ui
        # compute negative part of the loss function
        x_uj = torch.einsum("bf,bnf->bn", x_u, x_j)
        # print(x_ui.mean(dim=-1))
        # print(x_uj - self._margin)
        # check this scores....
        neg_loss = torch.relu(x_uj - self._margin)
        # neg_loss_sample_mean = neg_loss.mean(dim=-1)

        denominator = (neg_loss != 0).sum(dim=-1)
        numerator = neg_loss.sum(dim=-1)
        neg_loss_sample_mean = numerator / (denominator + 1e-12)

        # print(pos_loss.mean())
        # print(neg_loss.mean())
        loss = (
            1 - self._negative_weight
        ) * pos_loss + self._negative_weight * neg_loss_sample_mean
        # loss = pos_loss + neg_loss_sample_mean
        return loss.mean()
