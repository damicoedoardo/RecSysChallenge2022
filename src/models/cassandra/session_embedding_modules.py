import torch
import torch.nn as nn


class MeanAggregatorSessionEmbedding(nn.Module):
    def __init__(self) -> None:
        nn.Module.__init__(self)

    def forward(self, sess2items_tensor):
        # sess2items_tensor [batch_size, k, embedding_dimension]

        # shape batch_size number of non padded items each session has
        num_non_zero_item = (sess2items_tensor.sum(dim=-1) != 0).sum(dim=1)
        # sum items in the session shape: [batch_size, embedding_size]
        sum_session_items = sess2items_tensor.sum(dim=1)
        session_embedding = torch.div(sum_session_items, num_non_zero_item.view(-1, 1))
        # session_embedding = torch.mean(sess2items_tensor, dim=1)
        return session_embedding
