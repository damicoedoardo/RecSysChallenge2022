from unicodedata import bidirectional

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


class GRUSessionEmbedding(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        nn.Module.__init__(self)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, sess2items_tensor):
        # sess2items_tensor [batch_size, k, embedding_dimension]
        _, hidden_states = self.gru(sess2items_tensor)
        # final hidden state is of size [num_layers, batch_size, hidden_size] need squeeze
        return torch.squeeze(hidden_states[-1, :, :])
