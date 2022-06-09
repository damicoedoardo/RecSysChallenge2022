import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SelfAttentionSessionEmbedding(nn.Module):
    def __init__(self, input_size: int, num_heads: int) -> None:
        nn.Module.__init__(self)
        self.activation = torch.nn.LeakyReLU()
        self.qw = nn.Linear(input_size, input_size)
        self.kw = nn.Linear(input_size, input_size)
        self.vw = nn.Linear(input_size, input_size)
        # we use the input as value since is free
        self.mha = nn.MultiheadAttention(
            embed_dim=input_size, num_heads=num_heads, batch_first=True
        )
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=input_size,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, sess2items_tensor):
        # sess2items_tensor [batch_size, k, embedding_dimension]
        query = self.activation(self.qw(sess2items_tensor))
        key = self.activation(self.kw(sess2items_tensor))
        # value = self.activation(self.vw(sess2items_tensor))
        # [batch_size, k, embedding_dimension]
        attn_out, _ = self.mha(query, key, sess2items_tensor)
        # session_repr = torch.mean(attn_out, dim=1)
        _, hidden_states = self.gru(attn_out)
        # final hidden state is of size [num_layers, batch_size, hidden_size] need squeeze
        # return session_repr
        return torch.squeeze(hidden_states[-1, :, :])


class ContextAttention(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1) -> None:
        nn.Module.__init__(self)
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, sess2items_tensor):
        # sess2items_tensor -> [batch_size, k, embedding_dimension]
        output, (hn, cn) = self.lstm(sess2items_tensor)
        context_query = torch.squeeze(cn[-1, :, :])
        # context_query -> [batch_size, hidden_size]
        # output -> [batch_size, k, hidden_size]

        # normalise them to have cosine similarity
        output = F.normalize(output, dim=-1)
        sess2items_tensor = F.normalize(sess2items_tensor, dim=-1)
        context_query = F.normalize(context_query, dim=-1)

        # use torch einsum for the fot product here
        cos_sim = torch.einsum("bf,bkf->bk", context_query, output)

        # apply softmax function to obtain attention coefficients
        # attn_coeff -> [batch_size, k]
        attn_coeff = self.softmax(cos_sim)

        # compute weighted sum using the attention coefficients
        # session embeddings -> [batch_size, embedding_dimension]
        session_embedding = torch.einsum("bk,bkf->bf", attn_coeff, sess2items_tensor)

        return session_embedding
