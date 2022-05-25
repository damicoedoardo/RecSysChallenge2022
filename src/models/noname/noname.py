from typing import Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sps
import similaripy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from src.constant import *
from src.recommender_interface import RepresentationBasedRecommender
from src.utils.pandas_utils import remap_column_consecutive
from src.utils.sparse_matrix import interactions_to_sparse_matrix


class SparseDropout(torch.nn.Module):
    def __init__(self, dprob=0.2):
        super(SparseDropout, self).__init__()
        # dprob is ratio of dropout
        # convert to keep probability
        self.kprob = 1 - dprob

    def forward(self, x, num_ids, num_items):
        mask = ((torch.rand(x._values().size()) + (self.kprob)).floor()).type(
            torch.bool
        )
        rc = x._indices()[:, mask]
        val = x._values()[mask] * (1.0 / self.kprob)
        return torch.sparse.FloatTensor(
            rc,
            val,
            [num_ids, num_items],
        )


class WeightedSumSessEmbedding(nn.Module):
    """Compute user embeddings from item embeddings"""

    def __init__(
        self,
        train_data: pd.DataFrame,
        dataset,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        nn.Module.__init__(self)
        self.dropout = SparseDropout()
        self.dataset = dataset
        self.train_data = train_data
        self.train_data_dict = train_data.groupby(SESS_ID)[ITEM_ID].apply(list)
        self.device = device

    def forward(self, user_batch, embeddings):

        # user_item_list = [
        #     torch.tensor(self.train_data_dict[x], dtype=torch.int64).to(self.device)
        #     for x in user_batch.cpu().numpy()
        # ]
        # Move the user item list to the correct device
        # user_item_list = list(map(lambda x: x.to(self.device), user_item_list))
        # user_embeddings = torch.stack(
        #     [
        #         torch.mean(torch.index_select(embeddings, 0, x), dim=0)
        #         for x in user_batch
        #     ]
        # )
        row_idx, col_idx, data_tensor, num_ids = user_batch
        user_batch = torch.sparse_coo_tensor(
            torch.stack([row_idx, col_idx]),
            data_tensor,
            [num_ids, self.dataset._ITEMS_NUM],
        )
        user_batch = self.dropout(user_batch, num_ids, self.dataset._ITEMS_NUM)
        user_embeddings = user_batch.matmul(embeddings)
        return user_embeddings


class TimeAttentionSessEmbedding(nn.Module):
    pass


class NoName(nn.Module, RepresentationBasedRecommender):
    name = "NoName"

    def __init__(
        self,
        dataset,
        sess_embedding_module: nn.Module,
        features_layer: list[int],
        normalize_propagation: bool = False,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        # call init classes im extending
        RepresentationBasedRecommender.__init__(
            self,
            dataset=dataset,
        )
        nn.Module.__init__(self)

        self.features_layer = features_layer
        self.sess_embedding_module = sess_embedding_module
        self.device = device
        self.dataset = dataset
        self.normalize_propagation = normalize_propagation
        self.full_data_dict = (
            dataset.get_train_sessions().groupby(SESS_ID)[ITEM_ID].apply(list)
        )
        # self.user_embedding_module = user_embedding_module

        # set activation function for item features
        self.activation = torch.nn.LeakyReLU()

        # initialise the item features buffer
        feature_tensor = torch.Tensor(dataset.get_oh_item_features().values)
        self.register_buffer("item_features", feature_tensor)

        # save the number of features available
        self.features_num = feature_tensor.shape[1]
        print(f"Num features available: {self.features_num}")

        # one-hot embeddings for items
        self.item_embeddings = torch.nn.Embedding(
            num_embeddings=self.dataset._ITEMS_NUM,
            embedding_dim=1024,
        )
        # Xavier initialisation of item embeddings
        nn.init.xavier_normal_(self.item_embeddings.weight)

        self.weight_matrices = self._create_weights_matrices()

    def _create_weights_matrices(self) -> nn.ModuleDict:
        """Create linear transformation layers for oh features"""
        weights = dict()
        for i, layer_size in enumerate(self.features_layer):
            if i == 0:
                weights["W_{}".format(i)] = nn.Linear(
                    self.features_num, layer_size, bias=True
                )
            else:
                weights["W_{}".format(i)] = nn.Linear(
                    self.features_layer[i - 1], layer_size, bias=True
                )
        return nn.ModuleDict(weights)

    def forward(self, batch):
        # todo: check how guys of SimpleX do sampler part!
        item_embeddings = self.item_embeddings
        # item_features = self.item_features

        # final_embeddings = torch.cat((item_embeddings, item_features), 1)  # type: ignore
        # apply FFNN on features
        # for i, _ in enumerate(self.features_layer):
        #     linear_layer = self.weight_matrices[f"W_{i}"]
        #     if i == len(self.features_layer) - 1:
        #         item_features = linear_layer(item_features)
        #         item_features = self.S.matmul(item_features)  # type: ignore
        #     else:
        #         item_features = self.activation(linear_layer(item_features))
        #         item_features = self.S.matmul(item_features)  # type: ignore
        # item_features = linear_layer(item_features)

        # concat item embeddings and item features and then perform convolution
        # final_embeddings = torch.cat((item_embeddings, item_features), 1)  # type: ignore

        # final_embeddings = item_features
        final_embeddings = item_embeddings

        # for i in range(self.convolution_depth):
        #     final_embeddings = self.S.matmul(final_embeddings)  # type: ignore

        s = (batch[0], batch[1], batch[2], batch[3])
        s_emb = self.sess_embedding_module(s, final_embeddings)
        # s_emb = None
        return s_emb, final_embeddings

    def compute_representations(
        self, interactions: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        unique_sess_ids = interactions[SESS_ID].unique()

        col_idx = torch.cat(
            [
                torch.tensor(self.full_data_dict[u], dtype=torch.int64)
                for u in unique_sess_ids
            ]
        ).to(self.device)
        row_idx = torch.cat(
            [
                torch.tensor(
                    np.repeat(idx, len(self.full_data_dict[u])), dtype=torch.int64
                )
                for u, idx in zip(unique_sess_ids, range(len(unique_sess_ids)))
            ]
        ).to(self.device)
        data_tensor = torch.cat(
            [
                torch.tensor(
                    np.repeat(
                        1 / len(self.full_data_dict[u]), len(self.full_data_dict[u])
                    ),
                    dtype=torch.float32,
                )
                for u in unique_sess_ids
            ]
        ).to(self.device)

        # sess_ids = torch.unsqueeze(
        #     torch.tensor(unique_sess_ids, dtype=torch.int64), dim=0
        # )
        sess_emb, item_emb = self(
            [
                row_idx,
                col_idx,
                data_tensor,
                torch.tensor(len(unique_sess_ids)).to(self.device),
            ],
        )

        sess_emb = sess_emb.detach().cpu().numpy()
        item_emb = item_emb.detach().cpu().numpy()

        users_repr_df = pd.DataFrame(sess_emb, index=unique_sess_ids)
        items_repr_df = pd.DataFrame(item_emb)

        return users_repr_df, items_repr_df
