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
from utils.pytorch.losses import CosineContrastiveLoss


class SparseDropout(torch.nn.Module):
    def __init__(self, dprob=0.5):
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
        dataset,
    ) -> None:
        nn.Module.__init__(self)
        self.dropout = SparseDropout(dprob=0.3)
        self.dataset = dataset

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
        # if self.training:
        #     user_batch = self.dropout(user_batch, num_ids, self.dataset._ITEMS_NUM)
        # user_embeddings = user_batch.matmul(embeddings)
        user_embeddings = torch.sparse.mm(user_batch, embeddings)
        # user_embeddings = F.normalize(user_embeddings)
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
        embedding_dimension: int,
        loss_function: nn.Module,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        # call init classes im extending
        RepresentationBasedRecommender.__init__(
            self,
            dataset=dataset,
        )
        nn.Module.__init__(self)

        self.embedding_dimension = embedding_dimension
        self.features_layer = features_layer
        self.sess_embedding_module = sess_embedding_module
        self.device = device
        self.dataset = dataset
        self.full_data_dict = (
            dataset.get_train_sessions().groupby(SESS_ID)[ITEM_ID].apply(list)
        )
        self.loss_function = loss_function

        # set activation function for item features
        self.activation = torch.nn.LeakyReLU()

        # initialise the item features buffer
        feature_tensor = torch.Tensor(dataset.get_oh_item_features().values)
        self.register_buffer("item_features", feature_tensor)

        # save the number of features available
        self.features_num = feature_tensor.shape[1]
        print(f"Num features available: {self.features_num}")

        # propagation_m = self._compute_propagation_matrix()
        # self.register_buffer("S", propagation_m)

        # one-hot embeddings for items
        self.item_embeddings = torch.nn.Parameter(
            torch.empty(
                self.dataset._ITEMS_NUM,
                self.embedding_dimension,
            ),
            requires_grad=True,
        )
        # nn.init.xavier_normal_(self.item_embeddings)
        nn.init.normal_(self.item_embeddings, std=1e-4)

        # self.alphas_weight = torch.nn.Parameter()

        self.weight_matrices = self._create_weights_matrices()

    def _compute_propagation_matrix(self) -> torch.Tensor:
        print("Computing propagation matrix")
        train_sessions = self.dataset.get_train_sessions()
        sp_int, _, _ = interactions_to_sparse_matrix(
            train_sessions,
            items_num=self.dataset._ITEMS_NUM,
            users_num=None,
        )
        user_degree = np.array(sp_int.sum(axis=1))
        d_user_inv = np.power(user_degree, -0.5).flatten()
        d_user_inv[np.isinf(d_user_inv)] = 0.0
        d_user_inv_diag = sps.diags(d_user_inv)

        item_degree = np.array(sp_int.sum(axis=0))
        d_item_inv = np.power(item_degree, -0.5).flatten()
        d_item_inv[np.isinf(d_item_inv)] = 0.0
        d_item_inv_diag = sps.diags(d_item_inv)

        int_norm = d_user_inv_diag.dot(sp_int).dot(d_item_inv_diag)
        L = int_norm.T @ int_norm

        crow_indices = L.indptr  # type: ignore
        col_indices = L.indices  # type: ignore
        values = L.data

        # use torch sparse csr tensor to store propagation matrix
        propagation_m = torch.sparse_csr_tensor(
            crow_indices, col_indices, values, dtype=torch.float32  # type: ignore
        ).to_sparse_coo()
        return propagation_m

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
        item_features = self.item_features

        # apply FFNN on features
        for i, _ in enumerate(self.features_layer):
            linear_layer = self.weight_matrices[f"W_{i}"]
            if i == len(self.features_layer) - 1:
                item_features = linear_layer(item_features)
            else:
                item_features = self.activation(linear_layer(item_features))

        # concat item embeddings and item features and then perform convolution
        final_item_embeddings = torch.cat((item_embeddings, item_features), 1)  # type: ignore
        # final_item_embeddings = torch.sparse.mm(self.S, item_embeddings)
        # final_item_embeddings = item_features

        s = (batch[0], batch[1], batch[2], batch[3])

        # final_item_embeddings = F.normalize(final_item_embeddings)

        s_emb = self.sess_embedding_module(s, final_item_embeddings)

        # s_emb = F.normalize(s_emb)
        # final_item_embeddings = F.normalize(final_item_embeddings)
        return s_emb, final_item_embeddings

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

        # apply l2-norm
        # sess_emb = F.normalize(sess_emb)
        # item_emb = F.normalize(item_emb)

        sess_emb = sess_emb.detach().cpu().numpy()
        item_emb = item_emb.detach().cpu().numpy()

        users_repr_df = pd.DataFrame(sess_emb, index=unique_sess_ids)
        items_repr_df = pd.DataFrame(item_emb)

        return users_repr_df, items_repr_df
