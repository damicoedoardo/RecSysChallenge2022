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


class WeightedSumSessEmbedding(nn.Module):
    """Compute user embeddings from item embeddings"""

    def __init__(
        self,
        train_data: pd.DataFrame,
        dataset,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        nn.Module.__init__(self)
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
        user_embeddings = user_batch.matmul(embeddings)
        return user_embeddings


class TimeAttentionSessEmbedding(nn.Module):
    pass


class TAIGCN(nn.Module, RepresentationBasedRecommender):
    name = "TAIGCN"

    def __init__(
        self,
        dataset,
        sess_embedding_module: nn.Module,
        convolution_depth: int,
        features_layer: list,
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

        self.convolution_depth = convolution_depth
        self.activation = torch.nn.LeakyReLU()

        propagation_m = self._compute_propagation_matrix()
        self.register_buffer("S", propagation_m)
        # create features as torch tensor
        feature_tensor = torch.Tensor(dataset.get_oh_item_features().values)
        self.register_buffer("item_features", feature_tensor)

        # save the number of features available
        self.features_num = feature_tensor.shape[1]
        print(f"Num features available: {self.features_num}")

        self.item_embeddings = torch.nn.Parameter(
            torch.empty(
                self.dataset._ITEMS_NUM,
                256,
            ),
            requires_grad=True,
        )
        nn.init.xavier_normal_(self.item_embeddings)

        self.weight_matrices = self._create_weights_matrices()

    def _compute_propagation_matrix(self) -> torch.Tensor:
        """Compute the propagation matrix for the item-item graph"""
        print("Computing propagation matrix")
        train_sessions = self.dataset.get_train_sessions()

        split_dict = self.dataset.get_split()
        train, train_label = split_dict[TRAIN]
        full_train = pd.concat([train_sessions, train_label], axis=0)

        # full_train = train_sessions

        sparse_interaction, _, _ = interactions_to_sparse_matrix(
            full_train,
            items_num=self.dataset._ITEMS_NUM,
            users_num=None,
        )
        # similarity = cosine_similarity(sparse_interaction.T, dense_output=False)
        similarity = similaripy.cosine(sparse_interaction.T, k=20, format_output="csr")

        if self.normalize_propagation:
            similarity = similaripy.normalization.normalize(
                similarity, norm="l1", axis=0
            )

        # set similarity to 1
        # similarity.data = np.ones(len(similarity.data))

        degree = np.array(similarity.sum(axis=0)).squeeze()
        D = sps.diags(degree, format="csr")
        # D = D.power(-1 / 2)
        D = D.power(-1 / 2)
        similarity = D * similarity * D

        crow_indices = similarity.indptr  # type: ignore
        col_indices = similarity.indices  # type: ignore
        values = similarity.data

        # use torch sparse csr tensor to store propagation matrix
        propagation_m = torch.sparse_csr_tensor(
            crow_indices, col_indices, values, dtype=torch.float32  # type: ignore
        ).to_sparse_coo()

        print("Computing propagation matrix")
        return propagation_m

    def _create_weights_matrices(self) -> nn.ModuleDict:
        """Create linear transformation layers for graph convolution"""
        weights = dict()
        for i, embedding_size in enumerate(self.features_layer):
            if i == 0:
                weights["W_{}".format(i)] = nn.Linear(
                    self.features_num, embedding_size, bias=True
                )
            else:
                weights["W_{}".format(i)] = nn.Linear(
                    self.features_layer[i - 1], embedding_size, bias=True
                )
        return nn.ModuleDict(weights)

    def forward(self, batch):
        # item_embeddings = self.item_embeddings
        item_features = self.item_features

        # final_embeddings = torch.cat((item_embeddings, item_features), 1)  # type: ignore
        # apply FFNN on features
        for i, _ in enumerate(self.features_layer):
            linear_layer = self.weight_matrices[f"W_{i}"]
            if i == len(self.features_layer) - 1:
                item_features = linear_layer(item_features)
                item_features = self.S.matmul(item_features)  # type: ignore
            else:
                item_features = self.activation(linear_layer(item_features))
                item_features = self.S.matmul(item_features)  # type: ignore
                # item_features = linear_layer(item_features)

        # concat item embeddings and item features and then perform convolution
        # final_embeddings = torch.cat((item_embeddings, item_features), 1)  # type: ignore

        final_embeddings = item_features
        # final_embeddings = item_embeddings

        for i in range(self.convolution_depth):
            final_embeddings = self.S.matmul(final_embeddings)  # type: ignore

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
