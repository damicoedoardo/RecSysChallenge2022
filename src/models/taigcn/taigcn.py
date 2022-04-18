from pickle import NONE
from turtle import forward
from typing import Tuple

import numpy as np
import pandas as pd
import similaripy
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.pyplot import axis
from sklearn import datasets
from sklearn.metrics.pairwise import cosine_similarity
from src.constant import *
from src.recommender_interface import RepresentationBasedRecommender
from src.utils.pandas_utils import remap_column_consecutive
from utils.sparse_matrix import interactions_to_sparse_matrix


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

        user_item_list = [
            torch.tensor(self.train_data_dict[x], dtype=torch.int64)
            for x in user_batch.cpu().numpy()
        ]
        # Move the user item list to the correct device
        user_item_list = list(map(lambda x: x.to(self.device), user_item_list))
        user_embeddings = torch.stack(
            [
                torch.mean(torch.index_select(embeddings, 0, x), dim=0)
                for x in user_item_list
            ]
        )
        return user_embeddings


class TimeAttentionSessEmbedding(nn.Module):
    pass


class TAIGCN(nn.Module, RepresentationBasedRecommender):
    name = "TAIGCN"

    def __init__(
        self,
        dataset,
        sess_embedding_module: nn.Module,
        convolution_depth: list,
        normalize_propagation: bool = False,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        # call init classes im extending
        RepresentationBasedRecommender.__init__(
            self,
            dataset=dataset,
        )
        nn.Module.__init__(self)

        self.sess_embedding_module = sess_embedding_module
        self.device = device
        self.dataset = dataset
        self.normalize_propagation = normalize_propagation
        # self.user_embedding_module = user_embedding_module

        self.convolution_depth = convolution_depth

        propagation_m = self._compute_propagation_matrix()
        self.register_buffer("S", propagation_m)
        # create features as torch tensor
        feature_tensor = torch.Tensor(dataset.get_oh_item_features().values)
        self.register_buffer("item_features", feature_tensor)

        # save the number of features available
        self.features_num = feature_tensor.shape[1]
        print(f"Num features available: {self.features_num}")

        self.weight_matrices = self._create_weights_matrices()

    def _compute_propagation_matrix(self) -> torch.Tensor:
        """Compute the propagation matrix for the item-item graph"""
        print("Computing propagation matrix")
        train_sessions = self.dataset.get_train_sessions()
        sparse_interaction, _, _ = interactions_to_sparse_matrix(
            train_sessions,
            items_num=self.dataset._ITEMS_NUM,
            users_num=None,
        )
        similarity = cosine_similarity(sparse_interaction.T, dense_output=False)

        if self.normalize_propagation:
            similarity = similaripy.normalization.normalize(
                similarity, norm="l1", axis=0
            )

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
        for i, embedding_size in enumerate(self.convolution_depth):
            if i == 0:
                weights["W_{}".format(i)] = nn.Linear(
                    self.features_num, embedding_size, bias=False
                )
            else:
                weights["W_{}".format(i)] = nn.Linear(
                    self.convolution_depth[i - 1], embedding_size, bias=False
                )
        return nn.ModuleDict(weights)

    def forward(self, batch):
        item_embeddings = self.item_features
        for i, _ in enumerate(self.convolution_depth):
            prop_step = self.S.matmul(item_embeddings)  # type: ignore
            linear_layer = self.weight_matrices[f"W_{i}"]
            item_embeddings = linear_layer(prop_step)

        s = batch[0]
        s_emb = self.sess_embedding_module(s, item_embeddings)
        return s_emb, item_embeddings

    def compute_representations(
        self, interactions: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        unique_sess_ids = interactions[SESS_ID].unique()
        sess_ids = torch.unsqueeze(
            torch.tensor(unique_sess_ids, dtype=torch.int64), dim=0
        )
        sess_emb, item_emb = self(sess_ids)

        sess_emb = sess_emb.detach().cpu().numpy()
        item_emb = item_emb.detach().cpu().numpy()

        users_repr_df = pd.DataFrame(sess_emb, index=unique_sess_ids)
        items_repr_df = pd.DataFrame(item_emb)

        return users_repr_df, items_repr_df
