from typing import Tuple, final

import numpy as np
import pandas as pd
import scipy.sparse as sps
import similaripy
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cassandra.session_embedding_modules import (
    GRUSessionEmbedding,
    MeanAggregatorSessionEmbedding,
)
from sklearn.metrics.pairwise import cosine_similarity
from src.constant import *
from src.recommender_interface import RepresentationBasedRecommender
from src.utils.pandas_utils import remap_column_consecutive
from src.utils.sparse_matrix import interactions_to_sparse_matrix
from torch.utils.data import Dataset as TorchDataset
from utils.pytorch.losses import CosineContrastiveLoss


class Cassandra(nn.Module, RepresentationBasedRecommender):
    name = "Cassandra"

    def __init__(
        self,
        dataset,
        session_embedding_module: torch.nn.Module,
        embedding_dimension: int,
        loss_function: nn.Module,
        item_features_embedding_module: nn.Module,
        train_dataset,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        # call init classes im extending
        RepresentationBasedRecommender.__init__(
            self,
            dataset=dataset,
        )
        nn.Module.__init__(self)

        self.embedding_dimension = embedding_dimension
        self.train_dataset = train_dataset
        self.session_embedding_module = session_embedding_module
        self.item_features_embedding_module = item_features_embedding_module

        self.device = device
        self.dataset = dataset
        self.full_data_dict = (
            dataset.get_train_sessions().groupby(SESS_ID)[ITEM_ID].apply(list)
        )
        self.loss_function = loss_function
        self.dropout_item = torch.nn.Dropout(p=0.5)

        # initialise the item features buffer
        feature_tensor = torch.Tensor(dataset.get_oh_item_features().values)
        # padding feature tensor
        print("Padding feature tensor...")
        padding_feature_tensor = torch.zeros(1, feature_tensor.shape[1])
        padded_feature_tensor = torch.cat([feature_tensor, padding_feature_tensor])
        self.register_buffer("item_features", padded_feature_tensor)

        # save the number of features available
        self.features_num = feature_tensor.shape[1]
        print(f"Num features available: {self.features_num}")

        # one-hot embeddings for items
        self.item_embeddings = torch.nn.Embedding(
            num_embeddings=self.dataset._ITEMS_NUM + 1,
            embedding_dim=self.embedding_dimension,
            padding_idx=self.dataset._ITEMS_NUM,
            # scale_grad_by_freq=True
            # max_norm=1,
        )
        nn.init.normal_(self.item_embeddings.weight, std=1e-4)
        # nn.init.xavier_normal_(self.item_embeddings.weight)
        # nn.init.normal_(self.item_embeddings, std=1e-4)

        self.linear = torch.nn.Linear(
            self.embedding_dimension * 2, self.embedding_dimension * 2
        )
        # self.mha = torch.nn.MultiheadAttention(
        #     embed_dim=self.embedding_dimension * 2, num_heads=1, batch_first=True
        # )

        # self.alphas_weight = torch.nn.Parameter()

    # def _create_weights_matrices(self) -> nn.ModuleDict:
    #     """Create linear transformation layers for oh features"""
    #     weights = dict()
    #     for i, layer_size in enumerate(self.features_layer):
    #         if i == 0:
    #             weights["W_{}".format(i)] = nn.Linear(
    #                 self.features_num, layer_size, bias=True
    #             )
    #         else:
    #             weights["W_{}".format(i)] = nn.Linear(
    #                 self.features_layer[i - 1], layer_size, bias=True
    #             )
    #     return nn.ModuleDict(weights)

    def forward(self, batch):
        sess2items = batch[0]

        # todo: check how guys of SimpleX do sampler part!
        item_embeddings = self.item_embeddings
        item_features = self.item_features

        # apply FFNN on features
        # for i, _ in enumerate(self.features_layer):
        #     linear_layer = self.weight_matrices[f"W_{i}"]
        #     if i == len(self.features_layer) - 1:
        #         item_features = linear_layer(item_features)
        #     else:
        #         item_features = self.activation(linear_layer(item_features))

        # concat item embeddings and item features and then perform convolution
        # final_item_embeddings = torch.cat((item_embeddings, item_features), 1)  # type: ignore
        # final_item_embeddings = torch.sparse.mm(self.S, item_embeddings)
        # final_item_embeddings = item_features

        # s = (batch[0], batch[1], batch[2], batch[3])

        # final_item_embeddings = F.normalize(final_item_embeddings)

        # sess2items_tensor -> [batch_size, seq_len, item_embedding_dimension]

        # item_features_embedding -> [items_num+1, features_layer[-1]]
        item_features_embedding = self.item_features_embedding_module(item_features)
        all_items_embedding = torch.cat(
            (item_embeddings.weight, item_features_embedding), -1
        )
        # all_items_embedding = self.linear(all_items_embedding)
        # all_items_embedding = F.normalize(all_items_embedding, dim=-1)
        # all_items_embedding = self.dropout_item(all_items_embedding)

        # sess2items_tensor = item_embeddings(sess2items)
        sess2items_tensor = all_items_embedding[sess2items]
        # item_features_tensor = item_features_embedding[sess2items]
        # concatenate the features embeddings and item embeddings
        # final_item_embeddings -> [batch_size, seq_len, item_embedding_dimension + features_embedding_dim]
        # final_item_embeddings = torch.cat((sess2items_tensor, item_features_tensor), -1)  # type: ignore
        # final_item_embeddings = F.normalize(final_item_embeddings, dim=-1)
        # final_item_embeddings = self.dropout_item(final_item_embeddings)
        # final_item_embeddings = torch.mean((sess2items_tensor, item_features_tensor), -1)  # type: ignore

        # final_item_embeddings, _ = self.mha(
        #     final_item_embeddings, final_item_embeddings, final_item_embeddings
        # )
        # final_item_embeddings = self.linear(final_item_embeddings)
        session_embedding = self.session_embedding_module(sess2items_tensor)

        # all_items_embedding = torch.cat(
        #     (item_embeddings.weight, item_features_embedding), -1
        # )
        # session_embedding = F.normalize(
        #     self.session_embedding_module(sess2items_tensor)
        # )
        # purchase_embedding = F.normalize(item_embeddings(purchase))
        # negative_embeddings = F.normalize(item_embeddings(negative_samples))

        # s_emb = F.normalize(s_emb)
        # final_item_embeddings = F.normalize(final_item_embeddings)
        # loss = self.loss_function(
        #     session_embedding, purchase_embedding, negative_embeddings
        # )
        # return session_embedding, purchase_embedding, negative_embeddings
        # session_embedding = F.normalize(session_embedding)
        # item_embeddings = F.normalize(item_embeddings)
        return session_embedding, all_items_embedding

    def compute_sessions_embeddings(self, interactions: pd.DataFrame):
        unique_sess_ids = interactions[SESS_ID].unique()
        sess2items = np.stack(
            self.train_dataset.padded_sess.loc[unique_sess_ids].values
        )
        sess2items_tensor = self.item_embeddings(
            torch.LongTensor(sess2items).to(self.device)
        )
        session_embedding = self.session_embedding_module(
            sess2items_tensor.to(self.device)
        )
        return session_embedding, unique_sess_ids

    def compute_representations(
        self, interactions: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # sess_embeddings, sess_indices = self.compute_sessions_embeddings(interactions)
        # # sess_embeddings, item_embeddings = model()
        # item_embeddings = self.item_embeddings.weight[:-1, :]

        ##########
        unique_sess_ids = interactions[SESS_ID].unique()
        sess2items = torch.LongTensor(
            np.stack(self.train_dataset.padded_sess.loc[unique_sess_ids].values)
        ).to(self.device)
        sess_embeddings, item_embeddings = self([sess2items])

        sess_embeddings = F.normalize(sess_embeddings)
        item_embeddings = F.normalize(item_embeddings)

        sess_embeddings = sess_embeddings.detach().cpu().numpy()
        item_embeddings = item_embeddings.detach().cpu().numpy()

        users_repr_df = pd.DataFrame(sess_embeddings, index=unique_sess_ids)
        items_repr_df = pd.DataFrame(item_embeddings)

        return users_repr_df, items_repr_df
