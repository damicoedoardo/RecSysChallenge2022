from typing import Tuple

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
        session_embedding_kind: str,
        features_layer: list[int],
        embedding_dimension: int,
        loss_function: nn.Module,
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
        self.features_layer = features_layer
        self.train_dataset = train_dataset
        self.session_embedding_kind = session_embedding_kind
        self.session_embedding_module = self._initialize_session_embedding_module(
            session_embedding_kind
        )

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

        # self.alphas_weight = torch.nn.Parameter()

        # self.weight_matrices = self._create_weights_matrices()

    def _initialize_session_embedding_module(
        self, session_embedding_kind: str
    ) -> nn.Module:
        if session_embedding_kind == "mean":
            session_embedding_module = MeanAggregatorSessionEmbedding()
            return session_embedding_module
        elif session_embedding_kind == "gru":
            session_embedding_module = GRUSessionEmbedding()
            return session_embedding_module
        else:
            raise NotImplementedError(
                f"Aggregator {session_embedding_kind} not implemented!"
            )

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
        # unpack batch
        sess2items, _, _ = batch[0], batch[1], batch[2]

        # todo: check how guys of SimpleX do sampler part!
        item_embeddings = self.item_embeddings
        # item_features = self.item_features

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
        sess2items_tensor = item_embeddings(sess2items)
        session_embedding = self.session_embedding_module(sess2items_tensor)

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
        return session_embedding, item_embeddings

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
        sess_embeddings, sess_indices = self.compute_sessions_embeddings(interactions)
        # sess_embeddings, item_embeddings = model()
        item_embeddings = self.item_embeddings.weight[:-1, :]

        sess_embeddings = F.normalize(sess_embeddings)
        item_embeddings = F.normalize(item_embeddings)

        sess_embeddings = sess_embeddings.detach().cpu().numpy()
        item_embeddings = item_embeddings.detach().cpu().numpy()

        users_repr_df = pd.DataFrame(sess_embeddings, index=sess_indices)
        items_repr_df = pd.DataFrame(item_embeddings)

        return users_repr_df, items_repr_df
