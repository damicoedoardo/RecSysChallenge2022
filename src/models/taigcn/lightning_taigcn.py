from logging.handlers import TimedRotatingFileHandler
from typing import Tuple

import pandas as pd
import pytorch_lightning as pl
import similaripy
import torch
from evaluation import compute_mrr
from sklearn.metrics.pairwise import cosine_similarity
from src.constant import *
from src.recommender_interface import RepresentationBasedRecommender
from utils.decorator import timing
from utils.pytorch.losses import bpr_loss
from utils.sparse_matrix import interactions_to_sparse_matrix


class WeightedSumSessEmbedding(pl.LightningModule):
    """Compute user embeddings from item embeddings"""

    def __init__(
        self,
        train_data: pd.DataFrame,
        dataset,
    ) -> None:
        pl.LightningModule.__init__(self)
        self.dataset = dataset
        self.train_data = train_data
        self.train_data_dict = train_data.groupby(SESS_ID)[ITEM_ID].apply(list)

    def forward(self, user_batch, embeddings):

        user_item_list = [
            torch.tensor(self.train_data_dict[x], dtype=torch.int64)
            for x in user_batch.cpu().numpy()
        ]
        # Move the user item list to the correct device
        # user_item_list = list(map(lambda x: x.to(self.device), user_item_list))
        user_embeddings = torch.stack(
            [
                torch.mean(torch.index_select(embeddings, 0, x), dim=0)
                for x in user_item_list
            ]
        )
        return user_embeddings


class LightTAIGCN(pl.LightningModule, RepresentationBasedRecommender):
    def __init__(
        self,
        dataset,
        sess_embedding_module: pl.LightningModule,
        convolution_depth: list,
        normalize_propagation: bool = False,
    ) -> None:
        # call init classes im extending
        RepresentationBasedRecommender.__init__(
            self,
            dataset=dataset,
        )
        pl.LightningModule.__init__(self)

        self.sess_embedding_module = sess_embedding_module
        self.dataset = dataset
        self.normalize_propagation = normalize_propagation
        # self.user_embedding_module = user_embedding_module

        self.convolution_depth = convolution_depth
        self.activation = torch.nn.LeakyReLU()

        propagation_m = self._compute_propagation_matrix()
        # self.register_buffer("S", propagation_m)
        # create features as torch tensor
        feature_tensor = torch.Tensor(dataset.get_oh_item_features().values)
        self.register_buffer("item_features", feature_tensor)

        # save the number of features available
        self.features_num = feature_tensor.shape[1]
        print(f"Num features available: {self.features_num}")

        # self.item_embeddings = torch.nn.Parameter(
        #     torch.empty(
        #         self.dataset._ITEMS_NUM,
        #         128,
        #     ),
        #     requires_grad=True,
        # )
        # torch.nn.init.xavier_normal_(self.item_embeddings)

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

    def _create_weights_matrices(self) -> torch.nn.ModuleDict:
        """Create linear transformation layers for graph convolution"""
        weights = dict()
        for i, embedding_size in enumerate(self.convolution_depth):
            if i == 0:
                weights["W_{}".format(i)] = torch.nn.Linear(
                    self.features_num, embedding_size, bias=False
                )
            else:
                weights["W_{}".format(i)] = torch.nn.Linear(
                    self.convolution_depth[i - 1], embedding_size, bias=False
                )
        return torch.nn.ModuleDict(weights)

    def forward(self, batch):
        item_embeddings = self.item_features
        for i, _ in enumerate(self.convolution_depth):
            linear_layer = self.weight_matrices[f"W_{i}"]
            # item_embeddings = linear_layer(prop_step)
            if i == len(self.convolution_depth) - 1:
                item_embeddings = self.activation(linear_layer(item_embeddings))
            else:
                item_embeddings = linear_layer(item_embeddings)
            # item_embeddings = self.S.matmul(item_embeddings)  # type: ignore

        s = batch[0]
        s_emb = self.sess_embedding_module(s, item_embeddings)
        return s_emb, item_embeddings

    def configure_optimizers(self):
        # initialize the optimizer
        optimizer = torch.optim.Adam(params=self.parameters(), lr=1e-3, weight_decay=0)
        return optimizer

    def training_step(self, batch):
        x_u, item_embeddings = self(batch)
        x_i = torch.index_select(item_embeddings, 0, batch[1])  # type: ignore
        x_j = torch.index_select(item_embeddings, 0, batch[2])  # type: ignore
        loss = bpr_loss(x_u, x_i, x_j)
        self.log("train_loss", loss)
        return loss

    @timing
    def validation_step(self, val: pd.DataFrame, val_label: pd.DataFrame):
        self.compute_representations(val)
        print("Recommend...")
        recs = self.recommend(
            interactions=val, remove_seen=True, cutoff=100, leaderboard=False
        )
        print("Computing mrr...")
        mrr = compute_mrr(recs, val_label)
        self.log("mrr", mrr)

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
