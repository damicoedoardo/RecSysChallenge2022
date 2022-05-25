from typing import Union

import numpy as np
import pandas as pd
import scipy.sparse as sps
import similaripy
from src.recommender_interface import ItemSimilarityRecommender
from src.utils.sparse_matrix import interactions_to_sparse_matrix


class CEASE(ItemSimilarityRecommender):

    name = "CEASE"

    def __init__(self, dataset, l2: float, time_weight: Union[float, None] = False):
        """EASE

        Note:
            paper: https://dl.acm.org/doi/abs/10.1145/3308558.3313710?casa_token=BtGI7FceWgYAAAAA:rz8xxtv4mlXjYIo6aWWlsAm9CP7zh-JZGGmN5UYUA4XwefaRfD6ZJ015GFkiMoBACF6GgKP9HEbMwQ

        Attributes:
            train_data (pd.DataFrame): dataframe containing user-item interactions
            l2 (float): l2 regularization
        """
        super().__init__(dataset=dataset, time_weight=time_weight)
        self.l2 = l2

    def compute_similarity_matrix(self, interaction_df: pd.DataFrame) -> None:
        content_matrix = self.dataset.get_oh_item_features()
        # content_matrix = content_matrix.drop(columns=[ITEM_ID])
        cols_to_drop = [c for c in content_matrix.columns if "cat" in c]
        content_matrix = content_matrix.drop(columns=cols_to_drop)
        # sparse_content_matrix = sps.csr_matrix(content_matrix.values)
        sparse_content_matrix = sps.csr_matrix(content_matrix.values, dtype=np.float32)

        # user_degree = np.array(sparse_content_matrix.sum(axis=1))
        # d_user_inv = np.power(user_degree, -0.5).flatten()
        # d_user_inv[np.isinf(d_user_inv)] = 0.0
        # d_user_diag = sps.diags(d_user_inv)
        # print(d_user_diag.shape)

        # d_user = np.power(user_degree, 0.5).flatten()
        # d_user[np.isinf(d_user)] = 0.0
        # d_user_diag = sps.diags(d_user)

        # item_degree = np.array(sparse_content_matrix.sum(axis=0))
        # d_item_inv = np.power(item_degree, -0.5).flatten()
        # d_item_inv[np.isinf(d_item_inv)] = 0.0
        # d_item_diag = sps.diags(d_item_inv)

        # norm_sparse_content_matrix = d_user_diag @ sparse_content_matrix @ d_item_diag
        # print(norm_sparse_content_matrix.shape)

        # Compute gram matrix on augmented
        G = (sparse_content_matrix @ sparse_content_matrix.T).toarray()
        print(G.shape)
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += self.l2 * self.dataset._ITEMS_NUM
        print("Computing inverse")
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diagIndices] = 0
        self.similarity_matrix = B
