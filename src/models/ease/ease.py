from typing import Union

import numpy as np
import pandas as pd
import scipy.sparse as sps
from src.recommender_interface import ItemSimilarityRecommender
from src.utils.sparse_matrix import (interactions_to_sparse_matrix,
                                     weighted_interactions_to_sparse_matrix)


class EASE(ItemSimilarityRecommender):

    name = "EASE"

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
        (
            sparse_interaction,
            user_mapping_dict,
            _,
        ) = weighted_interactions_to_sparse_matrix(
            interaction_df,
            items_num=self.dataset._ITEMS_NUM,
            users_num=None,
        )

        # Compute gram matrix
        G = (sparse_interaction.T @ sparse_interaction).toarray()
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += self.l2 * self.dataset._ITEMS_NUM
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diagIndices] = 0
        self.similarity_matrix = B
