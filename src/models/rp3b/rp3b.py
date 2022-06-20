from calendar import c
from re import L
from typing import Union

import numpy as np
import pandas as pd
import scipy.sparse as sps
import similaripy
from sklearn.metrics.pairwise import cosine_similarity
from src.recommender_interface import ItemSimilarityRecommender
from src.utils.sparse_matrix import (interactions_to_sparse_matrix,
                                     truncate_top_k,
                                     weighted_interactions_to_sparse_matrix)


class Rp3b(ItemSimilarityRecommender):
    name = "Rp3b"

    def __init__(
        self,
        dataset,
        topk: int,
        alpha: float = 0.5,
        beta: float = 0.5,
        time_weight: Union[float, None] = None,
    ):
        super().__init__(dataset=dataset, time_weight=time_weight)
        self.topk = topk
        self.alpha = alpha
        self.beta = beta

    def compute_similarity_matrix(self, interaction_df: pd.DataFrame) -> None:
        sparse_interaction, user_mapping_dict, _ = interactions_to_sparse_matrix(
            interaction_df,
            items_num=self.dataset._ITEMS_NUM,
            users_num=None,
        )
        # (
        #     sparse_interaction,
        #     user_mapping_dict,
        #     _,
        # ) = weighted_interactions_to_sparse_matrix(
        #     interaction_df,
        #     items_num=self.dataset._ITEMS_NUM,
        #     users_num=None,
        # )

        sim = similaripy.rp3beta(
            sparse_interaction.T,
            k=self.topk,
            alpha=self.alpha,
            beta=self.beta,
        )

        self.similarity_matrix = sim
