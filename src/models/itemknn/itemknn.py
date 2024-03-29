from calendar import c
from re import L
from typing import Union

import numpy as np
import pandas as pd
import scipy.sparse as sps
import similaripy
from sklearn.metrics.pairwise import cosine_similarity
from src.recommender_interface import ItemSimilarityRecommender
from src.utils.sparse_matrix import interactions_to_sparse_matrix, truncate_top_k


class ItemKNN(ItemSimilarityRecommender):
    name = "item_knn"

    def __init__(
        self,
        dataset,
        topk: int,
        shrink: int = 0,
        time_weight: Union[float, None] = None,
    ):
        super().__init__(dataset=dataset, time_weight=time_weight)
        self.topk = topk
        self.shrink = shrink

    def compute_similarity_matrix(self, interaction_df: pd.DataFrame) -> None:
        sparse_interaction, user_mapping_dict, _ = interactions_to_sparse_matrix(
            interaction_df,
            items_num=self.dataset._ITEMS_NUM,
            users_num=None,
        )

        sim = similaripy.cosine(
            sparse_interaction.T,
            k=self.topk,
            shrink=self.shrink,
        )

        self.similarity_matrix = sim
