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
    name = "ItemKNN"

    def __init__(
        self,
        dataset,
        topk: int,
        shrink: int = 0,
        normalization: bool = False,
        time_weight: Union[float, None] = None,
        **kwargs
    ):
        super().__init__(dataset=dataset, time_weight=time_weight)
        self.topk = topk
        self.shrink = shrink
        self.normalization = normalization
        if "l" in kwargs:
            self.l = kwargs["l"]
        if "t1" in kwargs:
            self.t1 = kwargs["t1"]
        if "t2" in kwargs:
            self.t2 = kwargs["t2"]
        if "c" in kwargs:
            self.c = kwargs["c"]

    def compute_similarity_matrix(self, interaction_df: pd.DataFrame) -> None:
        sparse_interaction, user_mapping_dict, _ = interactions_to_sparse_matrix(
            interaction_df,
            items_num=self.dataset._ITEMS_NUM,
            users_num=None,
        )
        if self.normalization:
            sparse_interaction = similaripy.normalization.bm25(sparse_interaction)

        sim = similaripy.s_plus(
            sparse_interaction.T,
            k=self.topk,
            l=self.l,
            t1=self.t1,
            t2=self.t2,
            c=self.c,
            shrink=self.shrink,
        )
        self.similarity_matrix = sim
