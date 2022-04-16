from calendar import c
from re import L
from typing import Union

import numpy as np
import pandas as pd
import scipy.sparse as sps
import similaripy
from matplotlib.pyplot import axis
from sklearn.metrics.pairwise import cosine_similarity
from src.constant import ITEM_ID
from src.recommender_interface import ItemSimilarityRecommender
from src.utils.sparse_matrix import interactions_to_sparse_matrix, truncate_top_k
from torch import topk


class CBItemKNN(ItemSimilarityRecommender):
    name = "CB-ItemKNN"

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
        # load content matrix
        content_matrix = self.dataset.get_oh_item_features()
        content_matrix = content_matrix.drop(columns=[ITEM_ID])

        cols_to_drop = [c for c in content_matrix.columns if "cat" in c]
        content_matrix = content_matrix.drop(columns=cols_to_drop)
        # sparse_content_matrix = sps.csr_matrix(content_matrix.values)
        sparse_content_matrix = sps.csr_matrix(content_matrix.values, dtype=np.float32)

        if self.normalization:
            sparse_content_matrix = similaripy.normalization.bm25(sparse_content_matrix)

        # sim = similaripy.s_plus(
        #     sparse_content_matrix,
        #     k=self.topk,
        #     l=self.l,
        #     t1=self.t1,
        #     t2=self.t2,
        #     c=self.c,
        #     shrink=self.shrink,
        # )

        sim = similaripy.cosine(
            sparse_content_matrix,
            k=self.topk,
            shrink=self.shrink,
        )

        self.similarity_matrix = sim
