import numpy as np
import pandas as pd
import scipy.sparse as sps
import similaripy
from sklearn.metrics.pairwise import cosine_similarity
from src.recommender_interface import ItemSimilarityRecommender
from src.utils.sparse_matrix import interactions_to_sparse_matrix, truncate_top_k


class ItemKNN(ItemSimilarityRecommender):
    name = "ItemKNN"

    def __init__(self, dataset, topk: int, time_weight: bool = False):
        super().__init__(dataset=dataset, time_weight=time_weight)
        self.topk = topk

    def compute_similarity_matrix(self, interaction_df: pd.DataFrame) -> None:
        sparse_interaction, user_mapping_dict, _ = interactions_to_sparse_matrix(
            interaction_df,
            items_num=self.dataset._ITEMS_NUM,
            users_num=None,
        )
        # sp_int = similaripy.normalization.bm25(sparse_interaction)

        # sim = similaripy.rp3beta(sparse_interaction.T, k=2000, alpha=0.7, beta=0.3)
        sim = similaripy.rp3beta(sparse_interaction.T, k=2000, alpha=0.7, beta=0.3)

        # sim = similaripy.jaccard(sparse_interaction.T, k=2000)

        # sim = cosine_similarity(
        #     sparse_interaction.T, sparse_interaction.T, dense_output=False
        # )
        # setting diag to 0 preventing considering in topk similarity self-similarities
        # np.fill_diagonal(sim, 0)
        # sim = truncate_top_k(sim, self.topk)
        # sim = sps.csr_matrix(sim)
        self.similarity_matrix = sim
