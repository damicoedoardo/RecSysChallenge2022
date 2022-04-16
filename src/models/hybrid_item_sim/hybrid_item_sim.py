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


class HybridItemSimilarity(ItemSimilarityRecommender):
    name = "Hybrid_ItemSimilarity"

    def __init__(
        self,
        dataset,
        model_list: list,
        model_weight_list: list,
        normalization: Union[str, None] = "l1",
        normalization_axis: int = 0,
        time_weight: Union[float, None] = None,
    ):
        super().__init__(dataset=dataset, time_weight=time_weight)
        norm = ["l1", "l2", "max", None]
        assert (
            normalization in norm
        ), f"normalization: {normalization}, should be in {norm}"
        assert normalization_axis in [0, 1], "Normalization axis should be 0 or 1!"
        assert len(model_weight_list) == len(
            model_list
        ), "weights and model list have no the same length"
        self.model_list = model_list

        # normalise weights
        norm_model_weight_list = [
            mw / sum(model_weight_list) for mw in model_weight_list
        ]
        print(norm_model_weight_list)
        self.model_weight_list = norm_model_weight_list
        self.normalization = normalization
        self.normalization_axis = normalization_axis

    def compute_similarity_matrix(self, interaction_df: pd.DataFrame) -> None:
        """Compute an hybrid similarity from the mdoels"""
        hybrid_similarity = None
        for model, model_weight in zip(self.model_list, self.model_weight_list):
            # compute the similarity matrix
            model.compute_similarity_matrix(interaction_df)
            sim = model.similarity_matrix
            if self.normalization is not None:
                sim = similaripy.normalization.normalize(
                    sim, norm=self.normalization, axis=self.normalization_axis
                )
            # multiply for the weight
            sim = sim * model_weight
            if hybrid_similarity is None:
                hybrid_similarity = sim
            else:
                hybrid_similarity = hybrid_similarity + sim
        self.similarity_matrix = hybrid_similarity
