import logging
import math
from abc import ABC, abstractmethod
from textwrap import dedent
from time import time
from tkinter.messagebox import NO
from typing import Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sps
import similaripy
import torch
from p_tqdm import p_imap, p_map
from pathos.pools import ProcessPool
from tqdm import tqdm

from src.constant import *
from src.data_reader import DataReader
from src.utils.decorator import timing
from src.utils.logger import set_color
from src.utils.sparse_matrix import get_top_k, interactions_to_sparse_matrix

logger = logging.getLogger(__name__)


class AbstractRecommender(ABC):
    """Interface for recommender system algorithms"""

    name = "Abstract Recommender"

    def __init__(self, dataset):
        # self.train_data = dataset.get_train_df()
        self.dataset = dataset

    @abstractmethod
    def predict(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Compute items scores for each user inside interactions
        Args:
            interactions (pd.DataFrame): user interactions
        Returns:
            pd.DataFrame: items scores for each user
        """
        pass

    @staticmethod
    def remove_seen_items(
        scores: pd.DataFrame,
        interactions: pd.DataFrame,
        white_list_mb_item: np.ndarray = None,
    ) -> pd.DataFrame:
        """Methods to set scores of items used at training time to `-np.inf`

        Args:
            scores (pd.DataFrame): items scores for each user, indexed by user id
            interactions (pd.DataFrame): interactions of the users for which retrieve predictions
            items_multiple_buy (pd.DataFrame): items that can be recommended multiple times

        Returns:
            pd.DataFrame: dataframe of scores for each user indexed by user id
        """

        logger.debug(set_color(f"Removing seen items", "cyan"))

        # if white_list_mb_item is not None:
        #     print("Considering white list items...")
        #     interactions = interactions[
        #         ~(interactions[ITEM_ID].isin(white_list_mb_item))
        #     ]  # type: ignore

        user_list = interactions[SESS_ID].values  # type: ignore
        item_list = interactions[ITEM_ID].values  # type: ignore

        scores_array = scores.values

        user_index = scores.index.values
        arange = np.arange(len(user_index))
        mapping_dict = dict(zip(user_index, arange))
        user_list_mapped = np.array([mapping_dict.get(u) for u in user_list])

        scores_array[user_list_mapped, item_list] = -np.inf

        if white_list_mb_item is not None:
            print("Considering white list items...")
            # creating mask for candidate items
            mask_array = np.ones(scores_array.shape[1], dtype=bool)
            mask_array[white_list_mb_item] = False
            scores_array[:, mask_array] = -np.inf

        scores = pd.DataFrame(scores_array, index=user_index)

        return scores

    def recommend(
        self,
        interactions: pd.DataFrame,
        cutoff: int = 12,
        remove_seen: bool = True,
        leaderboard: bool = False,
    ) -> pd.DataFrame:
        """
        Give recommendations up to a given cutoff to users inside `user_idxs` list

        Note:
            predictions are in the following format | userID | itemID | prediction | item_rank

        Args:
            cutoff (int): cutoff used to retrieve the recommendations
            interactions (pd.DataFrame): interactions of the users for which retrieve predictions
            batch_size (int): size of user batch to retrieve recommendations for,
                If -1 no batching procedure is done
            remove_seen (bool): remove items that have been bought ones from the prediction

        Returns:
            pd.DataFrame: DataFrame with predictions for users
        """
        whitelist_items = None
        if leaderboard:
            whitelist_items = self.dataset.get_candidate_items().values.squeeze()

        logger.info(set_color(f"Recommending items MONOCORE", "cyan"))

        unique_user_ids = interactions[SESS_ID].unique()
        logger.info(set_color(f"Predicting for: {len(unique_user_ids)} users", "cyan"))

        # MONO-CORE VERSION
        scores = self.predict(interactions)
        # set the score of the items used during the training to -inf
        if remove_seen:
            scores = AbstractRecommender.remove_seen_items(
                scores, interactions, white_list_mb_item=whitelist_items
            )
        array_scores = scores.to_numpy()
        user_ids = scores.index.values
        items, scores = get_top_k(scores=array_scores, top_k=cutoff, sort_top_k=True)
        # create user array to match shape of retrievied items
        users = np.repeat(user_ids, cutoff).reshape(len(user_ids), -1)
        recommendation_df = pd.DataFrame(
            zip(users.flatten(), items.flatten(), scores.flatten()),  # type: ignore
            columns=[SESS_ID, ITEM_ID, PREDICTION_COL],
        )

        # add item rank
        recommendation_df["rank"] = np.tile(
            np.arange(1, cutoff + 1), len(unique_user_ids)
        )

        return recommendation_df  # type: ignore

    def recommend_multicore(
        self,
        interactions: pd.DataFrame,
        cutoff: int = 12,
        remove_seen: bool = True,
        white_list_mb_item: Union[np.ndarray, None] = None,
        batch_size: int = -1,
        num_cpus: int = 5,
    ) -> pd.DataFrame:
        """
        Give recommendations up to a given cutoff to users inside `user_idxs` list

        Note:
            predictions are in the following format | userID | itemID | prediction | item_rank

        Args:
            cutoff (int): cutoff used to retrieve the recommendations
            interactions (pd.DataFrame): interactions of the users for which retrieve predictions
            batch_size (int): size of user batch to retrieve recommendations for,
                If -1 no batching procedure is done
            num_cpus (int): number of cores to use to parallelise batch recommendations
            remove_seen (bool): remove items that have been bought ones from the prediction

        Returns:
            pd.DataFrame: DataFrame with predictions for users
        """
        # if interactions is None we are predicting for the wh  ole users in the train dataset
        logger.debug(set_color(f"Recommending items MULTICORE", "cyan"))

        unique_user_ids = interactions[SESS_ID].unique()
        # if  batch_size == -1 we are not batching the recommendation process
        num_batches = (
            1 if batch_size == -1 else math.ceil(len(unique_user_ids) / batch_size)
        )
        user_batches = np.array_split(unique_user_ids, num_batches)

        # MULTI-CORE VERSION
        train_dfs = [
            interactions[interactions[SESS_ID].isin(u_batch)]
            for u_batch in user_batches
        ]

        def _rec(interactions_df, white_list_mb_item=None):
            scores = self.predict(interactions_df)
            # set the score of the items used during the training to -inf
            if remove_seen:
                scores = AbstractRecommender.remove_seen_items(
                    scores, interactions_df, white_list_mb_item
                )
            array_scores = scores.to_numpy()
            user_ids = scores.index.values
            # TODO: we can use GPU here (tensorflow ?)
            items, scores = get_top_k(
                scores=array_scores, top_k=cutoff, sort_top_k=True
            )
            # create user array to match shape of retrievied items
            users = np.repeat(user_ids, cutoff).reshape(len(user_ids), -1)
            recs_df = pd.DataFrame(
                zip(users.flatten(), items.flatten(), scores.flatten()),  # type: ignore
                columns=[SESS_ID, ITEM_ID, PREDICTION_COL],
            )
            return recs_df

        if batch_size == -1:
            recs_dfs_list = [_rec(train_dfs[0])]
        else:
            # pool = ProcessPool(nodes=num_cpus)
            # results = pool.imap(_rec, train_dfs)
            # recs_dfs_list = list(results)
            if white_list_mb_item is not None:
                reps_white_list_mb_item = np.repeat(
                    np.array(white_list_mb_item)[np.newaxis, :], len(train_dfs), axis=0
                )
                recs_dfs_list = p_map(
                    _rec, train_dfs, reps_white_list_mb_item, num_cpus=num_cpus
                )
            else:
                recs_dfs_list = p_map(_rec, train_dfs, num_cpus=num_cpus)

        # concat all the batch recommendations dfs
        recommendation_df = pd.concat(recs_dfs_list, axis=0)
        # add item rank
        recommendation_df["rank"] = np.tile(
            np.arange(1, cutoff + 1), len(unique_user_ids)
        )

        return recommendation_df


class ItemSimilarityRecommender(AbstractRecommender, ABC):
    """Item similarity matrix recommender interface

    Each recommender extending this class has to implement compute_similarity_matrix() method
    """

    def __init__(
        self,
        dataset,
        time_weight: Union[float, None] = None,
    ):
        super().__init__(dataset=dataset)
        self.time_weight = time_weight
        self.similarity_matrix = None

    @abstractmethod
    def compute_similarity_matrix(self):
        """Compute similarity matrix and assign it to self.similarity_matrix"""
        pass

    def predict(self, interactions):
        assert (
            self.similarity_matrix is not None
        ), "Similarity matrix is not computed, call compute_similarity_matrix()"
        if self.time_weight:
            logger.debug(set_color("Predicting using time_weight importance...", "red"))
        sparse_interaction, user_mapping_dict, _ = interactions_to_sparse_matrix(
            interactions,
            items_num=self.dataset._ITEMS_NUM,
            users_num=None,
            time_weight=self.time_weight,
        )
        # compute scores as the dot product between user interactions and the similarity matrix
        if not sps.issparse(self.similarity_matrix):
            logger.debug(set_color(f"DENSE Item Similarity MUL...", "cyan"))
            scores = sparse_interaction @ self.similarity_matrix
        else:
            logger.debug(set_color(f"SPARSE Item Similarity MUL...", "cyan"))
            scores = sparse_interaction @ self.similarity_matrix
            scores = scores.toarray()

        scores_df = pd.DataFrame(scores, index=list(user_mapping_dict.keys()))
        return scores_df


class UserSimilarityRecommender(AbstractRecommender, ABC):
    """Item similarity matrix recommender interface

    Each recommender extending this class has to implement compute_similarity_matrix() method
    """

    def __init__(
        self,
        dataset,
        time_weight: Union[float, None] = None,
    ):
        super().__init__(dataset=dataset)
        self.time_weight = time_weight
        self.similarity_matrix = None

    @abstractmethod
    def compute_similarity_matrix(self):
        """Compute similarity matrix and assign it to self.similarity_matrix"""
        pass

    def predict(self, interactions: pd.DataFrame, cutoff: int, remove_seen: bool):
        assert (
            self.similarity_matrix is not None
        ), "Similarity matrix is not computed, call compute_similarity_matrix()"
        if self.time_weight:
            logger.debug(set_color("Predicting using time_weight importance...", "red"))
        sparse_interaction, user_mapping_dict, _ = interactions_to_sparse_matrix(
            interactions,
            items_num=self.dataset._ITEMS_NUM,
            users_num=None,
            time_weight=self.time_weight,
        )
        # compute scores as the dot product between user interactions and the similarity matrix
        if not sps.issparse(self.similarity_matrix):
            raise NotImplementedError(
                "user similarity can only be used with sparse similarity matrices!"
            )
        else:
            logger.debug(set_color(f"SPARSE Item Similarity MUL...", "cyan"))

            print(self.similarity_matrix.shape)
            print(sparse_interaction.shape)
            print(cutoff)

            filter_cols = sparse_interaction if remove_seen else None
            scores = similaripy.dot_product(
                self.similarity_matrix.T,
                sparse_interaction,
                k=cutoff,
                # filter_cols=filter_cols,
            )
            # scores = self.similarity_matrix.dot(sparse_interaction)
            scores = scores.toarray()
            print(scores)

        scores_df = pd.DataFrame(scores, index=list(user_mapping_dict.keys()))
        return scores_df


class RepresentationBasedRecommender(AbstractRecommender, ABC):
    """Representation based recommender interface"""

    def __init__(self, dataset):
        super().__init__(dataset)

    @abstractmethod
    def compute_representations(
        self, interactions: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute users and items representations

        Args:
            interactions (pd.Dataframe): interactions of the users for which
                retrieve predictions stored inside a pd.DataFrame

        Returns:
            pd.DataFrame, pd.DataFrame: user representations, item representations
        """
        pass

    def predict(self, interactions):
        # TODO CHECK THAT

        # we user the dor product between user and item embeddings to predict the user preference scores
        users_repr_df, items_repr_df = self.compute_representations(interactions)

        assert isinstance(users_repr_df, pd.DataFrame) and isinstance(
            items_repr_df, pd.DataFrame
        ), "Representations have to be stored inside pd.DataFrane objects!\n user: {}, item: {}".format(
            type(users_repr_df), type(items_repr_df)
        )
        assert (
            users_repr_df.shape[1] == items_repr_df.shape[1]
        ), "Users and Items representations have not the same shape!\n user: {}, item: {}".format(
            users_repr_df.shape[1], items_repr_df.shape[1]
        )

        # sort items representations
        items_repr_df.sort_index(inplace=True)

        # compute the scores as dot product between users and items representations
        device = torch.device(
            "cuda:{}".format(2) if (torch.cuda.is_available()) else "cpu"
        )

        u_tensor = torch.tensor(users_repr_df.to_numpy()).to(device)
        i_tensor = torch.tensor(items_repr_df.to_numpy()).to(device)
        arr_scores = torch.matmul(u_tensor, torch.t(i_tensor)).cpu().numpy()
        # arr_scores = users_repr_df.to_numpy().dot(items_repr_df.to_numpy().T)

        scores = pd.DataFrame(arr_scores, index=users_repr_df.index)
        return scores
