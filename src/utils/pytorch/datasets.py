import math
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from src.constant import *
from src.datasets.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import RandomSampler
from tqdm import tqdm
from utils.decorator import timing


class CCLDataset(TorchDataset):
    """Dataset to create sample accordingly to the Cosine Contrastive Loss function"""

    def __init__(
        self,
        dataset,
        purchase_df: pd.DataFrame,
        train_df: pd.DataFrame,
        neg_samples: int,
    ):
        self.dataset = dataset
        self.item_list = np.arange(dataset._ITEMS_NUM)
        self.purchase_df = purchase_df
        self.train_df = train_df
        self.neg_samples = neg_samples

        self.purchase_array = purchase_df[[SESS_ID, ITEM_ID]].values

        print("Creating user item dictionary lookup")
        # {
        #   sess_id:
        #   {
        #       item_id: 1,
        #       ...,
        #   },
        # }
        all_df = pd.concat([train_df, purchase_df], axis=0)
        # self.purchase_array = all_df[[SESS_ID, ITEM_ID]].values
        user, item = zip(*all_df[[SESS_ID, ITEM_ID]].values)

        train_score_dict = defaultdict(lambda: {})
        for u, i in tqdm(zip(user, item)):
            if i not in train_score_dict[u]:
                train_score_dict[u][i] = 1

        self.train_score_dict = train_score_dict

    def _get_random_key(self, list):
        L = len(list)
        i = np.random.randint(0, L)
        return list[i]

    def __len__(self):
        # the whole number of samples
        return len(self.purchase_array)

    def __getitem__(self, idx):
        # pick a session and the purchase associated to it
        sess_id, sess_purchase_id = self.purchase_array[idx]

        # sample negative samples
        negative_samples_list = []
        while len(negative_samples_list) < self.neg_samples:
            j = self._get_random_key(self.item_list)
            while j in self.train_score_dict[sess_id]:
                j = self._get_random_key(self.item_list)
            negative_samples_list.append(j)
        return sess_id, sess_purchase_id, negative_samples_list


class TripletsBPRDataset(TorchDataset):
    """Dataset to sample triplets for BPR optimization"""

    def __init__(self, purchase_df: pd.DataFrame, train_df: pd.DataFrame, dataset):

        self.dataset = dataset
        self.item_list = np.arange(dataset._ITEMS_NUM)
        self.purchase_df = purchase_df

        # filter on timescore
        # train_df["last_buy"] = train_df.groupby(SESS_ID)[DATE].transform(max)
        # train_df["first_buy"] = train_df.groupby(SESS_ID)[DATE].transform(min)
        # train_df["time_score"] = 1 / (
        #     (
        #         (train_df["last_buy"] - train_df[DATE]).apply(
        #             lambda x: x.total_seconds() / 3600
        #         )
        #     )
        #     + 1
        # )
        # train_df = train_df[train_df["time_score"] >= 0.7]

        # create lookup dictionary
        print("Creating user item dictionary lookup")
        all_df = pd.concat([train_df, purchase_df], axis=0)
        # create interaction list [(0, 1000), ...]

        # interaction_list = all_df[[SESS_ID, ITEM_ID]].values
        interaction_list = purchase_df[[SESS_ID, ITEM_ID]].values
        self.interactions_list = interaction_list

        user, item = zip(*all_df[[SESS_ID, ITEM_ID]].values)

        train_score_dict = defaultdict(lambda: {})
        for u, i in tqdm(zip(user, item)):
            if i not in train_score_dict[u]:
                train_score_dict[u][i] = 1

        self.train_score_dict = train_score_dict

    def _get_random_key(self, list):
        L = len(list)
        i = np.random.randint(0, L)
        return list[i]

    def __getitem__(self, item):
        # first pick at iteration associated to item
        u, i = self.interactions_list[item]

        # sample negative sample
        j = self._get_random_key(self.item_list)
        while j in self.train_score_dict[u]:
            j = self._get_random_key(self.item_list)
        return u, i, j

    def __len__(self):
        return len(self.interactions_list)


class ContextAwarePaddedDataset(TorchDataset):
    def __init__(
        self,
        dataset,
        purchase_df: pd.DataFrame,
        train_df: pd.DataFrame,
        negative_samples_num: int,
        padding_idx: int,
        k: int,
    ) -> None:
        self.dataset = dataset
        self.negative_samples_num = negative_samples_num
        self.purchase_df = purchase_df
        self.train_df = train_df

        # last k items to keep for the session
        self.k = k
        self.padding_idx = padding_idx

        self.sess2items = dataset.get_sess2items()
        self.item_list = np.arange(dataset._ITEMS_NUM)
        self.train_score_dict = self._create_sess2items_lookup()
        self.padded_sess = self._pad_sessions()

    def _create_sess2items_lookup(self):
        # sess2items dictionary lookup, data structure exploitex for negative sampling
        print("Creating sess2items dictionary lookup")
        all_df = pd.concat([self.train_df, self.purchase_df], axis=0)
        user, item = zip(*all_df[[SESS_ID, ITEM_ID]].values)
        train_score_dict = defaultdict(lambda: {})
        for u, i in tqdm(zip(user, item)):
            if i not in train_score_dict[u]:
                train_score_dict[u][i] = 1
        return train_score_dict

    @timing
    def _pad_sessions(self):
        print("Padding sessions")
        sess2items = self.sess2items
        padded_sess2items = sess2items.apply(
            lambda x: np.array(x[-self.k :])
            if len(x) >= self.k
            else np.pad(x, (self.k - len(x), 0), constant_values=self.padding_idx)
        )
        # padded_sess = np.array(
        #     [
        #         s[-self.k :]
        #         if len(s) >= self.k
        #         else np.pad(s, (self.k - len(s), 0), constant_values=self.padding_idx)
        #         for s in sess2items_array
        #     ]
        # )
        return padded_sess2items

    def __len__(self):
        return len(self.purchase_df)

    def _get_random_key(self, list):
        L = len(list)
        i = np.random.randint(0, L)
        return list[i]

    def __getitem__(self, item):
        # first pick at iteration associated to item
        purchase_row = self.purchase_df.iloc[item]
        purchase = purchase_row[ITEM_ID]
        sess_id = purchase_row[SESS_ID]
        sess2items = self.padded_sess.loc[sess_id]

        # sample negative samples
        negative_samples_list = []
        while len(negative_samples_list) < self.negative_samples_num:
            j = self._get_random_key(self.item_list)
            while j in self.train_score_dict[sess_id]:
                j = self._get_random_key(self.item_list)
            negative_samples_list.append(j)
        negative_samples = np.array(negative_samples_list)
        return sess2items, purchase, negative_samples


class ContextAwareContrastiveDataset(TorchDataset):
    def __init__(
        self,
        dataset,
        purchase_df: pd.DataFrame,
        train_df: pd.DataFrame,
        negative_samples_num: int,
        context_samples_num: int,
        padding_idx: int,
        k: int,
    ) -> None:
        self.dataset = dataset
        self.context_samples_num = context_samples_num
        self.negative_samples_num = negative_samples_num
        self.purchase_df = purchase_df
        self.train_df = train_df

        # last k items to keep for the session
        self.k = k
        self.padding_idx = padding_idx

        sess2items = dataset.get_sess2items()
        # truncate the sequences to k
        self.sess2items = sess2items.apply(lambda x: np.array(x[-self.k :]))
        # create ausiliary structure where to sample the context items for the session
        self.context_sess2items = self.sess2items.apply(
            lambda x: x
            if len(x) >= self.context_samples_num
            else np.tile(x, math.ceil(self.context_samples_num / len(x)))
        )
        self.item_list = np.arange(dataset._ITEMS_NUM)
        self.train_score_dict = self._create_sess2items_lookup()
        self.padded_sess = self._pad_sessions()

    def _create_sess2items_lookup(self):
        # sess2items dictionary lookup, data structure exploitex for negative sampling
        print("Creating sess2items dictionary lookup")
        all_df = pd.concat([self.train_df, self.purchase_df], axis=0)
        user, item = zip(*all_df[[SESS_ID, ITEM_ID]].values)
        train_score_dict = defaultdict(lambda: {})
        for u, i in tqdm(zip(user, item)):
            if i not in train_score_dict[u]:
                train_score_dict[u][i] = 1
        return train_score_dict

    @timing
    def _pad_sessions(self):
        print("Padding sessions")
        sess2items = self.sess2items
        padded_sess2items = sess2items.apply(
            lambda x: x
            if len(x) >= self.k
            else np.pad(x, (self.k - len(x), 0), constant_values=self.padding_idx)
        )
        return padded_sess2items

    def __len__(self):
        return len(self.purchase_df)

    def _get_random_key(self, list):
        L = len(list)
        i = np.random.randint(0, L)
        return list[i]

    def __getitem__(self, item):
        # first pick at iteration associated to item
        purchase_row = self.purchase_df.iloc[item]
        purchase = purchase_row[ITEM_ID]
        sess_id = purchase_row[SESS_ID]
        sess2items = self.padded_sess.loc[sess_id]

        context_items_arr = self.context_sess2items.loc[sess_id]
        # shuffle context items
        np.random.shuffle(context_items_arr)
        context_items = np.array(context_items_arr[: self.context_samples_num])

        # sample negative samples
        negative_samples_list = []
        while len(negative_samples_list) < self.negative_samples_num:
            j = self._get_random_key(self.item_list)
            while j in self.train_score_dict[sess_id]:
                j = self._get_random_key(self.item_list)
            negative_samples_list.append(j)
        negative_samples = np.array(negative_samples_list)
        return sess2items, purchase, negative_samples, context_items


if __name__ == "__main__":
    dataset = Dataset()
    split_dict = dataset.get_split()
    train, train_label = split_dict[TRAIN]

    ccl_dataset = ContextAwareContrastiveDataset(
        dataset=dataset,
        train_df=train,
        purchase_df=train_label,
        negative_samples_num=2,
        context_samples_num=3,
        padding_idx=dataset._ITEMS_NUM,
        k=5,
    )

    rnd_sampler = RandomSampler(
        ccl_dataset, replacement=False, num_samples=len(ccl_dataset)
    )
    print("Number of training samples: {}".format(len(ccl_dataset)))
    train_dataloader = DataLoader(
        ccl_dataset,
        batch_size=2,
        num_workers=32,
        sampler=rnd_sampler,
    )

    for b in tqdm(train_dataloader):
        print(b)
        break
