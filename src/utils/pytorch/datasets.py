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


if __name__ == "__main__":
    dataset = Dataset()
    split_dict = dataset.get_split()
    train, train_label = split_dict[TRAIN]

    ccl_dataset = CCLDataset(
        dataset=dataset, train_df=train, purchase_df=train_label, neg_samples=2
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
