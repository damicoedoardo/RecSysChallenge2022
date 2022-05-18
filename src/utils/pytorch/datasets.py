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

    bpr_dataset = TripletsBPRDataset(train, dataset)

    rnd_sampler = RandomSampler(
        bpr_dataset, replacement=True, num_samples=len(bpr_dataset)
    )
    print("Number of training samples: {}".format(len(bpr_dataset)))
    train_dataloader = DataLoader(
        bpr_dataset,
        batch_size=2048,
        num_workers=32,
        sampler=rnd_sampler,
    )

    for b in tqdm(train_dataloader):
        print(b)
