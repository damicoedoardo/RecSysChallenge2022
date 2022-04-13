import argparse
from calendar import c
from typing import Union

import pandas as pd
import wandb
from src.constant import *
from src.datasets.dataset import Dataset
from src.evaluation import compute_mrr
from src.models.itemknn.itemknn import ItemKNN

if __name__ == "__main__":
    parser = argparse.ArgumentParser("train itemknn")

    # Model parameters
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--shrink", type=int, default=0)
    parser.add_argument("--normalization", type=bool, default=False)
    parser.add_argument("--time_weight", default=None)
    parser.add_argument("--l", type=float, default=0.5)
    parser.add_argument("--t1", type=float, default=1.0)
    parser.add_argument("--t2", type=float, default=1.0)
    parser.add_argument("--c", type=float, default=0.5)

    parser.add_argument("--remove_seen", type=bool, default=True)
    parser.add_argument("--cutoff", type=int, default=100)

    # parse command line
    args = vars(parser.parse_args())

    # initialize wandb
    wandb.init(config=args)

    dataset = Dataset()
    split_dict = dataset.get_split()

    train, train_label = split_dict[TRAIN]
    val, val_label = split_dict[VAL]
    test, test_label = split_dict[TEST]

    model = ItemKNN(
        dataset,
        topk=args["topk"],
        normalization=args["normalization"],
        t1=args["t1"],
        t2=args["t2"],
        c=args["c"],
        l=args["l"],
        time_weight=args["time_weight"],
    )
    model.compute_similarity_matrix(train)

    recs = model.recommend(
        interactions=val, remove_seen=True, cutoff=100, leaderboard=False
    )

    mrr = compute_mrr(recs, val_label)
    # log the metric
    wandb.log({"mrr": mrr})
