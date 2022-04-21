import argparse
from typing import Union

import pandas as pd
import wandb
from src.constant import *
from src.datasets.dataset import Dataset
from src.evaluation import compute_mrr
from src.models.ease.ease import EASE

if __name__ == "__main__":
    parser = argparse.ArgumentParser("train itemknn")

    # Model parameters
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--time_weight", default="None")
    parser.add_argument("--time_score", default="None")

    parser.add_argument("--remove_seen", type=bool, default=True)
    parser.add_argument("--cutoff", type=int, default=100)

    # parse command line
    args = vars(parser.parse_args())
    if args["time_weight"] == "None":
        args["time_weight"] = None
    else:
        args["time_weight"] = float(args["time_weight"])
    if args["time_score"] == "None":
        args["time_score"] = None
    else:
        args["time_score"] = float(args["time_score"])

    # initialize wandb
    wandb.init(config=args)

    dataset = Dataset()
    split_dict = dataset.get_split()

    train, train_label = split_dict[TRAIN]
    val, val_label = split_dict[VAL]
    test, test_label = split_dict[TEST]

    split_dict = dataset.get_split()
    train, train_label = split_dict[TRAIN]
    val, val_label = split_dict[VAL]

    full_data = dataset.get_train_sessions()
    train_pur = pd.concat([full_data, train_label], axis=0)

    ts = args["time_score"]
    if ts is not None:
        train_pur["last_buy"] = train_pur.groupby(SESS_ID)[DATE].transform(max)
        train_pur["first_buy"] = train_pur.groupby(SESS_ID)[DATE].transform(min)
        train_pur["time_score"] = 1 / (
            (
                (train_pur["last_buy"] - train_pur[DATE]).apply(
                    lambda x: x.total_seconds() / 3600
                )
            )
            + 1
        )
        train_pur = train_pur[train_pur["time_score"] >= ts]

    model = EASE(dataset, time_weight=args["time_weight"], l2=args["l2"])
    model.compute_similarity_matrix(train_pur)

    recs = model.recommend(
        interactions=val, remove_seen=True, cutoff=100, leaderboard=False
    )
    val_mrr = compute_mrr(recs, val_label)
    # log the metric
    wandb.log({"val_mrr": val_mrr})

    recs = model.recommend(
        interactions=test, remove_seen=True, cutoff=100, leaderboard=False
    )
    test_mrr = compute_mrr(recs, test_label)
    # log the metric
    wandb.log({"test_mrr": test_mrr})
