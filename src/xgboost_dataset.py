from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from black import main

from src.constant import *
from src.data_reader import DataReader
from src.datasets.dataset import Dataset
from src.evaluation import compute_mrr
from src.models.itemknn.itemknn import ItemKNN

KIND = "train"
DATASET_NAME = "f_dataset"
RECS_LIST = [
    "EASE_tw_full",
    "EASE",
    "cease_tw",
    "cease",
    "hybrid_ease_tw",
    "rp3b_tw",
    "rp3b",
    "item_knn_tw",
    "item_knn",
    "context_attn",
]
# RECS_LIST = ["hybrid_ease", "cassandra"]
if __name__ == "__main__":
    dataset = Dataset()
    split_dict = dataset.get_split()
    train, train_label = split_dict[TRAIN]
    val, val_label = split_dict[VAL]
    test, test_label = split_dict[TEST]

    val_test = pd.concat([val, test])
    val_test_label = pd.concat([val_label, test_label])

    recs_df_list = []
    for model in RECS_LIST:
        recs = dataset.get_recs_df(model, kind=KIND)
        recs_df_list.append(recs)

    # recs_ease = dataset.get_recs_df("EASE_tw", kind=KIND)
    # recs_cease = dataset.get_recs_df("cease_tw", kind=KIND)
    # recs_knn = dataset.get_recs_df("Splus_tw", kind=KIND)

    recs = reduce(
        lambda df1, df2: pd.merge(
            df1,
            df2,
            on=[SESS_ID, ITEM_ID],
            how="outer",
        ),
        recs_df_list,
    )
    # recs = pd.merge(
    #     recs_ease,
    #     recs_knn,
    #     on=[SESS_ID, ITEM_ID],
    #     how="outer",
    # )
    print(f"Average recs per user: {recs.groupby(SESS_ID).size().mean()}")
    print(recs)

    if KIND == "train":
        val_test_label = val_test_label.rename(columns={ITEM_ID: "relevance"})
        val_test_label = val_test_label.drop(DATE, axis=1)
        print(f"GT len: {len(val_test_label)}")

        merged = pd.merge(
            recs,
            val_test_label,
            left_on=[SESS_ID, ITEM_ID],
            right_on=[SESS_ID, "relevance"],
            how="left",
        )

        merged.loc[merged["relevance"].notnull(), "relevance"] = 1
        merged["hit_sum"] = merged.groupby(SESS_ID)["relevance"].transform("sum")

        merged_filtered = merged[merged["hit_sum"] > 0]

        # we can drop the hit sum column
        merged_filtered = merged_filtered.drop("hit_sum", axis=1)

        # fill with 0 the nan values, the nan are the one for which we do not do an hit
        merged_filtered["relevance"] = merged_filtered["relevance"].fillna(0)

        print(f"Retained sessions: {merged_filtered[SESS_ID].nunique()}")
    else:
        merged_filtered = recs

    item_features = dataset.get_oh_item_features().reset_index()
    filter_cols_items = [c for c in item_features.columns if "cat" in c]
    filter_cols_items.append(ITEM_ID)
    item_features = item_features[filter_cols_items]

    sess_features = dataset.get_sess_features()
    filter_cols = [c for c in sess_features.columns if "cat" in c]
    filter_cols.append("session_length")
    filter_cols.append(SESS_ID)
    # sess_features = sess_features[[SESS_ID, "session_length"]]
    sess_features = sess_features[filter_cols]

    # print("merging features...")
    recs_item_f = pd.merge(merged_filtered, item_features, on=ITEM_ID)
    # print("item features done...")
    recs_final = pd.merge(recs_item_f, sess_features, on=SESS_ID)
    print("session features done...")

    # recs_final = merged_filtered

    print(recs_final)
    recs_final = recs_final.sort_values(SESS_ID)

    base_path = None
    if KIND == "train":
        base_path = dataset.get_train_xgboost_dataset_folder()
    elif KIND == "leaderboard":
        base_path = dataset.get_leaderboard_xgboost_dataset_folder()
    elif KIND == "final":
        base_path = dataset.get_final_xgboost_dataset_folder()
    else:
        raise NotImplementedError("Kind passed is wrong!")
    recs_final.reset_index(drop=True).to_feather(
        base_path / f"{DATASET_NAME}.feather"  # type: ignore
    )
    print(f"- Dataset:{DATASET_NAME}, kind: {KIND}, saved succesfully")
