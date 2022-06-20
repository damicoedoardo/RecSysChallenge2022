import math
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from xgboost import plot_importance

from src.constant import *
from src.datasets.dataset import Dataset

TRAIN_PERC = 0.8
VAL_PERC = 0.10
TEST_PERC = 0.10

KIND = "train"
DATASET_NAME = "f_dataset"
DATASET = f"{DATASET_NAME}.feather"
MODEL_NAME = f"xgb_{DATASET_NAME}.json"

if __name__ == "__main__":
    dataset = Dataset()

    base_path = None
    if KIND == "train":
        base_path = dataset.get_train_xgboost_dataset_folder()
    elif KIND == "leaderboard":
        base_path = dataset.get_leaderboard_xgboost_dataset_folder()
    elif KIND == "final":
        base_path = dataset.get_final_xgboost_dataset_folder()
    else:
        raise NotImplementedError("Kind passed is wrong!")

    path = base_path / Path(f"{DATASET_NAME}.feather")
    features_df = pd.read_feather(path)

    unique_session = features_df[SESS_ID].unique()
    print(f"Unique session:{len(unique_session)}")
    train_len = math.ceil(len(unique_session) * TRAIN_PERC)
    val_len = math.ceil(len(unique_session) * VAL_PERC)
    test_len = math.ceil(len(unique_session) * TEST_PERC)

    np.random.seed()
    # np.random.seed(1024)
    np.random.shuffle(unique_session)
    train_session, val_session, test_session = (
        unique_session[:train_len],
        unique_session[train_len : train_len + val_len],
        unique_session[train_len + val_len :],
    )

    train_df = features_df[features_df[SESS_ID].isin(train_session)]
    val_df = features_df[features_df[SESS_ID].isin(val_session)]
    test_df = features_df[features_df[SESS_ID].isin(test_session)]

    #####
    X_train = train_df.loc[:, ~train_df.columns.isin([SESS_ID, ITEM_ID, "relevance"])]
    Y_train = train_df["relevance"].copy().values
    qid_train = train_df[SESS_ID].copy().values

    X_val = val_df.loc[:, ~val_df.columns.isin([SESS_ID, ITEM_ID, "relevance"])]
    Y_val = val_df["relevance"].copy().values
    qid_val = val_df[SESS_ID].copy().values

    X_test = test_df.loc[:, ~test_df.columns.isin([SESS_ID, ITEM_ID, "relevance"])]
    Y_test = test_df["relevance"].copy().values
    qid_test = test_df[SESS_ID].copy().values

    model = xgb.XGBRanker(
        tree_method="hist",
        booster="gbtree",
        objective="rank:map",
        random_state=RANDOM_SEED,
        learning_rate=0.1,
        colsample_bytree=1,
        reg_lambda=0.01,
        reg_alpha=0.01,
        gamma=0.01,
        # eta=0.01,
        max_depth=4,
        n_estimators=500,
        # num_leaves=20,
        subsample=0.5,
        # sampling_method="gradient_based"
        # n_gpus=-1
        gpu_id=1,
    )

    model.fit(
        X_train,
        Y_train,
        qid=qid_train,
        eval_set=[(X_val, Y_val), (X_test, Y_test)],
        eval_qid=[qid_val, qid_test],
        eval_metric=["map@100"],
        verbose=True,
        early_stopping_rounds=20,
    )

    model_name = dataset.get_xgboost_model_folder() / MODEL_NAME
    model.save_model(model_name)
