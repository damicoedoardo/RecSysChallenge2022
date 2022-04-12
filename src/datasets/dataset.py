import math
import pickle
from ast import Str
from pathlib import Path
from typing import AnyStr, Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
from black import main
from sklearn import datasets
from src.constant import *
from src.data_reader import DataReader
from tqdm import tqdm


class Dataset:
    """Class storing all preprocessed data"""

    # Unique items in the dataset
    _ITEMS_NUM = 23_691

    _SUBMISSION_FOLDER = Path(__file__).parent.parent / "submissions"

    def __init__(self) -> None:
        self.dr = DataReader()
        self._ensure_dirs()

    def get_preprocessed_data_path(self) -> Path:
        return Path(self.dr.get_data_path() / "preprocessed")

    def get_submission_folder(self) -> Path:
        return self._SUBMISSION_FOLDER

    def get_mapping_dict_folder(self) -> Path:
        return self.get_preprocessed_data_path() / "mapping_dict"

    def get_item_mapping_dicts(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        """Return the item mapping dictionaries

        Returns:
            Tuple[Dict[int, int], Dict[int, int]]: raw_new_mapping dict, new_raw_mapping_dict
        """
        raw_new_path = "raw_new_item_ids_dict.pkl"
        new_raw_path = "new_raw_item_ids_dict.pkl"
        with open(self.get_mapping_dict_folder() / raw_new_path, "rb") as f:
            raw_new_dict = pickle.load(f)
        with open(self.get_mapping_dict_folder() / new_raw_path, "rb") as f:
            new_raw_dict = pickle.load(f)
        return raw_new_dict, new_raw_dict

    def _ensure_dirs(self) -> None:
        """Create the necessary dirs for the preprocessed data"""
        self.get_preprocessed_data_path().mkdir(parents=True, exist_ok=True)
        self.get_submission_folder().mkdir(parents=True, exist_ok=True)
        self.get_mapping_dict_folder().mkdir(parents=True, exist_ok=True)

    def preprocess_data(self) -> None:
        """Remap the item ids to consecutive on all the raw dfs"""

        # load the item features df
        item_features = self.dr.get_item_features()
        unique_item_ids = item_features[ITEM_ID].unique()
        print(f"Unique item ids: {len(unique_item_ids)}")
        mapped_item_ids = np.arange(len(unique_item_ids))
        # create the mappings dictionaries
        raw_new_item_ids_dict = dict(zip(unique_item_ids, mapped_item_ids))
        new_raw_item_ids_dict = {v: k for k, v in raw_new_item_ids_dict.items()}

        # saving mapping dicts
        with open(
            self.get_mapping_dict_folder() / "raw_new_item_ids_dict.pkl", "wb+"
        ) as f:
            pickle.dump(raw_new_item_ids_dict, f)
        with open(
            self.get_mapping_dict_folder() / "new_raw_item_ids_dict.pkl", "wb+"
        ) as f:
            pickle.dump(new_raw_item_ids_dict, f)
        print("- item mapping dicts saved")

        # load and remap all the raw dataframes with the new item ids
        train_sessions = self.dr.get_train_sessions()
        train_purchases = self.dr.get_train_purchases()
        leaderboard_sessions = self.dr.get_test_leaderboard_sessions()
        test_final_sessions = self.dr.get_test_final_sessions()
        candidate_items = self.dr.get_candidate_items()

        raw_dfs = [
            train_sessions,
            train_purchases,
            leaderboard_sessions,
            test_final_sessions,
            candidate_items,
            item_features,
        ]

        raw_df_names = [
            "train_sessions",
            "train_purchases",
            "leaderboard_sessions",
            "test_final_sessions",
            "candidate_items",
            "item_features",
        ]

        # map and save preprocessed data dfs
        for raw_df, raw_df_name in tqdm(zip(raw_dfs, raw_df_names)):
            raw_df[ITEM_ID] = raw_df[ITEM_ID].map(raw_new_item_ids_dict.get)

            df_name = raw_df_name + ".feather"
            raw_df.reset_index(drop=True).to_feather(
                self.get_preprocessed_data_path() / df_name
            )
            print(f"- {raw_df_name} saved!")

    def split_data(self, splits_perc: list[float]) -> None:
        """Split data into train val test userwise"""
        assert (
            len(splits_perc) == 3
        ), "splits perc should have train_perc, val_perc, test_perc"
        train_perc, val_perc, test_perc = splits_perc

        train_sessions = self.get_train_sessions()
        train_purchases = self.get_train_purchases()

        unique_sessions = train_sessions[SESS_ID].unique()
        # set the numpy random seed for reporducibility
        np.random.seed(RANDOM_SEED)
        # shuffle the sessions ids
        np.random.shuffle(unique_sessions)
        num_sessions = len(unique_sessions)

        train_len = math.ceil(num_sessions * train_perc)
        val_len = math.ceil(num_sessions * val_perc)
        test_len = math.ceil(num_sessions * test_perc)

        train_sess_ids, val_sess_ids, test_sess_ids = (
            unique_sessions[:train_len],
            unique_sessions[train_len : (train_len + val_len)],
            unique_sessions[(train_len + val_len) :],
        )

        train_data, train_label = (
            train_sessions[train_sessions[SESS_ID].isin(train_sess_ids)],
            train_purchases[train_purchases[SESS_ID].isin(train_sess_ids)],
        )
        val_data, val_label = (
            train_sessions[train_sessions[SESS_ID].isin(val_sess_ids)],
            train_purchases[train_purchases[SESS_ID].isin(val_sess_ids)],
        )
        test_data, test_label = (
            train_sessions[train_sessions[SESS_ID].isin(test_sess_ids)],
            train_purchases[train_purchases[SESS_ID].isin(test_sess_ids)],
        )

        assert len(train_data[SESS_ID].unique()) == len(
            train_label[SESS_ID].unique()
        ), "train data and label have different number of sessions!"
        assert len(val_data[SESS_ID].unique()) == len(
            val_data[SESS_ID].unique()
        ), "val data and label have different number of sessions!"
        assert len(test_data[SESS_ID].unique()) == len(
            test_data[SESS_ID].unique()
        ), "test data and label have different number of sessions!"

        print(f"Train sessions: {len(train_data[SESS_ID].unique())}")
        print(f"Val sessions: {len(val_data[SESS_ID].unique())}")
        print(f"Test sessions: {len(test_data[SESS_ID].unique())}")

        # save train val and test splits
        train_data.reset_index(drop=True).to_feather(
            self.get_preprocessed_data_path() / "train_data.feather"  # type: ignore
        )
        train_label.reset_index(drop=True).to_feather(
            self.get_preprocessed_data_path() / "train_label.feather"  # type: ignore
        )

        val_data.reset_index(drop=True).to_feather(
            self.get_preprocessed_data_path() / "val_data.feather"  # type: ignore
        )
        val_label.reset_index(drop=True).to_feather(
            self.get_preprocessed_data_path() / "val_label.feather"  # type: ignore
        )

        test_data.reset_index(drop=True).to_feather(
            self.get_preprocessed_data_path() / "test_data.feather"  # type: ignore
        )
        test_label.reset_index(drop=True).to_feather(
            self.get_preprocessed_data_path() / "test_label.feather"  # type: ignore
        )

        print("- Saved succesfully!")

    def get_split(self) -> Dict[str, List[pd.DataFrame]]:
        """Return a dictionary containing the train val and test data

        split = get_split()
        train_data, train_label = split["train"]
        val_data, val_label = split["val"]
        test_data, test_label = split["test"]

        Returns:
            Dict[Str, List[pd.DataFrame]]: key split, list with data df and label df
        """
        split_dict = {}

        train_data, train_label = pd.read_feather(
            self.get_preprocessed_data_path() / "train_data.feather"
        ), pd.read_feather(self.get_preprocessed_data_path() / "train_label.feather")
        train_data[DATE] = pd.to_datetime(train_data[DATE])
        train_label[DATE] = pd.to_datetime(train_label[DATE])

        val_data, val_label = pd.read_feather(
            self.get_preprocessed_data_path() / "val_data.feather"
        ), pd.read_feather(self.get_preprocessed_data_path() / "val_label.feather")
        val_data[DATE] = pd.to_datetime(val_data[DATE])
        val_label[DATE] = pd.to_datetime(val_label[DATE])

        test_data, test_label = pd.read_feather(
            self.get_preprocessed_data_path() / "test_data.feather"
        ), pd.read_feather(self.get_preprocessed_data_path() / "test_label.feather")
        test_data[DATE] = pd.to_datetime(test_data[DATE])
        test_label[DATE] = pd.to_datetime(test_label[DATE])

        split_dict[TRAIN] = [train_data, train_label]
        split_dict[VAL] = [val_data, val_label]
        split_dict[TEST] = [test_data, test_label]

        return split_dict

    ##########################################
    ##### GET PREPROCESSED DATA METHODS ######
    ##########################################

    def get_train_sessions(self) -> pd.DataFrame:
        path = self.get_preprocessed_data_path() / Path("train_sessions.feather")
        df = pd.read_feather(path)
        df[DATE] = pd.to_datetime(df[DATE])
        return df

    def get_train_purchases(self) -> pd.DataFrame:
        path = self.get_preprocessed_data_path() / Path("train_purchases.feather")
        df = pd.read_feather(path)
        df[DATE] = pd.to_datetime(df[DATE])
        return df

    def get_test_leaderboard_sessions(self) -> pd.DataFrame:
        path = self.get_preprocessed_data_path() / Path("leaderboard_sessions.feather")
        df = pd.read_feather(path)
        return df

    def get_test_final_sessions(self) -> pd.DataFrame:
        path = self.get_preprocessed_data_path() / Path("test_final_sessions.feather")
        df = pd.read_feather(path)
        return df

    def get_candidate_items(self) -> pd.DataFrame:
        path = self.get_preprocessed_data_path() / Path("candidate_items.feather")
        df = pd.read_feather(path)
        return df

    def get_item_features(self) -> pd.DataFrame:
        path = self.get_preprocessed_data_path() / Path("item_features.feather")
        df = pd.read_feather(path)
        return df


if __name__ == "__main__":
    dataset = Dataset()
    # dataset.preprocess_data()
    # dataset.split_data(splits_perc=[0.8, 0.1, 0.1])
    split_dict = dataset.get_split()
    train, train_label = split_dict[TRAIN]
    print(train)
    print(train_label)