import math
import pickle
from ast import Str
from datetime import datetime, timedelta
from pathlib import Path
from typing import AnyStr, Dict, List, Tuple

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

    _SUBMISSION_FOLDER = Path(__file__).parent.parent.parent / "submissions"

    def __init__(self) -> None:
        self.dr = DataReader()
        self._ensure_dirs()

    def get_preprocessed_data_path(self) -> Path:
        return Path(self.dr.get_data_path() / "preprocessed")

    def get_saved_models_path(self) -> Path:
        return Path(self.dr.get_data_path() / "saved_models")

    def get_submission_folder(self) -> Path:
        return self._SUBMISSION_FOLDER

    def get_mapping_dict_folder(self) -> Path:
        return self.get_preprocessed_data_path() / "mapping_dict"

    def get_recs_df_folder(self) -> Path:
        return self.get_preprocessed_data_path() / "recs_df"

    def get_train_recs_df_folder(self) -> Path:
        return self.get_recs_df_folder() / "train"

    def get_leaderboard_recs_df_folder(self) -> Path:
        return self.get_recs_df_folder() / "leaderboard"

    def get_final_recs_df_folder(self) -> Path:
        return self.get_recs_df_folder() / "final"

    def get_xgboost_dataset_folder(self) -> Path:
        return self.get_preprocessed_data_path() / "xgboost_dataset"

    def get_train_xgboost_dataset_folder(self) -> Path:
        return self.get_xgboost_dataset_folder() / "train"

    def get_leaderboard_xgboost_dataset_folder(self) -> Path:
        return self.get_xgboost_dataset_folder() / "leaderboard"

    def get_final_xgboost_dataset_folder(self) -> Path:
        return self.get_xgboost_dataset_folder() / "final"

    def get_xgboost_model_folder(self) -> Path:
        return self.get_preprocessed_data_path() / "xgb_saved_model"

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
        self.get_saved_models_path().mkdir(parents=True, exist_ok=True)

        self.get_recs_df_folder().mkdir(parents=True, exist_ok=True)
        self.get_train_recs_df_folder().mkdir(parents=True, exist_ok=True)
        self.get_leaderboard_recs_df_folder().mkdir(parents=True, exist_ok=True)
        self.get_final_recs_df_folder().mkdir(parents=True, exist_ok=True)

        self.get_xgboost_dataset_folder().mkdir(parents=True, exist_ok=True)
        self.get_train_xgboost_dataset_folder().mkdir(parents=True, exist_ok=True)
        self.get_leaderboard_xgboost_dataset_folder().mkdir(parents=True, exist_ok=True)
        self.get_final_xgboost_dataset_folder().mkdir(parents=True, exist_ok=True)

        self.get_xgboost_model_folder().mkdir(parents=True, exist_ok=True)

    ##########################################
    ####### PREPROCESS DATA METHODS #########
    ##########################################

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

    def split_data(self) -> None:
        """Split data into train val test userwise

        We replicate the split done by the data owner:
        we consider `train sessions` all the sessions where date <= max_train_date - 1month
        we consider `val_test sessions` all the sessions where date > max_train_date - 1month

        we then split in half the val_test session in validation and test
        """

        train_sessions = self.get_train_sessions()
        train_purchases = self.get_train_purchases()

        max_date = train_sessions[DATE].max()
        train_limit_date = max_date - timedelta(days=31)  # type: ignore

        local_train_sessions = train_sessions[train_sessions[DATE] <= train_limit_date]
        local_val_test_sessions = train_sessions[
            train_sessions[DATE] > train_limit_date
        ]
        print(
            f"Number of val_test sessions: {local_val_test_sessions[SESS_ID].nunique()}"  # type: ignore
        )

        # split the val_test sessions in half to constitute local validation and test set
        unique_val_test_sessions_ids = local_val_test_sessions[SESS_ID].unique()  # type: ignore

        # set the numpy random seed for reporducibility
        np.random.seed(RANDOM_SEED)
        # shuffle the sessions ids
        np.random.shuffle(unique_val_test_sessions_ids)
        num_sessions = len(unique_val_test_sessions_ids)

        val_len = math.ceil(num_sessions * 0.5)

        val_sess_ids, test_sess_ids = (
            unique_val_test_sessions_ids[:val_len],
            unique_val_test_sessions_ids[val_len:],
        )

        # train data should NOT be in val test sess ids
        train_data, train_label = (
            train_sessions[~train_sessions[SESS_ID].isin(unique_val_test_sessions_ids)],
            train_purchases[
                ~train_purchases[SESS_ID].isin(unique_val_test_sessions_ids)
            ],
        )
        val_data, val_label = (
            train_sessions[train_sessions[SESS_ID].isin(val_sess_ids)],
            train_purchases[train_purchases[SESS_ID].isin(val_sess_ids)],
        )
        test_data, test_label = (
            train_sessions[train_sessions[SESS_ID].isin(test_sess_ids)],
            train_purchases[train_purchases[SESS_ID].isin(test_sess_ids)],
        )

        assert len(train_data[SESS_ID].unique()) == len(  # type: ignore
            train_label[SESS_ID].unique()  # type: ignore
        ), "train data and label have different number of sessions!"
        assert len(val_data[SESS_ID].unique()) == len(
            val_data[SESS_ID].unique()
        ), "val data and label have different number of sessions!"
        assert len(test_data[SESS_ID].unique()) == len(
            test_data[SESS_ID].unique()
        ), "test data and label have different number of sessions!"

        print(f"Train sessions: {len(train_data[SESS_ID].unique())}")  # type: ignore
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

    def create_sess2items_list_dict(self) -> None:
        """Create a dictionary lookup mapping sessid the list of item in the session"""
        train_data = self.get_train_sessions()
        lead_data = self.get_test_leaderboard_sessions()
        test_data = self.get_test_final_sessions()
        all_data = pd.concat([train_data, lead_data, test_data], axis=0)
        all_data = all_data.sort_values([SESS_ID, DATE])

        sess2item_df = all_data.groupby(SESS_ID)[ITEM_ID].apply(list)
        sess2item_df.to_pickle(self.get_preprocessed_data_path() / "sess2items.pkl")
        print("- sess2item df saved!")

    def preprocess_item_features_oh(self) -> None:
        """Create and save one-hot features for the items"""
        item_features = self.get_item_features()
        oh_cat = pd.get_dummies(item_features[F_CAT], prefix="cat")
        oh_val = pd.get_dummies(item_features[F_VAL], prefix="val")
        item_features_oh = item_features.join(oh_cat).join(oh_val)
        item_features_oh = item_features_oh.groupby(ITEM_ID).sum()
        item_features_oh = item_features_oh.drop([F_VAL, F_CAT], axis=1).reset_index()
        # Save oh features
        item_features_oh.to_feather(
            self.get_preprocessed_data_path() / "oh_item_features.feather"  # type: ignore
        )
        print("- One Hot features Saved succesfully!")

    def create_local_candidate_items(self):
        """Create local candidate items, items which have been purchased in the last month data"""
        split_dict = self.get_split()
        val, val_label = split_dict[VAL]
        test, test_label = split_dict[TEST]
        local_candidates = pd.concat([val_label, test_label])[
            [ITEM_ID]
        ].drop_duplicates()
        local_candidates.reset_index(drop=True).to_feather(
            self.get_preprocessed_data_path() / "local_candidates_items.feather"  # type: ignore
        )
        print("- Local candidates items saved succefully succesfully!")

    def create_sess_features(self) -> None:
        sess2items = self.get_sess2items()
        temp = sess2items.explode().to_frame().reset_index()
        features = self.get_oh_item_features()
        joined = pd.merge(temp, features.reset_index(), on=ITEM_ID)
        user_f = joined.groupby(SESS_ID).sum()
        user_f = user_f.add_suffix("_sum")

        sess_len = temp.groupby(SESS_ID).count()
        sess_len = sess_len.rename(columns={ITEM_ID: "session_length"})

        final_sess_features = user_f.join(sess_len).reset_index()

        final_sess_features.reset_index(drop=True).to_feather(
            self.get_preprocessed_data_path() / "sess_features.feather"  # type: ignore
        )
        print("- Session Features created succesfully!")

    ##########################################
    ##### GET PREPROCESSED DATA METHODS ######
    ##########################################

    def get_recs_df(self, model_name: str, kind: str) -> pd.DataFrame:
        base_path = None
        if kind == "train":
            base_path = self.get_train_recs_df_folder()
        elif kind == "leaderboard":
            base_path = self.get_leaderboard_recs_df_folder()
        elif kind == "final":
            base_path = self.get_final_recs_df_folder()
        else:
            raise NotImplementedError("Kind passed is wrong!")
        m = model_name + ".feather"
        recs = pd.read_feather(base_path / m)
        recs = recs.rename(
            columns={"score": f"{model_name}_score", "rank": f"{model_name}_rank"}
        )
        return recs

    def get_sess_features(self) -> pd.DataFrame:
        path = self.get_preprocessed_data_path() / Path("sess_features.feather")
        df = pd.read_feather(path)
        return df

    def get_oh_item_features(self) -> pd.DataFrame:
        path = self.get_preprocessed_data_path() / Path("oh_item_features.feather")
        df = pd.read_feather(path)
        df = df.set_index(ITEM_ID)
        return df

    def get_sess2items(self) -> pd.DataFrame:
        path = self.get_preprocessed_data_path() / Path("sess2items.pkl")
        df = pd.read_pickle(path)
        return df

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
        df[DATE] = pd.to_datetime(df[DATE])
        return df

    def get_test_final_sessions(self) -> pd.DataFrame:
        path = self.get_preprocessed_data_path() / Path("test_final_sessions.feather")
        df = pd.read_feather(path)
        df[DATE] = pd.to_datetime(df[DATE])
        return df

    def get_candidate_items(self) -> pd.DataFrame:
        path = self.get_preprocessed_data_path() / Path("candidate_items.feather")
        df = pd.read_feather(path)
        return df

    def get_item_features(self) -> pd.DataFrame:
        path = self.get_preprocessed_data_path() / Path("item_features.feather")
        df = pd.read_feather(path)
        return df

    def get_local_candidate_items(self) -> pd.DataFrame:
        path = self.get_preprocessed_data_path() / Path(
            "local_candidates_items.feather"
        )
        df = pd.read_feather(path)
        return df

    ##########################################
    ########### Submission Handler ###########
    ##########################################

    def create_submission(self, recs_df: pd.DataFrame, sub_name: str) -> None:
        """Create a submission"""
        # dd/mm/YY H:M:S
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y__%H_%M_%S__")

        COLS = [SESS_ID, ITEM_ID, "rank"]
        assert all(
            [c in recs_df.columns for c in COLS]
        ), f"Missing one of the mandatory cols: {COLS}"
        # retrieve mapping dicts and map item id back to raw
        _, new_raw_md = self.get_item_mapping_dicts()
        recs_df[ITEM_ID] = recs_df[ITEM_ID].map(new_raw_md.get)
        recs_df = recs_df[[SESS_ID, ITEM_ID, "rank"]]
        for c in recs_df.dtypes:
            assert c == int, "Nan on sub probably!"
        recs_df.to_csv(
            str(self.get_submission_folder()) + "/" + sub_name + ".csv",
            index=False,
        )
        print(f"Submission with name: {sub_name} created succesfully!")


if __name__ == "__main__":
    dataset = Dataset()
    # dataset.create_sess_features()
    # print(dataset.get_candidate_items())
    # dataset.preprocess_data()
    # dataset.split_data()
    # dataset.preprocess_item_features_oh()
    # dataset.create_sess2items_list_dict()
    # df = dataset.get_sess2items()
    # print(df[4440001])
