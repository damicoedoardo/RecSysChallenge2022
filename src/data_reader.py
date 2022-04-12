import logging
import os
import pickle
from math import lcm
from mimetypes import init
from operator import index
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class DataReader:

    _DATA_PATH = Path(Path.home() / os.environ.get("DATA_PATH"))  # type: ignore
    _PREPROCESSED_DATA_PATH = Path(_DATA_PATH / "preprocessed")

    # Path to raw files
    _TRAIN_SESSION = "train_sessions.csv"
    _TRAIN_PURCHASES = "train_purchases.csv"
    _TEST_LEADEARBOARD_SESSIONS = "test_leaderboard_sessions.csv"
    _TEST_FINAL_SESSIONS = "test_final_sessions.csv"
    _CANDIDATE_ITEMS = "candidate_items.csv"
    _ITEM_FEATURES = "item_features.csv"

    def __init__(self):
        pass

    def get_data_path(self) -> Path:
        return self._DATA_PATH

    def get_train_sessions(self) -> pd.DataFrame:
        path = self.get_data_path() / self._TRAIN_SESSION
        df = pd.read_csv(path)
        return df

    def get_train_purchases(self) -> pd.DataFrame:
        path = self.get_data_path() / self._TRAIN_PURCHASES
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def get_test_leaderboard_sessions(self) -> pd.DataFrame:
        path = self.get_data_path() / self._TEST_LEADEARBOARD_SESSIONS
        df = pd.read_csv(path)
        return df

    def get_test_final_sessions(self) -> pd.DataFrame:
        path = self.get_data_path() / self._TEST_FINAL_SESSIONS
        df = pd.read_csv(path)
        return df

    def get_candidate_items(self) -> pd.DataFrame:
        path = self.get_data_path() / self._CANDIDATE_ITEMS
        df = pd.read_csv(path)
        return df

    def get_item_features(self) -> pd.DataFrame:
        path = self.get_data_path() / self._ITEM_FEATURES
        df = pd.read_csv(path)
        return df
