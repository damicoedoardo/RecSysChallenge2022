import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from src.utils.logger import set_color

logger = logging.getLogger(__name__)


def remap_column_consecutive(df:pd.DataFrame, column_name: str) -> Tuple[pd.DataFrame, Dict]:
    """Remap selected column of a given dataframe into consecutive numbers

    Args:
        df (pd.DataFrame): dataframe to be mapped
        column_name (str): column of the dataframe to be mapped

    Returns:
        Tuple[pd.DataFrame, Dict]: remapped dataframe, mapping_dict
    """
    assert (
        column_name in df.columns.values
    ), f"Column name: {column_name} not in df.columns: {df.columns.values}"

    copy_df = df.copy()
    unique_data = copy_df[column_name].unique()
    logger.debug(set_color(f"unique {column_name}: {len(unique_data)}", "yellow"))
    data_idxs = np.arange(len(unique_data))
    data_idxs_map = dict(zip(unique_data, data_idxs))
    copy_df[column_name] = copy_df[column_name].map(data_idxs_map.get)
    return copy_df, data_idxs_map
