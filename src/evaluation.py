import numpy as np
import pandas as pd

from src.constant import *


def compute_mrr(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    col_user=SESS_ID,
    col_item=ITEM_ID,
) -> float:
    merged = pd.merge(predictions, ground_truth, on=[col_user, col_item], how="right")
    merged["mrr"] = 1 / merged["rank"]
    merged["mrr"] = merged["mrr"].fillna(0)
    mrr = merged["mrr"].mean()
    print(f"MRR: {mrr}")
    return mrr  # type: ignore
