import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from src.constant import *
from src.datasets.dataset import Dataset
from src.evaluation import compute_mrr
from src.models.cassandra.cassandra import Cassandra
from src.models.cassandra.session_embedding_modules import (
    ContextAttention,
    GRUSessionEmbedding,
    MeanAggregatorSessionEmbedding,
    NNFeaturesEmbeddingModule,
    SelfAttentionSessionEmbedding,
)
from src.models.noname.noname import NoName, WeightedSumSessEmbedding
from src.utils.decorator import timing
from src.utils.general import pick_gpu_lowest_memory
from src.utils.pytorch.datasets import (
    CCLDataset,
    ContextAwarePaddedDataset,
    TripletsBPRDataset,
)
from src.utils.pytorch.losses import CosineContrastiveLoss, bpr_loss
from torch.utils import mkldnn as mkldnn_utils
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser("train NoName")

    # model parameters
    parser.add_argument("--session_embedding_kind", type=str, default="context_attn")
    parser.add_argument("--layers_size", type=list, default=[512, 128])
    parser.add_argument("--embedding_dimension", type=int, default=128)
    parser.add_argument("--features_num", type=int, default=963)
    parser.add_argument("--k", type=int, default=10)

    # train parameters
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--l2_reg", type=float, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    # loss function parameter
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--negative_weight", type=int, default=0.5)
    parser.add_argument("--negative_samples_num", type=int, default=1000)

    # GPU config
    parser.add_argument("--gpu", type=bool, default=True)

    # SEASONALITY DATA trimming
    parser.add_argument("--days_to_keep", type=int, default=150)

    # TRAIN for final prediction
    parser.add_argument("--train_valtest", type=bool, default=True)
    parser.add_argument("--model_save_name", type=str, default="gentle-firefly-5493")

    # get variables
    args = vars(parser.parse_args())

    print("Loading splits...")
    dataset = Dataset()
    split_dict = dataset.get_split()

    train, train_label = split_dict[TRAIN]
    val, val_label = split_dict[VAL]
    test, test_label = split_dict[TEST]

    # Considering seasonality by training using the sessions only on the last x days of the training set available
    print(
        "Using the last: {} days sessions to train the model".format(
            args["days_to_keep"]
        )
    )
    max_date = train[DATE].max()
    train_limit_date = max_date - timedelta(days=150)
    filtered_train = train[train[DATE] > train_limit_date].copy()
    id_filtered_train = filtered_train[SESS_ID].unique()
    final_train_data = train[train[SESS_ID].isin(id_filtered_train)]
    final_train_label = train_label[train_label[SESS_ID].isin(id_filtered_train)]

    # If the model has to be trained for the prediction train val and test has to be merged together

    if args["train_valtest"]:
        print("Training model for final predictions with the whole data available...")
        args["val_every"] = -1
        final_train_data = pd.concat([final_train_data, val, test])
        final_train_label = pd.concat([final_train_label, val_label, test_label])

    # put the model on the correct device
    # device = torch.device(
    #     "cuda:{}".format(pick_gpu_lowest_memory())
    #     if (torch.cuda.is_available() and args["gpu"])
    #     else "cpu"
    # )
    device = torch.device("cuda:1") if args["gpu"] else torch.device("cpu")
    # device = "cpu"
    print("Using device {}".format(device))

    # initialize loss function
    loss = CosineContrastiveLoss(
        margin=args["margin"], negative_weight=args["negative_weight"]
    )

    # setting up the pytorch dataset
    train_dataset = ContextAwarePaddedDataset(
        dataset=dataset,
        train_df=final_train_data,
        purchase_df=final_train_label,
        negative_samples_num=args["negative_samples_num"],
        padding_idx=dataset._ITEMS_NUM,
        k=args["k"],
    )

    rnd_sampler = RandomSampler(
        train_dataset, replacement=False, num_samples=len(train_dataset)
    )

    print("Number of training samples: {}".format(len(train_dataset)))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        num_workers=72,
        sampler=rnd_sampler,
        # collate_fn=collate_fn,
    )

    # choose item features embedding module
    item_features_embedding_module = NNFeaturesEmbeddingModule(
        layers_size=args["layers_size"]
    )

    # choose session embedding module
    concat_item_dim = args["embedding_dimension"] + args["layers_size"][-1]
    # concat_item_dim = args["embedding_dimension"]
    session_embedding_module = None
    if args["session_embedding_kind"] == "mean":
        session_embedding_module = MeanAggregatorSessionEmbedding()
    elif args["session_embedding_kind"] == "gru":
        session_embedding_module = GRUSessionEmbedding(
            input_size=concat_item_dim,
            hidden_size=concat_item_dim,
            num_layers=1,
        )
    elif args["session_embedding_kind"] == "attn":
        session_embedding_module = SelfAttentionSessionEmbedding(
            input_size=concat_item_dim, num_heads=1
        )
    elif args["session_embedding_kind"] == "context_attn":
        session_embedding_module = ContextAttention(
            input_size=concat_item_dim,
            hidden_size=concat_item_dim,
            num_layers=1,
        )
    else:
        raise NotImplementedError(
            "Aggregator {} not implemented!".format(args["session_embedding_kind"])
        )

    # initialize the model
    model = Cassandra(
        dataset,
        loss_function=loss,
        item_features_embedding_module=item_features_embedding_module,
        session_embedding_module=session_embedding_module,
        embedding_dimension=args["embedding_dimension"],
        device=device,
        train_dataset=train_dataset,
    )

    load_name = Path(args["model_save_name"] + ".pth")
    model = model.to(device)
    model.load_state_dict(torch.load(dataset.get_saved_models_path() / load_name))
    print("Loaded model with name: {}".format(args["model_save_name"]))
    model.eval()
    lead_data = dataset.get_test_leaderboard_sessions()
    final_data = dataset.get_test_final_sessions()

    recs = model.recommend(
        interactions=lead_data, remove_seen=True, cutoff=100, leaderboard=True
    )
    dataset.create_submission(recs, sub_name=args["model_save_name"].split(".pth")[0])
    print("Submission created succesfully!")
