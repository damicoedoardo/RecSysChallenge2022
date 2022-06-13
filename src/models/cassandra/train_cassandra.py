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
    ContextAwareContrastiveDataset,
    ContextAwarePaddedDataset,
    TripletsBPRDataset,
)
from src.utils.pytorch.losses import (
    ContextAwareCosineContrastiveLoss,
    CosineContrastiveLoss,
    bpr_loss,
)
from torch.utils import mkldnn as mkldnn_utils
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm


# define the training procedure
def train_routine(batch):
    model.train()
    optimizer.zero_grad()
    # optimization forward only the embeddings of the batch
    # compute embeddings
    x_u, item_embeddings = model(batch)

    _, purchase_items, negative_items = (
        batch[0],
        batch[1],
        batch[2],
    )
    x_i, x_j = (
        item_embeddings[purchase_items],
        item_embeddings[negative_items],
    )

    # x_j = torch.mean(x_j, dim=1)
    # loss = bpr_loss(x_u, x_i, x_j)

    x_u = F.normalize(x_u, dim=-1)
    x_i = F.normalize(x_i, dim=-1)
    x_j = F.normalize(x_j, dim=-1)

    loss = model.loss_function(x_u, x_i, x_j)
    # loss = model(batch)
    loss.backward()
    optimizer.step()
    return loss


@timing
def validation_routine():
    print("+++ Validation routine +++")
    model.eval()
    print("Computing representations...")
    model.compute_representations(val)
    print("Recommend...")
    recs = model.recommend(
        interactions=val, remove_seen=True, cutoff=100, leaderboard=False
    )
    print("Computing mrr...")
    mrr = compute_mrr(recs, val_label)
    return mrr


if __name__ == "__main__":
    parser = argparse.ArgumentParser("train NoName")

    # model parameters
    parser.add_argument("--session_embedding_kind", type=str, default="context_attn")
    parser.add_argument("--layers_size", type=list, default=[963, 256])
    parser.add_argument("--embedding_dimension", type=int, default=256)
    parser.add_argument("--features_num", type=int, default=963)
    parser.add_argument("--k", type=int, default=10)

    # train parameters
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--l2_reg", type=float, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--early_stopping_round", type=int, default=5)

    # loss function parameter
    parser.add_argument("--margin", type=float, default=0.8)
    parser.add_argument("--negative_weight", type=int, default=0.5)
    parser.add_argument("--negative_samples_num", type=int, default=1000)
    parser.add_argument("--context_weight", type=int, default=0.1)
    parser.add_argument("--context_samples_num", type=int, default=2)

    # GPU config
    parser.add_argument("--gpu", type=bool, default=True)

    # SEASONALITY DATA trimming
    parser.add_argument("--days_to_keep", type=int, default=150)

    # TRAIN for final prediction
    parser.add_argument("--train_valtest", type=bool, default=True)

    # get variables
    args = vars(parser.parse_args())

    # initialize wandb
    wandb.init(config=args)
    # get the run name
    run_name = wandb.run.name
    print(f"Run name: {run_name}")

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
    train_limit_date = max_date - timedelta(days=args["days_to_keep"])
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

    print("Using device {}".format(device))

    # initialize loss function
    loss = CosineContrastiveLoss(
        margin=args["margin"],
        negative_weight=args["negative_weight"],
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
    # concat_item_dim = args["embedding_dimension"] + args["layers_size"][-1]
    concat_item_dim = args["embedding_dimension"]
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
        session_embedding_module=session_embedding_module,
        item_features_embedding_module=item_features_embedding_module,
        embedding_dimension=args["embedding_dimension"],
        device=device,
        train_dataset=train_dataset,
    )

    model = model.to(device)

    # initialize the optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=args["learning_rate"], weight_decay=args["l2_reg"]
    )

    best_mrr = 0
    early_stopping_counter = 0
    for epoch in range(1, args["epochs"]):
        if early_stopping_counter == args["early_stopping_round"]:
            print("Early stopping ended training procedure!")
            break
        cum_loss = 0
        t1 = time.time()
        for batch in tqdm(train_dataloader):
            # move the batch to the correct device
            batch = [b.to(device) for b in batch]
            loss = train_routine(batch)
            cum_loss += loss

        cum_loss /= len(train_dataloader)
        log = "Epoch: {:03d}, Loss: {:.4f}, Time: {:.4f}s"
        print(log.format(epoch, cum_loss, time.time() - t1))
        wandb.log({"BPR loss": cum_loss}, step=epoch)

        if (epoch % args["val_every"] == 0) and (args["val_every"] != -1):
            with torch.no_grad():
                mrr = validation_routine()
                wandb.log({"mrr": mrr}, step=epoch)
                if mrr > best_mrr:
                    best_mrr = mrr
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

    save_path = Path(run_name + ".pth")
    if args["train_valtest"]:
        torch.save(
            model.state_dict(),
            dataset.get_saved_models_path() / save_path,
        )
        print("Trained model with name: {}, saved succesfully".format(run_name))
