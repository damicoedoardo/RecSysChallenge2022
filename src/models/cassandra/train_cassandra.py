import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from models.cassandra.cassandra import Cassandra
from models.noname.noname import NoName, WeightedSumSessEmbedding
from src.constant import *
from src.datasets.dataset import Dataset
from src.evaluation import compute_mrr
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


# define the training procedure
def train_routine(batch):
    model.train()
    optimizer.zero_grad()
    # optimization forward only the embeddings of the batch
    # compute embeddings
    x_u, item_embeddings = model(batch)

    _, purchase_items, negative_items = batch[0], batch[1], batch[2]
    x_i, x_j = item_embeddings(purchase_items), item_embeddings(negative_items)

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
        interactions=val, remove_seen=True, cutoff=100, leaderboard=True
    )
    print("Computing mrr...")
    mrr = compute_mrr(recs, val_label)
    return mrr


if __name__ == "__main__":
    parser = argparse.ArgumentParser("train NoName")

    # model parameters
    parser.add_argument("--session_embedding_kind", type=str, default="mean")
    parser.add_argument("--features_layer", type=list, default=[968, 256])
    parser.add_argument("--embedding_dimension", type=int, default=512)
    parser.add_argument("--features_num", type=int, default=968)
    parser.add_argument("--k", type=int, default=10)

    # train parameters
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--l2_reg", type=float, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    # loss function parameter
    parser.add_argument("--margin", type=float, default=0.7)
    parser.add_argument("--negative_weight", type=int, default=0.5)
    parser.add_argument("--negative_samples_num", type=int, default=500)

    # GPU config
    parser.add_argument("--gpu", type=bool, default=True)

    # get variables
    args = vars(parser.parse_args())

    # initialize wandb
    wandb.init(config=args)

    print("Loading splits...")
    dataset = Dataset()
    split_dict = dataset.get_split()

    train, train_label = split_dict[TRAIN]
    val, val_label = split_dict[VAL]
    test, test_label = split_dict[TEST]

    # put the model on the correct device
    device = torch.device(
        "cuda:{}".format(pick_gpu_lowest_memory())
        if (torch.cuda.is_available() and args["gpu"])
        else "cpu"
    )
    # device = "cpu"
    print("Using device {}".format(device))

    # initialize loss function
    loss = CosineContrastiveLoss(
        margin=args["margin"], negative_weight=args["negative_weight"]
    )

    # setting up the pytorch dataset
    train_dataset = ContextAwarePaddedDataset(
        dataset=dataset,
        train_df=train,
        purchase_df=train_label,
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

    # initialize the model
    model = Cassandra(
        dataset,
        loss_function=loss,
        session_embedding_kind=args["session_embedding_kind"],
        embedding_dimension=args["embedding_dimension"],
        features_layer=args["features_layer"],
        device=device,
        train_dataset=train_dataset,
    )

    model = model.to(device)

    # initialize the optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=args["learning_rate"], weight_decay=args["l2_reg"]
    )

    for epoch in range(1, args["epochs"]):
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

        if epoch % args["val_every"] == 0:
            with torch.no_grad():
                mrr = validation_routine()
                wandb.log({"mrr": mrr}, step=epoch)
