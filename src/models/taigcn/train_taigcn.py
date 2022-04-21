import argparse
import time
from ast import arg

import numpy as np
import pandas as pd
import torch
import wandb
from src.constant import *
from src.datasets.dataset import Dataset
from src.evaluation import compute_mrr
from src.models.taigcn.taigcn import TAIGCN, WeightedSumSessEmbedding
from src.utils.decorator import timing
from src.utils.general import pick_gpu_lowest_memory
from src.utils.pytorch.datasets import TripletsBPRDataset
from src.utils.pytorch.losses import bpr_loss
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
    x_i = torch.index_select(item_embeddings, 0, batch[4])  # type: ignore
    x_j = torch.index_select(item_embeddings, 0, batch[5])  # type: ignore
    loss = bpr_loss(x_u, x_i, x_j)

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
    # wandb log validation metric
    # if args["wandb"]:
    #     res_dict = val_evaluator.result_dict
    #     wandb.log(res_dict, step=epoch)
    # if args["verbose"]:
    #     val_evaluator.print_result_dict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("train TAIGCN")

    # model parameters
    parser.add_argument("--convolution_depth", type=int, default=2)
    parser.add_argument("--features_layer", type=list, default=[1024, 512])
    parser.add_argument("--embedding_dimension", type=int, default=64)
    parser.add_argument("--features_num", type=int, default=968)
    parser.add_argument("--normalize_propagation", type=bool, default=False)

    # train parameters
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--val_every", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--l2_reg", type=float, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    # GPU config
    parser.add_argument("--gpu", type=bool, default=True)

    # get variables
    args = vars(parser.parse_args())

    # initialize wandb
    wandb.init(config=args)

    dataset = Dataset()
    split_dict = dataset.get_split()

    train, train_label = split_dict[TRAIN]
    val, val_label = split_dict[VAL]
    test, test_label = split_dict[TEST]

    # full_data = dataset.get_train_sessions()
    full_data = dataset.get_train_sessions()
    full_label = dataset.get_train_purchases()

    # full_data = pd.concat([full_data, full_label], axis=0)

    # put the model on the correct device
    device = torch.device(
        "cuda:{}".format(pick_gpu_lowest_memory())
        if (torch.cuda.is_available() and args["gpu"])
        else "cpu"
    )
    # device = "cpu"
    print("Using device {}".format(device))

    # initialize session embedding module
    sess_emb_module = WeightedSumSessEmbedding(full_data, dataset, device)

    # initialize the model
    model = TAIGCN(
        dataset,
        sess_embedding_module=sess_emb_module,
        convolution_depth=args["convolution_depth"],
        features_layer=args["features_layer"],
        normalize_propagation=args["normalize_propagation"],
        device=device,
    )

    # setting up the pytorch dataset
    train_dataset = TripletsBPRDataset(train_label, full_data, dataset)
    rnd_sampler = RandomSampler(
        train_dataset, replacement=True, num_samples=len(train_dataset)
    )
    print("Number of training samples: {}".format(len(train_dataset)))

    # filter on timescore
    # full_data["last_buy"] = full_data.groupby(SESS_ID)[DATE].transform(max)
    # full_data["first_buy"] = full_data.groupby(SESS_ID)[DATE].transform(min)
    # full_data["time_score"] = 1 / (
    #     (
    #         (full_data["last_buy"] - full_data[DATE]).apply(
    #             lambda x: x.total_seconds() / 3600
    #         )
    #     )
    #     + 1
    # )
    # full_data = full_data[full_data["time_score"] >= 0.9]
    full_data_dict = full_data.groupby(SESS_ID)[ITEM_ID].apply(list)

    def collate_fn(data):
        user_ids, item_i, item_j = zip(*data)
        col_idx = torch.cat(
            [torch.tensor(full_data_dict[u], dtype=torch.int64) for u in user_ids]
        )
        row_idx = torch.cat(
            [
                torch.tensor(np.repeat(idx, len(full_data_dict[u])), dtype=torch.int64)
                for u, idx in zip(user_ids, range(len(user_ids)))
            ]
        )
        data_tensor = torch.cat(
            [
                torch.tensor(
                    np.repeat(1 / len(full_data_dict[u]), len(full_data_dict[u])),
                    dtype=torch.float32,
                )
                for u in user_ids
            ]
        )

        # row_idx = torch.cat([np.arange(len(x)) for x in col_idx], dtype=torch.int64)
        # data_tensor = torch.tensor(np.ones(len(col_idx)), dtype=torch.float32)

        return (
            row_idx,
            col_idx,
            data_tensor,
            torch.tensor(len(user_ids)),
            torch.tensor(item_i),
            torch.tensor(item_j),
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        num_workers=72,
        sampler=rnd_sampler,
        collate_fn=collate_fn,
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

        # wandb log loss every epoch
        # if args["wandb"]:
        #     wandb.log({"loss": cum_loss}, step=epoch)
        # log = "Epoch: {:03d}, Loss: {:.4f}, Time: {:.4f}s"
        # if args["verbose"]:
        #     print(log.format(epoch, cum_loss, time.time() - t1))
        # if epoch % args["val_every"] == 0:
        #     with torch.no_grad():
        #         validation_routine()
        #     if args["early_stopping"]:
        #         es_metric = val_evaluator.result_dict[args["es_metric"]]
        #         es_handler.update(epoch, es_metric, args["es_metric"], model)
        #         if es_handler.stop_training():
        #             break
