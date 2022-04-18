import argparse
import time
from ast import arg

import torch
from evaluation import compute_mrr
from models.taigcn.taigcn import TAIGCN, WeightedSumSessEmbedding
from src.constant import *
from src.datasets.dataset import Dataset
from src.utils.pytorch.datasets import TripletsBPRDataset
from torch.utils import mkldnn as mkldnn_utils
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from utils.decorator import timing
from utils.pytorch.losses import bpr_loss


# define the training procedure
def train_routine(batch):
    model.train()
    optimizer.zero_grad()
    # optimization forward only the embeddings of the batch
    # compute embeddings
    x_u, item_embeddings = model(batch)
    x_i = torch.index_select(item_embeddings, 0, batch[1])  # type: ignore
    x_j = torch.index_select(item_embeddings, 0, batch[2])  # type: ignore
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

    # wandb log validation metric
    # if args["wandb"]:
    #     res_dict = val_evaluator.result_dict
    #     wandb.log(res_dict, step=epoch)
    # if args["verbose"]:
    #     val_evaluator.print_result_dict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("train TAIGCN")

    # model parameters
    parser.add_argument("--convolution_depth", type=list, default=[256, 128])
    parser.add_argument("--embedding_dimension", type=int, default=64)
    parser.add_argument("--features_num", type=int, default=968)
    parser.add_argument("--normalize_propagation", type=bool, default=True)

    # train parameters
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--val_every", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--l2_reg", type=float, default=1e-5)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    # GPU config
    parser.add_argument("--gpu", type=bool, default=True)
    parser.add_argument("--gpu_number", type=str, default="2")

    # get variables
    args = vars(parser.parse_args())

    dataset = Dataset()
    split_dict = dataset.get_split()

    train, train_label = split_dict[TRAIN]
    val, val_label = split_dict[VAL]
    test, test_label = split_dict[TEST]

    full_data = dataset.get_train_sessions()

    # put the model on the correct device
    device = torch.device(
        "cuda:{}".format(args["gpu_number"])
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
        normalize_propagation=args["normalize_propagation"],
        device=device,
    )

    # setting up the pytorch dataset
    train_dataset = TripletsBPRDataset(train, dataset)
    rnd_sampler = RandomSampler(
        train_dataset, replacement=True, num_samples=len(train_dataset)
    )
    print("Number of training samples: {}".format(len(train_dataset)))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        num_workers=32,
        sampler=rnd_sampler,
    )

    model = model.to(device)

    # initialize the optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=args["learning_rate"], weight_decay=args["l2_reg"]
    )

    for epoch in range(1, args["epochs"]):
        cum_loss = 0
        current_batch = 0
        t1 = time.time()
        for batch in tqdm(train_dataloader):
            # move the batch to the correct device
            batch = [b.to(device) for b in batch]
            loss = train_routine(batch)

            cum_loss += loss
            current_batch += 1
            if current_batch % 5 == 0:
                cum_loss /= current_batch
                log = "Batch: {:03d}, Loss: {:.4f}, Time: {:.4f}s"
                print(log.format(current_batch, cum_loss, time.time() - t1))
            if current_batch % args["val_every"] == 0:
                with torch.no_grad():
                    validation_routine()

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
