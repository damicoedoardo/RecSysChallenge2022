import argparse
import time
from ast import arg

import pytorch_lightning as pl
import torch
from evaluation import compute_mrr
from models.taigcn.lightning_taigcn import LightTAIGCN, WeightedSumSessEmbedding
from models.taigcn.taigcn import TAIGCN
from src.constant import *
from src.datasets.dataset import Dataset
from src.utils.pytorch.datasets import TripletsBPRDataset
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser("train TAIGCN")

    # model parameters
    parser.add_argument("--convolution_depth", type=list, default=[256, 128])
    parser.add_argument("--embedding_dimension", type=int, default=64)
    parser.add_argument("--features_num", type=int, default=968)
    parser.add_argument("--normalize_propagation", type=bool, default=False)

    # train parameters
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--val_every", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--l2_reg", type=float, default=0)
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

    # initialize session embedding module
    sess_emb_module = WeightedSumSessEmbedding(full_data, dataset)

    # initialize the model
    model = LightTAIGCN(
        dataset,
        sess_embedding_module=sess_emb_module,
        convolution_depth=args["convolution_depth"],
        normalize_propagation=args["normalize_propagation"],
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

    trainer = pl.Trainer(accelerator="gpu", devices=4)
    trainer.fit(model=model, train_dataloaders=train_dataloader)
