import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import similaripy
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv
from tqdm import tqdm

from src.constant import *
from src.data_reader import DataReader
from src.datasets.dataset import Dataset
from src.evaluation import compute_mrr
from src.models.itemknn.itemknn import ItemKNN
from src.utils.sparse_matrix import interactions_to_sparse_matrix


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(graph_dataset.num_node_features, 256)
        self.conv2 = SAGEConv(256, 128)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x


if __name__ == "__main__":
    torch.cuda.empty_cache()
    dataset = Dataset()
    split_dict = dataset.get_split()
    train, train_label = split_dict[TRAIN]
    sparse_interaction, user_mapping_dict, _ = interactions_to_sparse_matrix(
        train,
        items_num=dataset._ITEMS_NUM,
        users_num=None,
    )
    sim = similaripy.cosine(sparse_interaction.T, k=100)
    coo_sim = sim.tocoo()

    # we have to create the edges both ways
    start_node_edge = np.concatenate([coo_sim.row, coo_sim.col])
    arrival_node_edge = np.concatenate([coo_sim.col, coo_sim.row])

    # create the graph
    edge_index = torch.tensor(
        np.array([start_node_edge, arrival_node_edge]), dtype=torch.long
    )

    item_features = dataset.get_oh_item_features()
    x = torch.tensor(item_features.values, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    graph_dataset = data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCN().to(device)
    data = graph_dataset.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for x in tqdm(range(100)):
        out = model(data)
