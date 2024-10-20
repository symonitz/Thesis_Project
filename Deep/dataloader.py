from torch_geometric.data import Data, Dataset, DataLoader, InMemoryDataset
import networkx as nx
from typing import List, Tuple
import numpy as np
from torch_geometric.utils import from_networkx
import pandas as pd
# from model import load_model
# from utils import get_data_path
# from conf_pack.configuration import *
# import os
from Deep.feature_extraction_deep import time_series_to_features, time_series_to_images, images_to_feature_vector, \
    time_series_to_basic_features
from conf_pack.paths import *
from pre_process import load_scans, build_graphs_from_corr


class GraphsDataset(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(GraphsDataset, self).__init__(root, transform, pre_transform)
        self.root = root
        self.csv = pd.read_csv(os.path.join(root, 'Data.csv'))
        self.filenames = self.csv['Subject']
        self.labels = self.csv['Class']




    def len(self):
        return len(self.csv)

    def get(self, idx: int) -> Tuple[Data, int]:
        full_path = os.path.join(self.root, 'nifti', f'{self.filenames[idx]}.nii')
        data_lst = load_scans([full_path], data_type='both', dir_path=self.root)
        activations = data_lst[0][1].swapaxes(0, 1)
        correlation = data_lst[0][0]
        graph = build_graphs_from_corr('density', [correlation], 0.01)[0]

        features = time_series_to_basic_features(activations)
        # features_from_activations = time_series_to_images(activations)
        # features = images_to_feature_vector(features_from_activations).detach().numpy()

        # features_from_activations = time_series_to_features(activations)
        features_to_nodes = dict(zip(range(len(features)),
                                                       features))
        nx.set_node_attributes(graph, features_to_nodes, 'activations')
        # Todo: understand why the label return is int type 64 and not int.
        return from_networkx(graph), int(self.labels[idx])


def nx_lst_to_dl(graphs: List[nx.Graph]) -> DataLoader:
    lst_torch_graphs = []
    for graph in graphs:
        torch_graph = from_networkx(graph)
        lst_torch_graphs.append(torch_graph)
    dl = DataLoader(lst_torch_graphs)
    return dl


# if __name__ == '__main__':
#     dataset = GraphsDataset(root=get_data_path())
#     model = load_model(num_feat=218, num_classes=2)
#     dl = DataLoader(dataset, batch_size=default_params.getint('batch_size'))
#     for graph, label, filename in dl:
#         model(graph)
#         print(f'label of {filename} is {label}')
