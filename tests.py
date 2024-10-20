from typing import List, NoReturn, Dict
import numpy as np
import networkx as nx
from nilearn import plotting
import pandas as pd
import nilearn.datasets as datasets

from collections import defaultdict

from pre_process import get_anatomical_node_labels
from utils import load_graphs_features, get_y_true_regression, get_y_true
from visualization import build_features_for_scatters, scatter_plot, hist_class




def plot_feature(feature_name: str) -> NoReturn:
    features = build_features_for_scatters('threshold', np.arange(start=0.42, stop=0.46, step=0.01),
                                feature_name, get_y_true())

    scatter_plot(features, feature_name)
    hist_class(features, feature_name)

if __name__ == '__main__':
    # nodes = get_anatomical_node_labels()
    # nodes = get_anatomical_node_labels()
    power_atlas = datasets.fetch_coords_power_2011()
    # coords = np.vstack((power_atlas.rois['x'], power_atlas.rois['y'], power_atlas.rois['z'])).T
    example = datasets.fetch_atlas_yeo_2011()['thick_17']

    coorinates = plotting.find_parcellation_cut_coords(labels_img=example)
    plot_feature('rich_club_coefficient_24')
    plot_feature('rich_club_coefficient_25')
    plot_feature('rich_club_coefficient_26')
    plot_feature('rich_club_coefficient_27')
    plot_feature('average_neighbor_degree_88')
    plot_feature('pagerank_numpy_variance')
    plot_feature('average_neighbor_degree_2')