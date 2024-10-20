import os
from typing import Dict, List, NoReturn, DefaultDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
import seaborn as sns

from utils import get_save_path, load_graphs_features, get_results_path

colors = ['black', 'gray', 'rosybrown', 'brown', 'darkred', 'red', 'mistyrose', 'salmon', 'sienna', 'sandybrown',
          'peru', 'bisque', 'gold', 'lightyellow', 'lawngreen', 'darkgreen', 'turquoise', 'teal', 'deepskyblue',
          'slategray', 'blue']


def box_plot(paths, col_name, task, y_col, criteria, title):
    plt.clf()
    df_res = pd.DataFrame()
    for path in paths:
        df = pd.read_csv(path)
        path_parts = path.split("/")

        df['experiment type'] = path[path.rfind('_') + 1:path.rfind('.')]
        df_res = pd.concat([df_res, df])

    fig = sns.boxplot(y=y_col, x='experiment type', data=df_res, showmeans=True)
    fig.set_title(title)
    # fig.set_xticklabels(fig.get_xticklabels(), fontsize=9)
    fig.set_xlabel('Experiment Type', fontsize=16)
    fig.set_ylabel(col_name, fontsize=16)
    fig.set_xticklabels(fig.get_xticklabels(), rotation=40, ha="right", fontsize=14)
    plt.tight_layout()
    plt.figure(figsize=(4, 2))
    fig.figure.savefig(f'box_plot_{criteria}_{task}_{y_col}.png')
    plt.close()
    plt.clf()


def build_features_for_scatters(filter_type: str, thresh_lst: List[float], col_name: str, y: np.ndarray) -> DefaultDict:
    res = defaultdict(dict)
    for thresh in thresh_lst:
        df = load_graphs_features(filter_type, thresh)
        df_relevant = np.zeros(len(df))
        if col_name in df.columns:
            df_relevant = df[col_name].values
        for i in range(len(df)):
            res[i]['values'] = res[i].get('values', []) + [df_relevant[i]]
            res[i]['target'] = res[i].get('target', []) + [y[i]]
    return res


def hist_class(features: Dict, feature_name: str) -> NoReturn:

    zeroes_vals, ones_vals = [], []
    for key, item in features.items():
        if 0 in features[key]['target']:
            zeroes_vals += features[key]['values']
        elif 1 in features[key]['target']:
            ones_vals += features[key]['values']

    min_val = min(min(zeroes_vals), min(ones_vals))
    max_val = max(max(zeroes_vals), max(ones_vals))

    plot_histogram(min_val, max_val, [zeroes_vals], 'Zeroes Distribution', 'feature value', 'counting values',
                   f'zeroes_dist_{feature_name}.png', ['r'], ['Zero Dist'])
    plt.clf()

    plot_histogram(min_val, max_val, [ones_vals], 'Ones Distribution', 'feature value', 'counting values',
               f'ones_dist_{feature_name}.png', ['b'], ['One Dist'])

    plt.clf()

    plot_histogram(min_val, max_val, [zeroes_vals, ones_vals], 'Combined Distribution', 'feature value',
                   'counting values',
                   f'combined_dist_{feature_name}.png', ['b', 'r'], ['Zero Dist', 'One Dist'])

    # plot_histogram(min_val, max_val, ones_vals, 'Ones Distribution', 'feature value', 'counting values',
    #                f'combined_dist_{feature_name}.png', 'b')
    plt.clf()


def plot_histogram(min_val: float, max_val: float, to_plot: List[List], title: str, x_label: str,
                   y_label: str, save_path: str, colors: List[str], labels: List[str]) -> NoReturn:
    for hist, label, color in zip(to_plot, labels, colors):
        plt.hist(hist, label=label, color=color)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xlim((min_val, max_val))
    plt.legend()
    plt.savefig(os.path.join(get_results_path(), save_path))


def scatter_plot(features: Dict, feature_name: str) -> NoReturn:

    colors_dict = {key: val for key, val in zip(features.keys(), colors)}

    for key, item in features.items():

        plt.scatter(x=features[key]['values'], y=features[key]['target'],
                    c=[colors_dict[key]] * len(features[key]['target']),  cmap='viridis')
        plt.xlabel('Values of feature')
        plt.ylabel('Target function')

    plt.savefig(os.path.join(get_results_path(), f'scatter_graph_{feature_name}.png'))
    plt.clf()

    for key in features.keys():
        features[key]['subject'] = [key] * len(features[key]['values'])
    df_res = pd.concat([pd.DataFrame(features[key]) for key in features.keys()])
    df_res.sort_values(by='target', inplace=True)
    # labels = [df_res[key]['target'][0] for key in features.keys()]    df_res['index'] = df_res.index
    fig = sns.boxplot(y='values', x='subject', data=df_res, showmeans=True, order=df_res['subject'].unique())
    # plt.legend(labels)
    plt.savefig(os.path.join(get_results_path(), f'box_plot.png'))
    plt.clf()
    return None
    for key, item in features.items():

        fig = sns.boxplot(y='values', x='target', data=features[key], showmeans=True)
        plt.savefig(os.path.join(get_results_path(), f'{key}.png'))
        plt.clf()
    # plt.savefig(os.path.join(get_results_path(), f'box_plot_{feature_name}.png'))

