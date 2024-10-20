import os
from collections import defaultdict

from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from typing import Union, Tuple
import pandas as pd
from sklearn.metrics import accuracy_score
from typing import List
import numpy as np
from sklearn import preprocessing
from torch import nn

from Deep.model import Net
from sklearn.feature_selection import mutual_info_classif, SelectKBest, VarianceThreshold
from multiprocessing import Process, Pool

from conf_pack.configuration import default_params, c
# from feature_extraction import features_by_type
# from main import get_graphs
# from pre_process import get_corr_lst
from utils import write_selected_features, load_graphs_features, get_results_path, get_names, get_y_true, by_task, \
    get_y_true_regression


class lso:
    def __init__(self,  names: List[str]):
        self.names = np.array(names)

    def split(self, df: pd.DataFrame):

        for subj_name in set(self.names):
            test_idx = np.where(self.names == subj_name)
            train_idx = np.where(self.names != subj_name)
            yield train_idx, test_idx


def train_lso(df1: pd.DataFrame, df2: pd.DataFrame, num_features: int):
    dict_df1 = df1.to_dict(orient=list)
    dict_df2 = df2.to_dict(orient=list)
    dict_df1.update(dict_df2)


def train_model_subject_out(df1: pd.DataFrame, y1: np.ndarray, df2: pd.DataFrame, y2: np.ndarray, num_features: int) ->\
        Tuple[float, RandomForestClassifier, List[str]]:
    # loo = lso(get_names())
    loo = LeaveOneOut()
    df1, df2 = df1.fillna(0), df2.fillna(0)
    df1, df2 = normalize_features(df1), normalize_features(df2)
    res = []

    for train_idx, test_idx in loo.split(df1):
        X_train1, X_test1 = df1.iloc[train_idx],  df1.iloc[test_idx]
        X_train2, X_test2 = df2.iloc[train_idx], df2.iloc[test_idx]
        y_train1, y_test1 = y1[train_idx], y1[test_idx]
        y_train2, y_test2 = y2[train_idx], y2[test_idx]
        df_train_new = pd.concat([X_train1, X_train2])
        y_train_new = np.concatenate((y_train1, y_train2))
        df_test_new = pd.concat([X_test1, X_test2])
        y_test_new = np.concatenate((y_test1, y_test2))
        df_train_new = df_train_new.fillna(0)
        df_test_new = df_test_new.fillna(0)
        res.append(train_model_iteration(df_train_new, y_train_new, df_test_new, y_test_new, num_features))
    return train_suffix(pd.concat([df1, df2]), num_features, res, np.concatenate((y1, y2)))


def train_model(df: pd.DataFrame, y: np.ndarray, num_features: int, relevant_names) -> \
        Tuple[float, RandomForestClassifier, List[str]]:
    loo = lso(relevant_names)
    df = df.fillna(0)
    df = normalize_features(df)
    args_lst, res = [], []

    df = df.reset_index(drop=True)
    for train_idx, test_idx in loo.split(df):

        X_train, X_test = df.iloc[train_idx], df.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        args_lst.append((X_train, y_train, X_test, y_test, num_features))
        res.append(train_model_iteration(X_train, y_train, X_test, y_test, num_features))
        # with Pool(1) as p:
    #     res = p.starmap(train_model_iteration, args_lst)
    return train_suffix(df, num_features, res, y)


def train_suffix(df, num_features, res, y):
    df = df.fillna(0)
    avg_acc = sum(res)
    model = load_model('rf')
    feat_names, feat_values = select_features(df, y, num_features)
    model.fit(df[feat_names], y)
    avg_acc /= len(y)
    print(avg_acc)
    return avg_acc, model, feat_names


def train_model_iteration(X_train: pd.DataFrame, y_train: np.ndarray,
                          X_test: pd.DataFrame, y_test: np.ndarray, num_features: int) -> float:
    model = load_model('rf')

    feat_names, feat_values = select_features(X_train, y_train, num_features)
    X_train, X_test = X_train[feat_names], X_test[feat_names]
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    # write_selected_features(prediction, y_test)
    return accuracy_score(prediction, y_test)


def predict_by_criterions(**kwargs) -> Tuple[float, float]:
    col_names = kwargs['col_names']
    df = kwargs['df']
    idx = kwargs['idx']
    y = kwargs['y']
    model = kwargs['model']

    df = df[col_names]
    df = df.fillna(0)
    df = normalize_features(df)
    df_relevant_features = df.iloc[idx]
    y_relevant = y[idx]
    prediction = model.predict(df_relevant_features)
    acc = accuracy_score(prediction, y_relevant)
    return acc, prediction


def predict_by_proba(model, X_test, thresh: float = 0.6):
    rf = RandomForestClassifier()
    rf.predict_proba(X_test)
    prob = model.predict_proba(X_test)[0]
    if prob > thresh:
        return 1
    elif prob < (1 - thresh):
        return 0
    else:
        return -1

        # Between 1-thresh to thresh


def load_model(model_type: str) -> Union[nn.Module, RandomForestClassifier, None]:
    if model_type == 'deep':
        model = Net()
    elif model_type == 'rf':
        model = RandomForestClassifier(n_estimators=300)
    else:
        raise ValueError('Invalid model_type as input')
    return model


def compute_inf_gain(X, feat_names, y):
    return mutual_info_classif(X[feat_names], y)


def select_features(x_train: pd.DataFrame, y_true: np.ndarray, num_features: int) -> Tuple[List[str], List[float]]:
    def inf_gain(X, y):
        return mutual_info_classif(X, y)

    if default_params.get('features_type') == 'globals':
        # vt = VarianceThreshold()
        # vt.fit(x_train)
        # x_train = x_train[x_train.columns[vt.get_support(indices=True)]]
        selector = SelectKBest(inf_gain, k=num_features).fit(x_train, y_true)
        mask = selector.get_support()
        values = mutual_info_classif(x_train, y_true)[mask]
        feature_names = x_train.columns[mask]
        return feature_names, values
    else:
        return list(x_train.columns), list(np.ones(len(x_train.columns)))


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    min_max = preprocessing.MinMaxScaler()
    df[df.columns] = min_max.fit_transform(df)
    return df


def clean_df(df: pd.DataFrame, feat_names: List[str]) -> pd.DataFrame:
    df = df.fillna(0)
    df = normalize_features(df)
    return df[feat_names]

def info_gain_all_features(df: pd.DataFrame, y_true: np.ndarray, threshold: float):
    df_res = defaultdict(list)
    values = mutual_info_classif(X=df.fillna(0), y=y_true)
    for col, val in zip(df.columns, values):
        df_res[col].append(val)
    df_res['threshold'].append(threshold)
    full_path = os.path.join(get_results_path(), 'all_features.csv')
    if os.path.exists(full_path):
        pd.DataFrame(df_res, index=df_res['threshold']).to_csv(full_path, header=False, mode='a')
    else:
        pd.DataFrame(df_res, index=df_res['threshold']).to_csv(full_path)


def plot():

    graphs = load_graphs_features('threshold', 0.44)
    rich_21 = graphs['rich_club_coefficient_21'].values
    rich_24 = graphs['rich_club_coefficient_24'].values
    rich_25 = graphs['rich_club_coefficient_25'].values
    avg_neigh_deg10 = graphs['average_neighbor_degree_10'].values
    avg_neigh_deg88 = graphs['average_neighbor_degree_88'].values
    pagerank_var = graphs['pagerank_numpy_variance']
    labels = get_y_true_regression()
    df = {'rich_club_coefficient_21': rich_21, 'rich_club_coefficient_24': rich_24,
          'rich_club_coefficient_25': rich_25,
          'average_neighbor_degree_10': avg_neigh_deg10,
          'average_neighbor_degree_88': avg_neigh_deg88, 'labels': labels,
          'pagerank_var': pagerank_var}
    pd.DataFrame(df).to_csv('results_or3.csv', index=False)



if __name__ == '__main__':
    plot()




    # c.set('Default Params', 'project', 'stroke_before')
    # c.set('Default Params', 'class_name', 'CBM_T1_Classification')
    # df = features_by_type('wave', graphs=get_graphs(get_corr_lst(), np.arange(0.01, 0.2, step=0.01)), )
    # # df = load_graphs_features('threshold', 0.43)
    # y1 = get_y_true()
    #
    # c.set('Default Params', 'project', 'stroke_after')
    # c.set('Default Params', 'class_name', 'CBM_T2_Classification')
    # df2 = load_graphs_features('threshold', 0.43)
    # y2 = get_y_true()
    # train_model_subject_out(df, y1, df2, y2, 6)
