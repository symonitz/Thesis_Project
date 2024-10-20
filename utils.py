import os
from typing import NoReturn, DefaultDict, Dict, Tuple, Any, Callable
import datetime

import networkx as nx
import pandas as pd
import numpy as np
from conf_pack.paths import *
from conf_pack.configuration import default_params, c
from collections import defaultdict
from typing import List



def write_time_of_function(func_name: str, old_time: datetime.datetime) -> NoReturn:
    df = pd.DataFrame()
    row = {'function': [func_name], 'time': [datetime.datetime.now() - old_time]}
    row = pd.DataFrame(row)
    name_file = 'function_times.csv'
    if os.path.exists(name_file):
        df = pd.read_csv(name_file)
    df = pd.concat([df, row])
    df.to_csv(name_file, index=False)


def dict_to_df(dict_to_conv: DefaultDict, key_name: str, val_name: str, df_name: str) -> pd.DataFrame:
    res = defaultdict(list)
    for key, val in dict_to_conv.items():
        res[key_name].append(key)
        res[val_name].append(val)
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(get_results_path(), df_name), index=False)
    return df


def create_stability_df(count_table_df: pd.DataFrame) -> NoReturn:
    '''
    This method takes the count table of the hyper-parameters and creates txt file with the variance of their values
    :param count_table_df:
    :return:
    '''
    params = count_table_df['params'].values
    num_counts = count_table_df['num_counts'].values
    total_criteria, total_num_features = 0, 0
    for (criteria, features_num), num_appears in zip(params, num_counts):
        total_criteria += criteria * num_appears
        total_num_features += features_num * num_appears
    criteria_var = np.var(np.array(total_criteria))
    num_features_var = np.var(np.array(total_num_features))
    with open(os.path.join(get_results_path(), 'hss.txt'), 'a') as f:
        f.write(f'The variance of criteria selection is {criteria_var} \n. The variance of number of features is '
                f'{num_features_var}')

#
# def compute_rs_stability(results_df: pd.DataFrame) -> NoReturn:
#     sorted_df = results_df.sort_values(by=['number_of_features', 'criteria'])
#     for


def load_graphs_features(filter_type: str, thresh: float):
        return pd.read_pickle(os.path.join('Graphs_pickle', default_params.get('project'),
                                       filter_type, f'graph_{thresh:.2f}.pkl'))


def by_task(func: Callable):
    task = default_params.get('task')
    if task == 'prediction':
        c.set('Default Params', 'project', 'stroke_before')
        c.set('Default Params', 'class_name', 'Is Efficient')
        return func()
    elif task == 'derive':
        c.set('Default Params', 'project', 'stroke_before')
        c.set('Default Params', 'class_name', 'CBM_T1_Classification')
        x1 = func()

        c.set('Default Params', 'project', 'stroke_after')
        c.set('Default Params', 'class_name', 'CBM_T2_Classification')
        x2 = func()
        if type(x1) == np.ndarray:
            return np.concatenate([x1, x2])
        elif type(x1) == list:
            return x1 + x2
        elif type(x1) == pd.DataFrame:
            return pd.concat([x1, x2]).dropna(axis=1)
    elif task == 'derive_T1':
        c.set('Default Params', 'project', 'stroke_before')
        c.set('Default Params', 'class_name', 'CBM_T1_Classification')
        return func()
    elif task == 'derive_T2':
        c.set('Default Params', 'project', 'stroke_after')
        c.set('Default Params', 'class_name', 'CBM_T2_Classification')
        return func()

def save_config(conf_to_save: Dict):
    df = pd.DataFrame()
    for key in conf_to_save.keys():
        df_temp = pd.DataFrame({key: conf_to_save[key]})
        df = pd.concat([df, df_temp], axis=1)
    df.to_csv(os.path.join(get_results_path(), 'configuration.csv'))


def write_selected_features(feature_names: List[str], info_gain: List[float]) -> NoReturn:
    df = defaultdict(list)

    full_path = os.path.join(get_results_path(), 'selected_features.csv')
    if os.path.exists(full_path):
        df = pd.read_csv(full_path).to_dict(orient='list')

    for feature, info_feature in zip(feature_names, info_gain):
        if feature in df['feature_names']:
            idx = df['feature_names'].index(feature)
            df['number_of_times_chosen'][idx] = df['number_of_times_chosen'][idx] + 1
            df['summed_inf_gain'][idx] = df['summed_inf_gain'][idx] + info_feature

        else:
            df['feature_names'].append(feature)
            df['number_of_times_chosen'].append(1)
            df['summed_inf_gain'].append(info_feature)

    pd.DataFrame(df).to_csv(full_path, index=False)


def get_y_true() -> np.ndarray:
    df = get_meta_data()
    df.sort_values(by=[default_params.get('subject')], inplace=True)
    return df[default_params.get('class_name')].values


def get_subjects() -> np.ndarray:
    df = get_meta_data()
    subject_col = default_params.get('subject')
    df.sort_values(by=[subject_col], inplace=True)
    return df[subject_col].values


def get_y_true_regression() -> np.ndarray:
    df = get_meta_data()
    df.sort_values(by=['Subject'], inplace=True)
    return df['Delta Change'].values

def get_save_path() -> str:
    save_mapping_by_proj = {'stroke_before': STROKE_SAVE_PATH_PARENT, 'stroke_after': STROKE_SAVE_PATH_PARENT,
                            'adhd': ADHD_SAVE_PATH_PARENT}
    project_type = default_params.get('project')
    return save_mapping_by_proj[project_type]


def get_data_path() -> str:
    # data_path_mapping_by_proj = {'stroke': SCANS_DIR_BEFORE, 'adhd': ADHD_DATA_PATH}
    data_path_mapping_by_proj = {'stroke_before': SCANS_DIR_BEFORE, 'stroke_after': SCANS_DIR_AFTER,
                                'adhd': ADHD_DATA_PATH}
    project_type = default_params.get('project')
    return data_path_mapping_by_proj[project_type]


def get_meta_data() -> pd.DataFrame:
    data_path_mapping_by_proj = {'stroke_before': STROKE_EXCEL_DATA, 'stroke_after': STROKE_EXCEL_DATA,
                                 'adhd': ADHD_EXCEL_DATA}
    project_type = default_params.get('project')
    return pd.read_excel(data_path_mapping_by_proj[project_type])


def get_results_path() -> str:
    if default_params.get('result_path') == 'default':
        time = datetime.datetime.now()
        format_time = f'{time.year}_{time.month}_{time.day}_{time.hour}_{time.minute}'
        c.set('Default Params', 'result_path', format_time)
    full_path = os.path.join(get_save_path(), default_params.get('result_path'))
    os.makedirs(full_path, exist_ok=True)
    return full_path


def save_results(perf: DefaultDict) -> NoReturn:
    res = defaultdict(list)
    for (criteria, num_features), acc in perf.items():
        res['criteria'].append(criteria)
        res['number of features'].append(num_features)
        res['Accuracy'].append(acc)
    pd.DataFrame(res).to_csv(os.path.join(get_results_path(), 'results.csv'))



def get_names() -> List[str]:
    path = get_data_path()
    file = open(os.path.join(path, 'names.txt'))
    names = file.read().split('\n')
    return names[:-1]


if __name__ == '__main__':
    print(get_names())
#     d_type = 'correlation'
#     for file in os.listdir(os.path.join(get_data_path(),    )