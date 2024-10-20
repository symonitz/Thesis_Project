import pandas as pd

from conf_pack.opts import parser
import copy
from conf_pack.configuration import tune_parameters
from pre_process import build_graphs_from_corr, load_scans, create_graphs_features_df, get_corr_lst, \
    initialize_hyper_parameters
from feature_extraction import main_global_features, features_by_type
from train import train_model, predict_by_criterions, info_gain_all_features, train_model_subject_out, lso, clean_df, \
    compute_inf_gain
import nilearn
from numpy import corrcoef
from utils import *
from collections import defaultdict
from typing import Callable
import matplotlib.pyplot as plt

from visualization import build_features_for_scatters, scatter_plot, hist_class

k = 100


def main():
    performances = defaultdict(list)
    labels = by_task(lambda: get_y_true())
    names = get_subjects()
    min_feat = default_params.getint('min_features')
    max_feat = default_params.getint('max_features')
    filter_type = default_params.get('filter')
    min_thresh = default_params.getfloat('min_thresh')
    max_thresh = default_params.getfloat('max_thresh')
    is_globals = default_params.get('features_type') == 'globals'
    step = default_params.getfloat('step')

    if not is_globals:
        graphs = get_graphs(by_task(lambda: get_corr_lst()), list(np.arange(min_thresh, max_thresh, step)))

    for thresh in np.arange(min_thresh, max_thresh, step):

        if is_globals:
            features = by_task(lambda: load_graphs_features(filter_type, thresh))
        # print(thresh)

        for feat_num in range(min_feat, max_feat):

            if not is_globals:
                features = features_by_type(default_params.get('features_type'), graphs[thresh], feat_num)
            relevant_names = by_task(lambda: get_names())
            acc, _, _ = train_model(features, labels, feat_num, relevant_names)
            performances[(thresh, feat_num)].append(acc)
    save_results(performances)

    config_update({filter_type: tune_parameters[filter_type]})


def fetch_data_example():
    data = nilearn.datasets.fetch_adhd(n_subjects=40, data_dir='C:/Users/orsym/Documents/Data/ADHD')
    return data


def hyper_parameter(hyper_parameters: Dict):
    # Todo: Another table like count table only for features. Each feature is a column.
    # Todo: Change to lso when not regular case
    loo = lso(by_task(lambda: get_names()))
    meta_data = defaultdict(list)
    performances, counts_table, features_table, y, avg_acc, corr_lst, filter_type = initialize_hyper_parameters()
    preds = [0] * len(corr_lst)
    config_update(copy.deepcopy(hyper_parameters))
    is_globals = default_params.get('features_type') == 'globals'
    if not is_globals:
        graphs = get_graphs(by_task(lambda: get_corr_lst()), hyper_parameters[filter_type])

    for train_idx, test_idx in loo.split(corr_lst):

        best_thresh, best_acc, best_num, best_model, feat_names_best = 0, 0, 0, None, None

        for criteria_thresh in hyper_parameters[default_params.get('filter')]:

            if is_globals:

                df = by_task(lambda: load_graphs_features(filter_type, criteria_thresh))

            # info_gain_all_features(df, y, threshold=criteria_thresh)

            for num_features in hyper_parameters['num_features']:

                if not is_globals:
                    df = features_by_type(default_params.get('features_type'),
                                                          graphs[criteria_thresh], num_features)
                X_train, X_test = df.iloc[train_idx], df.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                relevant_names = np.array(by_task(lambda: get_names()))[train_idx]
                acc, model, feat_names = train_model(X_train, y_train, num_features, relevant_names)
                feat_values = compute_inf_gain(X_train.fillna(0), feat_names, y_train)
                write_selected_features(feat_names, feat_values)
                X_test = clean_df(X_test, feat_names)
                meta_data['pred'].append(model.predict(X_test))
                meta_data['accuracy'].append(acc)
                meta_data['threshold'].append(criteria_thresh)
                meta_data['number of features'].append(num_features)
                # meta_data['features'].append(feat_names.values)
                meta_data['classification_true_ground'].append(y_test)
                meta_data['CBM_true_ground'].append(by_task(lambda: get_y_true_regression())[test_idx])

                if acc > best_acc:
                    best_acc = acc
                    feat_names_best = feat_names
                    best_thresh, best_num, best_model = criteria_thresh, num_features, model
                    preds[test_idx] = best_model.predict(X_test)
        if is_globals:
            df = load_graphs_features(filter_type, best_thresh)

        else:
            df = features_by_type(default_params.get('features_type'), graphs[best_thresh], best_num)

        acc, pred = predict_by_criterions(model=best_model, filter_type=filter_type, thresh=best_thresh, idx=test_idx,
                                         y=y, col_names=feat_names_best, best_num=best_num, df=df)
        avg_acc += acc

        counts_table[(best_thresh, best_num)] = counts_table[(best_thresh, best_num)] + 1

        for feat in feat_names_best:
            features_table[feat] = features_table[feat] + 1
    print(avg_acc)
    avg_acc = np.float(avg_acc)
    avg_acc /= len(y)

    counts_table_refactored = dict_to_df(counts_table, 'params', 'num_counts', 'count_table.csv')
    pd.DataFrame(meta_data).to_csv(os.path.join(get_results_path(), 'meta_data.csv'))
    create_stability_df(counts_table_refactored)
    feat_table_refactored = dict_to_df(features_table, 'feature', 'num_counts', 'feat_count_table.csv')
    feat_table_refactored.sort_values(by='num_counts', inplace=True)

    # for i in range(0, 5):
    #     plot_hyper_parameters(feat_table_refactored, filter_type, hyper_parameters, i)
    pd.DataFrame({'preds': preds}).to_csv(os.path.join(get_results_path(), 'preds.csv'), index=False)
    with open(os.path.join(get_results_path(), 'Results.txt'), 'a') as f:
        f.write(f'The accuracy of this experiment is {avg_acc}\n')

        # pd.DataFrame(performances).to_csv(os.path.join(get_results_path(), 'hyper_parameters.csv'), index=False)
    config_to_save = copy.deepcopy(hyper_parameters)

    config_update(config_to_save)

    return performances


def config_update(config_to_save):
    config_to_save.update({'filtering criteria': [default_params.get('filter')], 'class predict': \
        [default_params.get('class_name')], 'features_type': [default_params.get('features_type')]})
    save_config(config_to_save)


def plot_hyper_parameters(feat_table_refactored, filter_type, hyper_parameters, i):
    feat_name_to_plot = feat_table_refactored.iloc[i]['feature']
    scatter_plot(build_features_for_scatters(filter_type, hyper_parameters['threshold'],
                                             feat_name_to_plot, get_y_true_regression()), feat_name_to_plot)
    hist_class(build_features_for_scatters(filter_type, hyper_parameters['threshold'], feat_name_to_plot,
                                           by_task(lambda: get_y_true())), feat_name_to_plot)


def example():
    thresh = 0.4
    corr_lst = load_scans([os.path.join(SCANS_DIR_BEFORE, name) for name in os.listdir(SCANS_DIR_BEFORE)])
    graphs = build_graphs_from_corr(corr_lst=corr_lst, filter_type='threshold', param=thresh)
    main_global_features(graphs)

    print()


def graph_pre_process():

    corr_lst = by_task(lambda: get_corr_lst())
    create_graphs_features_df(corr_lst=corr_lst, filter_type='threshold', thresholds=np.arange(start=0, stop=0.75,
                                                                                               step=0.01))


def build_and_save_graphs(param):
    corr_lst = by_task(lambda: get_corr_lst())
    build_graphs_from_corr(default_params.get('filter'), corr_lst, param)


def get_graphs(corr_lst: List[np.ndarray], params: List[float]) -> Dict:
    graphs_by_param = {}
    for param in params:
        graphs_by_param[param] = build_graphs_from_corr(default_params.get('filter'), corr_lst, param)
    return graphs_by_param


def embedding_experiments(func: Callable, *args) -> NoReturn:
    for exp in ['wave', 'heat', 'fgsd', 'graph2vec']:
    # for exp in  ['fgsd']:
        c.set('Default Params', 'features_type', exp)
        c.set('Default Params', 'result_path', 'default')
        func(args[0])


def objective_func_experiments(func: Callable, *args) -> NoReturn:
    for target_func in ['Is Efficient', 'CBM_T1_Classification', 'CBM_T2_Classification']:
        c.set('Default Params', 'class_name', target_func)
        c.set('Default Params', 'result_path', 'default')
        func(args)


def wrap_func(func_wrapper: Callable, func: Callable) -> Callable:
    return lambda: func_wrapper(func)


def main_derivate(params: Dict):
    c.set('Default Params', 'project', 'stroke_before')
    c.set('Default Params', 'class_name', 'CBM_T1_Classification')
    df = load_graphs_features('threshold', 0.43)
    y1 = get_y_true()

    c.set('Default Params', 'project', 'stroke_after')
    c.set('Default Params', 'class_name', 'CBM_T2_Classification')
    df2 = load_graphs_features('threshold', 0.43)
    y2 = get_y_true()
    for num_features in params['num_features']:
        train_model_subject_out(df, y1, df2, y2, num_features)


def initalize_params(args):
    c.set('Default Params', 'min_thresh', str(args.min_threshold))
    c.set('Default Params', 'max_thresh', str(args.max_threshold))
    c.set('Default Params', 'min_features', str(args.min_features))
    c.set('Default Params', 'max_features', str(args.max_features))
    c.set('Default Params', 'filter', args.criteria)
    c.set('Default Params', 'task', args.task)
    c.set('Default Params', 'features_type', args.f_type)
    c.set('Default Params', 'step', str(args.step))



if __name__ == '__main__':
    # graph_pre_process()
    # plot()

    ###############
    initalize_params(parser.parse_args())
    print(parser.parse_args().min_threshold)
    config = {default_params.get('filter'): list(np.arange(default_params.getfloat('min_thresh'),
                                                           default_params.getfloat('max_thresh'),
                                                           step=default_params.getfloat('step'))),
              'num_features': list(range(default_params.getint('min_features'), default_params.getint('max_features')))}
    main()
    # build_and_save_graphs('0.02')
    hyper_parameter(config)
    ######  #########
    # embedding_experiments(hyper_parameter, config)
    # wrap_func(embedding_experiments, hyper_parameter)
    # plot()