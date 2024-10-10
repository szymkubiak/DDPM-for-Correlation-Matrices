import numpy as np
import pandas as pd
import pickle
import copy
import datetime
import fastcluster
import powerlaw
from scipy.stats import wasserstein_distance
from scipy.cluster.hierarchy import cophenet
from configs import configs
from utility import select_empirical_dataset, load_dataset, extract, diffusion_process

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def adj_corr_matrix(corrmat):
    a, b = np.triu_indices(corrmat.shape[0], k=1)
    np.fill_diagonal(corrmat, 1)
    corrmat[b, a] = corrmat[a, b]
    np.fill_diagonal(corrmat, 1)
    corrmat = np.clip(corrmat, -1, 1)
    return corrmat

def adjust_all_matrices(samples_list):
    samples_list_ = copy.deepcopy(samples_list)
    samples_list = []
    for corr_matrix in samples_list_:
        corr_matrix = adj_corr_matrix(corr_matrix)
        samples_list.append(corr_matrix)
    return samples_list

def eigenval_gini_coefficient(eigenvals):
    sorted_eigenvalues_list = sorted(eigenvals.astype(float).tolist())
    height, area = 0, 0
    for value in sorted_eigenvalues_list:
        height = height + value
        area = area + (height - value / 2)
    fair_area = height * len(sorted_eigenvalues_list) / 2
    return (fair_area - area) / fair_area

def coph_corr(corr_matrix, method):
    a, b = np.triu_indices(len(corr_matrix), k=1)
    dist = np.sqrt(2 * (1 - corr_matrix)).values
    Z = fastcluster.linkage(dist[a, b], method=method)
    return cophenet(Z, dist[a, b])[0]

def perr_frob_property_neg(eigenvecs):
    return np.where(eigenvecs[0] < 0, eigenvecs[0], 0).sum()

def power_law_exponent(data):
    fitted_pl = powerlaw.Fit(data, verbose=False)
    return fitted_pl.alpha

def comparison_metrics(corr_matrix):
    # Mean correlation
    mean_corr = corr_matrix.values.ravel().mean()

    # Gini coefficient of eigenvalues
    eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
    sort = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[sort]
    eigenvecs = eigenvecs[:, sort]
    eigen_gini = eigenval_gini_coefficient(eigenvals)

    # Cophenetic correlations of correlation coefficients
    coph_corr_single = coph_corr(corr_matrix, 'single')
    coph_corr_ward = coph_corr(corr_matrix, 'ward')

    # Sum of negative entries of the first eigenvector (related to Perron Frobenius property)
    neg_first_eigenvector = -perr_frob_property_neg(eigenvecs)

    # Power law exponent of the eigenvalue distribution
    eigenvals_power_law_alpha = power_law_exponent(eigenvals)

    return {"mean_correl": mean_corr,
            "eigen_gini": eigen_gini,
            "coph_corr_single": coph_corr_single,
            "coph_corr_ward": coph_corr_ward,
            "perron_frob_sum_neg": neg_first_eigenvector,
            "power_eigen_values": eigenvals_power_law_alpha}

dataset_names = ['futures', 'fixed income', 'stocks']
for dataset_name in dataset_names:
    config = configs[dataset_name]

    # Load data
    train_end_dt = config['train_end_dt']
    data_file = open('Data/' + dataset_name + ' data.pickle', 'rb')
    if dataset_name in ['futures', 'stocks']:
        emp_corr_matrices_list, emp_dates_list, emp_context_data_list = pickle.load(data_file)
    elif dataset_name in ['fixed income', 'fixed income_cond']:
        emp_corr_matrices_list, _, _, _, emp_dates_list, emp_context_data_list = pickle.load(data_file)

    start_date = datetime.date(1990, 1, 1)
    end_date = train_end_dt
    train_corr_matrices_list, train_dates_list, train_context_list = select_empirical_dataset(start_date, end_date,
                                                                                              emp_corr_matrices_list,
                                                                                              emp_dates_list,
                                                                                              emp_context_data_list)
    start_date = train_end_dt
    end_date = max(emp_dates_list)
    test_corr_matrices_list, test_dates_list, test_context_list = select_empirical_dataset(start_date, end_date,
                                                                                           emp_corr_matrices_list,
                                                                                           emp_dates_list,
                                                                                           emp_context_data_list)

    filehandler = open('samples_DM_' + dataset_name + '.pickle',"rb")
    dm_sampled_data = pickle.load(filehandler)

    filehandler = open('samples_CDM_Train_' + dataset_name + '_cond.pickle',"rb")
    cdm_sampled_data = pickle.load(filehandler)

    filehandler = open('samples_CDM_Test_' + dataset_name + '_cond.pickle',"rb")
    cdm_test_sampled_data = pickle.load(filehandler)

    #Diagonals
    corrs_lists = [dm_sampled_data, cdm_sampled_data, cdm_test_sampled_data]
    corr_names = ['DM', 'CDM_Train', 'CDM_Test']
    diagonals_results_df = pd.DataFrame()
    for corr_list_num, corr_list in enumerate(corrs_lists):
        diagonals_list = []
        for corr_matrix in corr_list:
            diagonals_list.append(np.diag(corr_matrix))
        diagonals_array = np.array(diagonals_list)
        avg_abs_dist = np.abs((diagonals_array - 1)).mean()
        std = np.abs((diagonals_array - 1)).std()
        results_row = pd.DataFrame([[avg_abs_dist, std]], index = [corr_names[corr_list_num]])
        diagonals_results_df = pd.concat([diagonals_results_df, results_row], axis=0)
    diagonals_results_df.columns = ['Average Abs Dist', 'Std Abs Dist']
    print(diagonals_results_df)
    diagonals_results_df.T.to_excel('diagonals_' + dataset_name + '.xlsx')

    #Symmetry
    corrs_lists = [dm_sampled_data, cdm_sampled_data, cdm_test_sampled_data]
    corr_names = ['DM', 'CDM', 'CDM_Test']
    symmetry_results_df = pd.DataFrame()
    for corr_list_num, corr_list in enumerate(corrs_lists):
        all_diffs = []
        for corr_matrix in corr_list:
            corr_tr_diffs = np.triu(corr_matrix).T - np.tril(corr_matrix)
            for row in range(0, corr_matrix.shape[0]):
                for col in range(0, corr_matrix.shape[1]):
                    if row > col:
                        diff = corr_tr_diffs[row, col]
                        all_diffs.append(diff)
        diffs_array = np.array(all_diffs)
        avg = np.abs(diffs_array).mean()
        std = diffs_array.std()
        results_row = pd.DataFrame([[avg, std]], index = [corr_names[corr_list_num]])
        symmetry_results_df = pd.concat([symmetry_results_df, results_row], axis=0)
    symmetry_results_df.columns = ['Average', 'Std']
    print(symmetry_results_df)
    symmetry_results_df.T.to_excel('symmetry_' + dataset_name + '.xlsx')

    #Adjust matrices for further assessment
    dm_all_gen_corrs = adjust_all_matrices(dm_sampled_data)
    cdm_sampled_data = adjust_all_matrices(cdm_sampled_data)
    cdm_test_sampled_data = adjust_all_matrices(cdm_test_sampled_data)

    #Wasserstein distances
    w_distances_df = pd.DataFrame()

    datasources = [dm_all_gen_corrs, cdm_sampled_data, cdm_test_sampled_data]
    source_names = ['dm', 'cdm_train', 'cdm_test']

    empirical_train_flat = np.array([matrix.flatten() for matrix in train_corr_matrices_list]).flatten()
    empirical_test_flat = np.array([matrix.flatten() for matrix in test_corr_matrices_list]).flatten()

    for data_num, data in enumerate(datasources):
        d_flat = np.array([matrix.flatten() for matrix in data]).flatten()
        distance_to_train = wasserstein_distance(d_flat, empirical_train_flat)
        distance_to_test = wasserstein_distance(d_flat, empirical_test_flat)
        data_row_df = pd.DataFrame([[distance_to_train, distance_to_test]], index = [source_names[data_num]], columns = ['Distance to Train', 'Distance to Test'])
        w_distances_df = pd.concat([w_distances_df, data_row_df], axis=0)
    print(w_distances_df)
    w_distances_df.T.to_excel('wasserstein distances_' + dataset_name + '.xlsx')

    #Stylized facts of correlation matrices
    datasources = [train_corr_matrices_list, dm_all_gen_corrs, cdm_sampled_data, cdm_test_sampled_data]
    source_names = ['empirical_train', 'dm', 'cdm_train', 'cdm_test']

    all_results_df = pd.DataFrame()
    for corr_source_num, corrs in enumerate(datasources):
        results_df = pd.DataFrame()
        source_name = source_names[corr_source_num]
        for corr_matrix in corrs:
            corr_matrix_metrics = comparison_metrics(pd.DataFrame(corr_matrix))
            corr_matrix_row = pd.DataFrame.from_dict([corr_matrix_metrics])
            results_df = pd.concat([results_df, corr_matrix_row], axis=0)
        results_df['source'] = source_name
        all_results_df = pd.concat([all_results_df, results_df], axis=0).reset_index(drop=True)

    print(all_results_df.groupby('source').mean().round(3))
    print(all_results_df.groupby('source').std().round(3))
    all_results_df.T.to_pickle('stylized facts_' + dataset_name + '.pickle')

    #Conditional generation
    test_context_quintiles_list = []
    train_quintiles_list = []
    test_quintiles_list = []
    cdm_train_quintiles_list = []
    cdm_test_quintiles_list = []

    def aggregate_quintiles(emp_data, cdm_data, context_data, cond_num):
        context_quintiles_list = []
        emp_quintiles_list = []
        cdm_quintiles_list = []
        for d in range(0, 5):
            emp_quintile_data = []
            cdm_quintile_data = []
            context_quintile_data = []
            for c_num, c in enumerate(emp_data):
                context = context_data[c_num][cond_num]
                if (context >= np.quantile(np.array(context_data)[:, cond_num], (d*2) / 10)) and (context < np.quantile(np.array(context_data)[:, cond_num], ((d+1) * 2) / 10)):
                    context_quintile_data.append(context)
                    emp_quintile_data.append(c)
                    cdm_quintile_data.append(cdm_data[c_num])

            context_quintiles_list.append(copy.deepcopy(context_quintile_data))
            emp_quintiles_list.append(copy.deepcopy(emp_quintile_data))
            cdm_quintiles_list.append(copy.deepcopy(cdm_quintile_data))
        return emp_quintiles_list, cdm_quintiles_list, context_quintiles_list

    cond_num = 0
    train_quintiles_list0, cdm_train_quintiles_list0, train_context_quintiles_list0 = aggregate_quintiles(train_corr_matrices_list, cdm_sampled_data, train_context_list, cond_num)

    cond_num = 1
    train_quintiles_list1, cdm_train_quintiles_list1, train_context_quintiles_list1 = aggregate_quintiles(train_corr_matrices_list, cdm_sampled_data, train_context_list, cond_num)

    w_distances_df = pd.DataFrame()
    empirical_train_flat = np.array([matrix.flatten() for matrix in train_corr_matrices_list]).flatten()
    dm_all_flat = np.array([matrix.flatten() for matrix in dm_all_gen_corrs]).flatten()

    for i in range(0, 5):
        train_quintile0 = train_quintiles_list0[i]
        cdm_quintile0 = cdm_train_quintiles_list0[i]
        train_quintile1 = train_quintiles_list1[i]
        cdm_quintile1 = cdm_train_quintiles_list1[i]
        pairs = [[cdm_quintile0, train_quintile0], [cdm_quintile0, empirical_train_flat],
                [cdm_quintile1, train_quintile1], [cdm_quintile1, empirical_train_flat],
                [train_quintile0, empirical_train_flat], [train_quintile1, empirical_train_flat],
                [dm_all_flat, train_quintile0], [dm_all_flat, train_quintile1]]
        pair_names = ['Rates: CDM vs Train Quintile', 'Rates: CDM vs All Train',
                      'Equity: CDM vs Train Quintile', 'Equity: CDM vs All Train',
                      'Rates: Train Quintile vs All Train', 'Equity: Train Quintile vs All Train',
                      'Rates: All DM vs. Train Quintile', 'Equity: All DM vs. Train Quintile']
        data_row = []
        for pair in pairs:
            d_flat0 = np.array([matrix.flatten() for matrix in pair[0]]).flatten()
            if type(pair[1]) is list:
                d_flat1 = np.array([matrix.flatten() for matrix in pair[1]]).flatten()
            else:
                d_flat1 = pair[1]
            distance = wasserstein_distance(d_flat0, d_flat1)
            data_row.append(distance)
        w_distances_df = pd.concat([w_distances_df, pd.DataFrame([data_row], index = [i], columns = pair_names)], axis=0)

    w_distances_df.T.to_excel('conditional generation_' + dataset_name + '.xlsx')