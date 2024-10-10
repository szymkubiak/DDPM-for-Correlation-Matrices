import datetime
import numpy as np
import pandas as pd
import pickle


#Load conditioning data
context_df = pd.read_excel('Data//SPX and 10Y.xlsx', index_col=[0])
context_df = context_df.sort_index(ascending=True)
context_df = context_df.loc[~context_df.index.weekday.isin([5,6])]

def context_vol(df, d, periods, freq):
    df_ = df.loc[df.index <= pd.to_datetime(d)].copy()
    if freq == 'weekly':
        df_ = df_.loc[df_.index.weekday == df_.index.weekday[-1]]
        vol_m = np.sqrt(52)
    elif freq == 'daily':
        vol_m = np.sqrt(260)
    vol1 = df_['US10Y'].diff().iloc[-periods:].std() * vol_m
    vol2 = df_['SPX'].pct_change().iloc[-periods:].std() * vol_m
    return vol1, vol2


### Futures dataset ------------------------------------------------------------------------------
data_df = pd.read_excel('Data//Futures Data.xlsx', index_col=[0])
data_df = data_df.dropna().sort_index()
data_df = data_df.loc[~data_df.index.weekday.isin([5,6])]
data_ch_df = data_df.pct_change().dropna()

days_num = 65
corrs_list = []
dates_list = []
emp_context_list = []

for i in range(days_num, data_ch_df.shape[0]):
    d = data_ch_df.index[i]
    corr = data_ch_df.iloc[i-days_num:i].corr()
    corrs_list.append(corr)
    dates_list.append(data_ch_df.index[i])
    emp_context_list.append(context_vol(context_df, d, days_num, 'daily'))

filehandler = open("Data//futures data.pickle","wb")
pickle.dump([np.array(corrs_list),
             np.array(dates_list),
             np.array(emp_context_list)], filehandler)
filehandler.close()


### Stocks dataset ------------------------------------------------------------------------------
data_df = pd.read_excel('Data//MSCI Europe Data_19 Oct 2023.xlsx', index_col=[0])
data_df = data_df.loc[data_df.index >= pd.to_datetime(datetime.datetime(1993,10,18))]
null_cols = data_df.columns[data_df.isnull().any()]
data_df = data_df.drop(null_cols, axis = 1)
data_df = data_df.sort_index()
data_df = data_df.loc[~data_df.index.weekday.isin([5,6])]
data_ch_df = data_df.pct_change().dropna()

days_num = 520
corrs_list = []
dates_list = []
emp_context_list = []

for i in range(days_num, data_ch_df.shape[0]):
    d = data_ch_df.index[i]
    corr = data_ch_df.iloc[i-days_num:i].corr()
    corrs_list.append(corr)
    dates_list.append(data_ch_df.index[i])
    emp_context_list.append(context_vol(context_df, d, days_num, 'daily'))

filehandler = open("Data//stocks data.pickle","wb")
pickle.dump([np.array(corrs_list),
             np.array(dates_list),
             np.array(emp_context_list)], filehandler)
filehandler.close()


### Fixed income dataset ------------------------------------------------------------------------------
tr_indices_df = pd.read_excel("Data//Fixed Income Data_2022 10.xlsx", sheet_name = "TR Indices", index_col=[0])
yields_df = pd.read_excel("Data//Fixed Income Data_2022 10.xlsx", sheet_name = "Yields", index_col=[0])
bond_currencies_df = pd.read_excel('Bloomberg Tickers.xlsx', sheet_name = "Fixed Income", index_col=[0],
                                   skiprows=[0,1,2,3])[['Currency']]

#remove weekends
tr_indices_df = tr_indices_df.loc[~tr_indices_df.index.weekday.isin([5,6])]
yields_df = yields_df.loc[~yields_df.index.weekday.isin([5,6])]

#Load expected returns
expected_returns_df = (yields_df.copy() / 100)

#Add hedge yield to expected returns
for bond_idx, row in bond_currencies_df.iterrows():
    currency = row["Currency"]
    if currency != "USD":
        expected_returns_df[bond_idx] = expected_returns_df[bond_idx] - expected_returns_df[currency]
expected_returns_df = expected_returns_df.drop(columns="USD 3m Libor")

def generate_empirical_corr_matrices(data, lookback_period = 52):
    vols_list = []
    corr_matrices_list = []
    corr_matrix_dates = []
    forward_tot_rets_list = []
    macro_data_list = []
    data_df = data.copy()
    available_idx = expected_returns_df.dropna().index.map(datetime.datetime.date)
    first_available_dt = available_idx[-1]
    dates = data_df.index
    data_df = data_df.reset_index(drop=True)
    for row_num, row in data_df.iterrows():
        corr_matrix_date = dates[row_num - lookback_period]
        if corr_matrix_date >= first_available_dt:
            if row_num >= lookback_period:
                forward_tot_ret = weekly_tr_indices_df.shift(4).loc[corr_matrix_date] / weekly_tr_indices_df.loc[corr_matrix_date] -1
                returns_data_df = (data_df / data_df.shift(-1) - 1).dropna().iloc[row_num - lookback_period : row_num].copy()
                vols = returns_data_df.std()

                if not forward_tot_ret.isnull().any():
                    corr_matrix = returns_data_df.corr()
                    corr_matrix = corr_matrix.fillna(0)
                    corr_matrices_list.append(corr_matrix)
                    corr_matrix_dates.append(corr_matrix_date)
                    vols_list.append(vols)
                    forward_tot_rets_list.append(forward_tot_ret)

                    #macro data
                    macro_data = context_vol(context_df, corr_matrix_date, 52, 'weekly')
                    macro_data_list.append(macro_data)

    return corr_matrices_list, vols_list, corr_matrix_dates, forward_tot_rets_list, macro_data_list

emp_corr_matrices_list = []
emp_corr_matrices_dates_list = []
emp_vols_list = []
emp_forward_tot_rets_list = []
emp_context_data_list = []
for weekday in range(0,5):
    weekly_tr_indices_df = tr_indices_df.loc[tr_indices_df.index.weekday == weekday].dropna()
    weekly_tr_indices_df.index = weekly_tr_indices_df.index.map(datetime.datetime.date)
    cm_weekday_list, vols_list, dates_weekday_list, forward_tot_rets_list, context_data_list = generate_empirical_corr_matrices(weekly_tr_indices_df, lookback_period = 52)
    emp_corr_matrices_list = emp_corr_matrices_list + cm_weekday_list
    emp_corr_matrices_dates_list = emp_corr_matrices_dates_list + dates_weekday_list
    emp_vols_list = emp_vols_list + vols_list
    emp_forward_tot_rets_list = emp_forward_tot_rets_list + forward_tot_rets_list
    emp_context_data_list = emp_context_data_list + context_data_list

expected_returns_df.index = expected_returns_df.index.map(datetime.datetime.date)
expected_returns_list = []
for d in emp_corr_matrices_dates_list:
    expected_returns_list.append(expected_returns_df.loc[d])

filehandler = open("Data//fixed income data.pickle","wb")
pickle.dump([np.array(emp_corr_matrices_list),
             np.array(emp_vols_list),
             np.array(expected_returns_list),
             np.array(emp_forward_tot_rets_list),
             np.array(emp_corr_matrices_dates_list),
             np.array(emp_context_data_list)], filehandler)
filehandler.close()