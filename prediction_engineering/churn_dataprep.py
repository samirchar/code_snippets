from itertools import product
import multiprocessing as mp
import pandas as pd
import numpy as np
import dask.dataframe as dd
from pandas.tseries.offsets import MonthEnd, MonthBegin


def label_customer(customer_id, customer_id_col, churn_date_col, data, current_month_col, cutoff='M',
                   lead_time=1, prediction_window=1):
    """Make labels for a single customer (churn = 1 or 0)
    
    :param customer_id: the ID for the customer
    :type customer_id: str
    :param customer_id_col: the name of the column of customer ID
    :type customer_id_col: str
    :param churn_date_col: the name of the column containing churn dates for clients
    :type churn_date_col: str
    :param data: the dataframe containing the required columns
    :type data: :py:class:`pandas.DataFrame`
    :param current_month_col: the name of the column containing the date of the data.
    :type current_month_col: str
    :param cutoff: the time at which predictions are made, by default 'M' which
        means predictions are done at the end of the month, defaults to 'M'
    :type cutoff: str, required
    :param lead_time: number of periods in advance to make predictions for, defaults to 1
    :type lead_time: int, required
    :param prediction_window: number of periods over which to consider churn, defaults to 1
    :type prediction_window: int, required
    :return: a table of customer id, the cutoff times at the specified frequency, the 
        label for each cutoff time and the date on which the churn itself occurred
    :rtype: :py:class:`pandas.DataFrame`
    """
    # Don't modify original
    data_copy = data.copy()

    data_copy = data_copy[data_copy[customer_id_col] == customer_id]

    first_input = data_copy[current_month_col].min()
    last_input = data_copy[current_month_col].max()

    start_date = pd.datetime(first_input.year, first_input.month, 1)

    # Handle December
    if last_input.month == 12:
        end_date = pd.datetime(last_input.year + 1, 1, 1)
    else:
        end_date = pd.datetime(last_input.year, last_input.month + 1, 1)

    # Make dataframe of customer id and cutoff
    label_times = pd.DataFrame({'cutoff_time': pd.date_range(start_date, end_date, freq=cutoff),
                                'customer_id': customer_id
                                })

    # Use the lead time and prediction window parameters to establish the prediction window
    # Prediction window is for each cutoff time
    label_times['prediction_window_start'] = label_times['cutoff_time'] + \
        pd.DateOffset(months=lead_time)
    label_times['prediction_window_end'] = label_times['cutoff_time'] + \
        pd.DateOffset(months=lead_time + prediction_window)

    for i, row in label_times.iterrows():

        # Default values if unknown
        churn_date = pd.NaT
        label = np.nan
        # Find the window start and end
        window_start = row['prediction_window_start']
        window_end = row['prediction_window_end']
        # Determine if there were any churns during the prediction window
        churns = data_copy.loc[(data_copy[churn_date_col] >= window_start) &
                               (data_copy[churn_date_col] < window_end), churn_date_col]
        # print(churns)
        # Positive label if there was a churn during window
        if not churns.empty:
            label = 1
            churn_date = churns.values[0]

        # No churns, but need to determine if an active member
        else:
            label = 0

        # Assign values
        label_times.loc[i, 'label'] = label
        label_times.loc[i, churn_date_col] = churn_date

        # Handle case with no churns
        if not np.any(label_times['label'] == 1):
            label_times[churn_date_col] = pd.NaT

    return label_times

def make_label_times(data, customer_id_col, churn_date_col, current_month_col, cutoff='M',
                     lead_time=1, prediction_window=1):
    """[summary]
    
    :param data: the dataframe containing the required columns
    :type data: :py:class:`pandas.DataFrame`
    :param customer_id_col: the name of the column of customer ID
    :type customer_id_col: str
    :param churn_date_col: the name of the column containing churn dates for clients
    :type churn_date_col: str
    :param current_month_col: the name of the column containing the date of the data
    :type current_month_col: str
    :param cutoff: the time at which predictions are made, by default 'M' which
        means predictions are done at the end of the month, defaults to 'M'
    :type cutoff: str, required
    :param lead_time: number of periods in advance to make predictions for, defaults to 1
    :type lead_time: int, optional
    :param prediction_window: number of periods over which to consider churn, defaults to 1
    :type prediction_window: int, optional
    :return: a table of customer ids, the cutoff times at the specified frequency, the 
         label for each cutoff time and the date on which the churn itself occurred
    :rtype: :py:class:`pandas.DataFrame`
    """
    customers = data[customer_id_col].unique()
    max_pool = mp.cpu_count()

    with mp.Pool(processes=max_pool) as pool:

        labeled_customers_data = pool.starmap(label_customer, product(customers, [customer_id_col], [churn_date_col], [data],
                                                                      [current_month_col], [
                                                                          cutoff],
                                                                      [lead_time], [prediction_window]))

    # Concatenate into a single dataframe
    return pd.concat(labeled_customers_data, axis=0)


def label_customer_dask(customer_id_col, churn_date_cols, df, current_month_col, period='end',
                        lead_time=1, prediction_window=1, return_pandas = False):
    """[summary]
    
    :param customer_id_col: the name of the column of customer ID
    :type customer_id_col: str
    :param churn_date_cols: a list with the names of the different types of churn
    :type churn_date_cols: list
    :param df: the dataframe containing the required columns
    :type df: :py:class:`dask.dataframe.DataFrame`
    :param current_month_col: the name of the column containing the date of the data.
    :type current_month_col: str
    :param period: the preiodicity of predictions (also known as cutoff), defaults to 'end'.
        By default, prediction take place at the end of the month.
    :type period: str, required
    :param lead_time: number of periods in advance to make predictions for, defaults to 1
    :type lead_time: int, required
    :param prediction_window: number of periods over which to consider churn, defaults to 1
    :type prediction_window: int, required
    :return: a table of customer id, the cutoff times at the specified frequency, the 
        label for each cutoff time and the date on which the churn itself occurred
    :rtype: :py:class:`dask.dataframe.DataFrame`
    """
    def get_cutoffs(data,column,current_format,desired_format,period = 'end'):
        if period == 'end':
            return dd.to_datetime((dd.to_datetime(data[column],format=current_format)+MonthEnd(0)).dt.strftime(desired_format),format=desired_format)
        elif period =='start':
            return dd.to_datetime(data[column],format=current_format)
        else:
            print('Prediodicity not supported')

    dates_df = df.groupby(customer_id_col)[current_month_col].agg(['min','max'])
    df = df.set_index(customer_id_col)
    df_aug = df.merge(dates_df)
    df_aug['cutoff_time'] = get_cutoffs(df_aug,current_month_col,'%Y%m','%Y-%m-%d', period)

    df_aug['prediction_window_start'] = df_aug['cutoff_time'] + \
        pd.DateOffset(months=lead_time)
    df_aug['prediction_window_end'] = df_aug['cutoff_time'] + \
        pd.DateOffset(months=lead_time + prediction_window)
    
    for churn_date_col in churn_date_cols:
        df_aug['label_{}'.format(churn_date_col)]=0
        df_aug['label_{}'.format(churn_date_col)] = df_aug['label_{}'.format(churn_date_col)].where(~((df_aug[churn_date_col] >= df_aug['prediction_window_start']) & (df_aug[churn_date_col] < df_aug['prediction_window_end'])),1)
    
    if return_pandas:
        return df_aug.compute()
    else:
        return df_aug
    

def data_split(label_times, cutoff_col, train_size=0.6, val_size=0.2):
    """[summary]
    
    :param label_times: a table of customer ids, the cutoff times at the specified frequency, the 
         label for each cutoff time and the date on which the churn itself occurred
    :type label_times: :py:class:`pandas.DataFrame`
    :param cutoff_col: the name of the column containing churn cutoff dates
    :type cutoff_col: str
    :param train_size: proportion of the training set, by default 0.6, defaults to 0.6
    :type train_size: float, required
    :param val_size: the proportion of the validation set, by default 0.2, defaults to 0.2
    :type val_size: float, required
    :return: one dataframe for each split (train, val and test)
    :rtype: :py:class:`pandas.DataFrame`
    """
    label_times.sort_values(by=cutoff_col, inplace=True)

    train_val_split_date = label_times[cutoff_col].quantile(train_size)
    val_test_split_date = label_times[cutoff_col].quantile(train_size+val_size)

    train = label_times[label_times[cutoff_col] < train_val_split_date].copy()
    val = (label_times[(label_times[cutoff_col] >= train_val_split_date) &
                       (label_times[cutoff_col] < val_test_split_date)].copy())
    test = label_times[label_times[cutoff_col] >= val_test_split_date].copy()

    print(' train ranges from {} to {}'.format(train[cutoff_col].min(), train[cutoff_col].max()), '\n\n',
          'val ranges from {} to {}'.format(
              val[cutoff_col].min(), val[cutoff_col].max()), '\n\n',
          'test ranges from {} to {}'.format(test[cutoff_col].min(), test[cutoff_col].max()))

    return train, val, test
