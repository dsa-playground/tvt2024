import datetime as datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scripts.preprocess.preprocess_collect import collect_str_input

def load_timeseries_data():
    df = pd.read_csv(filepath_or_buffer='data/Timeseries.csv', sep=';')
    df.set_index('Datum', inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df.asfreq('D')
    return df


# def kies_onderwerp():
#     vraag = """
#         Welke reeks wil je voorspellen (a, b, c):
#                 a. ZZP (verwachting clienten)
#                 b. Ziekteverzuimpercentage
#                 c. Inkoop materialen
#                 d. Flexpool (aantal personen)
#         """
#     mogelijke_antwoorden = ['a', 'b', 'c', 'd']

#     str = collect_str_input(
#         question=vraag, 
#         possible_entries=mogelijke_antwoorden)
#     dict_antwoorden = {'a': 'ZZP', 'b': 'Ziekteverzuim', 'c': 'Inkoop', 'd': 'Flexpool'}
#     print(f'Gekozen antwoord: {dict_antwoorden[str]}')
#     return dict_antwoorden[str]


def create_subset_df(df, start, end):
    df_subset = df.loc[(df.index >= start) & (df.index < end)].copy()
    return df_subset

def create_df_X_and_y(df, base_col):
    df_Xy = df.copy()
    df_y = df_Xy[base_col]
    df_X = pd.DataFrame(index=df_Xy.index)
    return df_X, df_y

def get_second_nan_index(s):
    first_valid = s.index.get_loc(s.first_valid_index())
    last_valid = s.index.get_loc(s.last_valid_index())
    second_nan_index = s.iloc[last_valid:].isna().argmin() + last_valid
    return second_nan_index

def apply_sin_cos_transformation(df, date_col, period='year'):
    if period == 'year':
        # Convert the date to the day of the year
        df[f'num_col_{period}'] = df[date_col].dt.dayofyear
        cycle_length = 365.25
    elif period == 'week':
        # Convert the date to the day of the week (0 = Monday, 6 = Sunday)
        df[f'num_col_{period}'] = df[date_col].dt.dayofweek
        cycle_length = 7
    else:
        raise ValueError(f"Period {period} is not supported. Please choose 'year' or 'week'.")
    
    # Apply the sine transformation
    df[f'sin_transformation_{period}'] = np.sin(2 * np.pi * df[f'num_col_{period}'] / cycle_length)

    # Apply the cosine transformation
    df[f'cos_transformation_{period}'] = np.cos(2 * np.pi * df[f'num_col_{period}'] / cycle_length)

    return df


def LinearRegressionTransformation(X_data, yearly_seasonality=False, weekly_seasonality=False):
    # Reset the index of X_train
    X_train_num = X_data.copy().reset_index()

    # Convert X_train to ordinal values
    X_train_num['date_transformation'] = pd.to_datetime(X_train_num['Datum']).map(datetime.datetime.toordinal)
    
    # Apply the sine and cosine transformations
    if yearly_seasonality:
        X_train_num = apply_sin_cos_transformation(df=X_train_num, date_col='Datum', period='year')
    if weekly_seasonality:
        X_train_num = apply_sin_cos_transformation(df=X_train_num, date_col='Datum', period='week')
    # Convert X_train_num to a DataFrame
    # X_train_df = X_train_num.to_frame()

    # Select the columns with the transformations
    X_train_num = X_train_num[[col for col in X_train_num.columns if 'transformation' in col]]
    return X_train_num


def make_dates_datetime(date_vars):

    # Convert all date variables from string to datetime.datetime
    for i, date_var in enumerate(date_vars):
        if isinstance(date_var, str):
            date_vars[i] = datetime.datetime.strptime(date_var, '%Y-%m-%d')

    for i, date_var in enumerate(date_vars):
        if isinstance(date_var, pd.Timestamp):
            date_vars[i] = date_var.to_pydatetime()

    return date_vars


def make_X_y(df, onderwerp, vanaf_datum_train_periode, tot_datum_train_periode, vanaf_datum_test_periode, tot_datum_test_periode):    
    # date_vars = [vanaf_datum_train_periode, tot_datum_train_periode, 
    #          vanaf_datum_test_periode, tot_datum_test_periode]

    datetime_vars = make_dates_datetime(date_vars=[vanaf_datum_train_periode, tot_datum_train_periode, 
             vanaf_datum_test_periode, tot_datum_test_periode])
    vanaf_datum_train_periode, tot_datum_train_periode, \
             vanaf_datum_test_periode, tot_datum_test_periode = datetime_vars
    # # Convert all date variables from string to datetime.datetime
    # for i, date_var in enumerate(date_vars):
    #     if isinstance(date_var, str):
    #         date_vars[i] = datetime.datetime.strptime(date_var, '%Y-%m-%d')

    # for i, date_var in enumerate(date_vars):
    #     if isinstance(date_var, pd.Timestamp):
    #         date_vars[i] = date_var.to_pydatetime()

    # # Unpack the converted date variables
    # vanaf_datum_train_periode, tot_datum_train_periode, \
    # vanaf_datum_test_periode, tot_datum_test_periode = date_vars

    # return vanaf_datum_train_periode, tot_datum_train_periode, \
    # vanaf_datum_test_periode, tot_datum_test_periode

    if tot_datum_train_periode < vanaf_datum_train_periode:
        raise ValueError("Let op: tot_datum_train_periode moet groter zijn dan vanaf_datum_train_periode. Kies andere waarden aub.")
    if vanaf_datum_test_periode < tot_datum_train_periode:
        raise ValueError("Let op: vanaf_datum_test_periode moet groter zijn dan tot_datum_train_periode. Kies andere waarden aub.")
    if tot_datum_test_periode < vanaf_datum_test_periode:
        raise ValueError("Let op: tot_datum_test_periode moet groter zijn dan vanaf_datum_test_periode. Kies andere waarden aub.")
    if tot_datum_train_periode > df.index.max():
        raise ValueError("Let op: tot_datum_train_periode moet kleiner zijn dan de maximale datum in de dataset. Kies andere waarden aub.")
    
    minimum_date = datetime.datetime(2019,1,1)
    maximum_date = datetime.datetime.now() - datetime.timedelta(days=1)
    data = df.loc[(df.index >= minimum_date) & (df.index < maximum_date)].copy()
    if tot_datum_test_periode > data.index.max():
        range_dates_extend = pd.date_range(start=data.index.max() + datetime.timedelta(days=1), 
                                                end=pd.to_datetime(tot_datum_test_periode) - datetime.timedelta(days=1), 
                                                freq='D')
        df_extend = pd.DataFrame({onderwerp: [np.nan]*len(range_dates_extend)}, index=range_dates_extend)
        df_extended = pd.concat([data, df_extend])
    else:
        df_extended = data
    df_extended.index.name = 'Datum'

    # Create a subset of the data for the training period
    df_train = create_subset_df(df=df_extended, start=vanaf_datum_train_periode, end=tot_datum_train_periode)

    # Create a subset of the data for the test period
    df_test = create_subset_df(df=df_extended, start=vanaf_datum_test_periode, end=tot_datum_test_periode)

    # Create the X and y DataFrames for the training data
    df_X_train, df_y_train = create_df_X_and_y(df_train, onderwerp)

    # Create the X and y DataFrames for the test data
    df_X_test, df_y_test = create_df_X_and_y(df_test, onderwerp)

    return df_X_train, df_y_train, df_X_test, df_y_test

def combine_dfs_of_models(list_of_dfs):
    df_combined = pd.DataFrame()
    for df in list_of_dfs:
        if df.empty:
            continue
        else:
            common_cols = df_combined.columns.intersection(df.columns)
            df = df.drop(columns=common_cols)
            df_combined = pd.concat([df_combined, df], axis=1)
    return df_combined

# def plot_timeseries(df, col, date_untill='2024-05-15'):
#     df_plot = df.loc[df.index < date_untill].copy()
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[col], mode='lines', name=col))
#     if col == 'ZZP':
#         y_axis_title = f'Aantal clienten'
#     elif col == 'Ziekteverzuim':
#         y_axis_title = f'Ziekteverzuimpercentage (%)'
#     elif col == 'Inkoop':
#         y_axis_title = f'Kosten inkoop (euro)'
#     elif col == 'Flexpool':
#         y_axis_title = f'Flexpool (aantal personen)'
#     else: 
#         y_axis_title = f'Onbekende eenheid'
#     fig.update_layout(title=f'Weergave van {col} over de tijd', xaxis_title='Datum', yaxis_title=y_axis_title)
#     fig.show()

# def bekijk_data():
#     df = load_timeseries()

#     for col in df.columns:
#         plot_timeseries(df, col)

#     return df