import datetime as datetime
import pandas as pd
from scripts.evaluate.evaluate import plot_prediction_with_shapes
from scripts.preprocess.preprocess import load_timeseries_data, collect_str_input, make_X_y, combine_dfs_of_models
from scripts.evaluate.evaluate import plot_timeseries, plot_errors, plot_distribution, calc_r2_score, calc_sum_of_errors, calc_average_error, calc_max_error, calc_MAE, calc_MAPE, calc_RMSE
from scripts.model.model import calc_average, predict_with_average, calc_moving_average, LinearRegressionTrain, LinearRegressionPredict

def bekijk_data():
    df = load_timeseries_data()

    for col in df.columns:
        plot_timeseries(df, col)

    return df


def kies_onderwerp():
    vraag = """
        Welke reeks wil je voorspellen (a, b, c):
                a. ZZP (verwachting clienten)
                b. Ziekteverzuimpercentage
                c. Inkoop materialen
                d. Flexpool (aantal personen)
        """
    mogelijke_antwoorden = ['a', 'b', 'c', 'd']

    str = collect_str_input(
        question=vraag, 
        possible_entries=mogelijke_antwoorden)
    dict_antwoorden = {'a': 'ZZP', 'b': 'Ziekteverzuim', 'c': 'Inkoop', 'd': 'Flexpool'}
    print(f'Gekozen antwoord: {dict_antwoorden[str]}')
    return dict_antwoorden[str]

def pas_gemiddelde_toe(df, 
                     onderwerp, 
                     vanaf_datum_train_periode = '2019-01-01',
                     tot_datum_train_periode = '2023-05-15',
                     vanaf_datum_test_periode = '2023-05-15',
                     tot_datum_test_periode = datetime.datetime.now().strftime('%Y-%m-%d')):
    df_X_train, df_y_train, df_X_test, df_y_test = make_X_y(df, 
                                                            onderwerp, 
                                                            vanaf_datum_train_periode, 
                                                            tot_datum_train_periode, 
                                                            vanaf_datum_test_periode, 
                                                            tot_datum_test_periode)

    # Apply average model
    # Calculate the average of the train data
    average = calc_average(df_y_train)
    # Predict the average for the train data
    y_preds_train_avg = predict_with_average(df_y_train, average)
    # Predict the average for the test data
    y_preds_test_avg = predict_with_average(df_y_test, average)
    y_avg = pd.concat([y_preds_train_avg, y_preds_test_avg])

    df_y_all = pd.DataFrame()
    df_y_all['Gemiddelde'] = y_avg

    df_real = pd.concat([df_y_train, df_y_test], axis=0)
    df_real.name = onderwerp
    df_total = pd.concat([df_real, df_y_all], axis=1)

    plot_prediction_with_shapes(df=df_total, 
                                start_train=vanaf_datum_train_periode, 
                                end_train=tot_datum_train_periode, 
                                start_test=vanaf_datum_test_periode, 
                                end_test=tot_datum_test_periode, 
                                shapes=True)

    return df_total

def pas_voortschrijdend_gemiddelde_toe(df, 
                     onderwerp, 
                     vanaf_datum_train_periode = '2019-01-01',
                     tot_datum_train_periode = '2023-05-15',
                     vanaf_datum_test_periode = '2023-05-15',
                     tot_datum_test_periode = datetime.datetime.now().strftime('%Y-%m-%d'),
                     window_size = 7,
                     shift_period = 365,
                     predict=False):
    
    df_X_train, df_y_train, df_X_test, df_y_test = make_X_y(df, 
                                                            onderwerp, 
                                                            vanaf_datum_train_periode, 
                                                            tot_datum_train_periode, 
                                                            vanaf_datum_test_periode, 
                                                            tot_datum_test_periode)

    # Apply moving average model
    # Calculate the moving average of the training data
    df_y_train_test = pd.concat([df_y_train, df_y_test])
    y_preds_train_mov_avg = calc_moving_average(df_y_train_test, 
                                                window_size=window_size, 
                                                shift_period=shift_period, 
                                                predict_to_date=tot_datum_test_periode,
                                                predict=predict)

    df_y_all = pd.DataFrame()
    df_y_all['Voortschrijdend gemiddelde'] = y_preds_train_mov_avg

    df_real = pd.concat([df_y_train, df_y_test], axis=0)
    df_real.name = onderwerp
    df_total = pd.concat([df_real, df_y_all], axis=1)

    plot_prediction_with_shapes(df=df_total, 
                                start_train=vanaf_datum_train_periode, 
                                end_train=tot_datum_train_periode, 
                                start_test=vanaf_datum_test_periode, 
                                end_test=tot_datum_test_periode, 
                                shapes=True)

    return df_total

def pas_lineaire_regressie_toe(df, 
                     onderwerp, 
                     vanaf_datum_train_periode = '2019-01-01',
                     tot_datum_train_periode = '2023-05-15',
                     vanaf_datum_test_periode = '2023-05-15',
                     tot_datum_test_periode = datetime.datetime.now().strftime('%Y-%m-%d'),
                     yearly_seasonality=False,
                     weekly_seasonality=False,
                     transformation=None, n_bins=4, strategy='uniform', n_knots=10, degree=3):
    
    df_X_train, df_y_train, df_X_test, df_y_test = make_X_y(df, 
                                                            onderwerp, 
                                                            vanaf_datum_train_periode, 
                                                            tot_datum_train_periode, 
                                                            vanaf_datum_test_periode, 
                                                            tot_datum_test_periode)

    # Apply Linear Regression model
    # Get Linear Regression model
    model = LinearRegressionTrain(df_X_train, 
                                  df_y_train, 
                                  yearly_seasonality=yearly_seasonality, 
                                  weekly_seasonality=weekly_seasonality,
                                  transformation=transformation, n_bins=n_bins, strategy=strategy, n_knots=n_knots, degree=degree)
    # Make predictions with the Linear Regression model for the train data
    y_preds_train_lin_reg = LinearRegressionPredict(df_X_train, 
                                                    model, 
                                                    yearly_seasonality=yearly_seasonality, 
                                                    weekly_seasonality=weekly_seasonality)
    # Make predictions with the Linear Regression model for the test data
    y_preds_test_lin_reg = LinearRegressionPredict(df_X_test, 
                                                   model, 
                                                   yearly_seasonality=yearly_seasonality, 
                                                   weekly_seasonality=weekly_seasonality)
    y_lin_reg = pd.concat([y_preds_train_lin_reg, y_preds_test_lin_reg])

    df_real = pd.concat([df_y_train, df_y_test], axis=0)
    df_real.name = onderwerp

    df_y_all = pd.DataFrame()
    df_y_all['Lineaire regressie'] = y_lin_reg
    df_y_all.index = df_real.index
    df_total = pd.concat([df_real, df_y_all], axis=1)

    plot_prediction_with_shapes(df=df_total, 
                                start_train=vanaf_datum_train_periode, 
                                end_train=tot_datum_train_periode, 
                                start_test=vanaf_datum_test_periode, 
                                end_test=tot_datum_test_periode, 
                                shapes=True)

    return df_total


def pas_modellen_toe(df, 
                     onderwerp, 
                     vanaf_datum_train_periode = '2019-01-01',
                     tot_datum_train_periode = '2023-05-15',
                     vanaf_datum_test_periode = '2023-05-15',
                     tot_datum_test_periode = datetime.datetime.now().strftime('%Y-%m-%d'),
                     window_size = 7,
                     shift_period = 365,
                     yearly_seasonality=False,
                     weekly_seasonality=False,
                     transformation=None, n_bins=4, strategy='uniform', n_knots=10, degree=3,
                     predict=False):
    
    df_X_train, df_y_train, df_X_test, df_y_test = make_X_y(df, 
                                                            onderwerp, 
                                                            vanaf_datum_train_periode, 
                                                            tot_datum_train_periode, 
                                                            vanaf_datum_test_periode, 
                                                            tot_datum_test_periode)

    # Apply average model
    # Calculate the average of the train data
    average = calc_average(df_y_train)
    # Predict the average for the train data
    y_preds_train_avg = predict_with_average(df_y_train, average)
    # Predict the average for the test data
    y_preds_test_avg = predict_with_average(df_y_test, average)
    y_avg = pd.concat([y_preds_train_avg, y_preds_test_avg])
    
    # Apply moving average model
    # Calculate the moving average of the training data
    df_y_train_test = pd.concat([df_y_train, df_y_test])
    y_mov_avg = calc_moving_average(df_y_train_test, 
                                                window_size=window_size, 
                                                shift_period=shift_period, 
                                                predict_to_date=tot_datum_test_periode,
                                                predict=predict)

    # Apply Linear Regression model
    # Get Linear Regression model
    model = LinearRegressionTrain(df_X_train, 
                                  df_y_train, 
                                  yearly_seasonality=yearly_seasonality, 
                                  weekly_seasonality=weekly_seasonality,
                                  transformation=transformation, n_bins=n_bins, strategy=strategy, n_knots=n_knots, degree=degree)
    # Make predictions with the Linear Regression model for the train data
    y_preds_train_lin_reg = LinearRegressionPredict(df_X_train, 
                                                    model, 
                                                    yearly_seasonality=yearly_seasonality, 
                                                    weekly_seasonality=weekly_seasonality)
    # Make predictions with the Linear Regression model for the test data
    y_preds_test_lin_reg = LinearRegressionPredict(df_X_test, 
                                                   model, 
                                                   yearly_seasonality=yearly_seasonality, 
                                                   weekly_seasonality=weekly_seasonality)
    y_lin_reg = pd.concat([y_preds_train_lin_reg, y_preds_test_lin_reg])
    
    df_real = pd.concat([df_y_train, df_y_test], axis=0)
    df_real.name = onderwerp

    df_y_all = pd.DataFrame()
    df_y_all['Gemiddelde'] = y_avg
    df_y_all['Voortschrijdend gemiddelde'] = y_mov_avg
    df_y_all['Lineaire regressie'] = y_lin_reg

    
    df_total = pd.concat([df_real, df_y_all], axis=1)

    plot_prediction_with_shapes(df=df_total, 
                                start_train=vanaf_datum_train_periode, 
                                end_train=tot_datum_train_periode, 
                                start_test=vanaf_datum_test_periode, 
                                end_test=tot_datum_test_periode, 
                                shapes=True)

    return df_total

def onderzoek_afwijkingen(list_of_dfs, onderwerp, start=None, end=None, show='errors'):
    df = combine_dfs_of_models(list_of_dfs)
    if start is None:
        start = df.index.min()
    if end is None:
        end = df.index.max()
    if show == 'errors':
        plot_errors(df=df, base_col=onderwerp, start=start, end=end)
    elif show == 'distribution':
        plot_distribution(df=df, base_col=onderwerp, start=start, end=end)
    elif show == 'both':
        plot_errors(df=df, base_col=onderwerp, start=start, end=end)
        plot_distribution(df=df, base_col=onderwerp, start=start, end=end)
    else:
        raise ValueError("Kies een van de volgende opties: 'errors', 'distribution' of 'both'.")


def bereken_metrieken(list_of_dfs, onderwerp, start, end, list_metrics=[calc_r2_score, calc_MAE]):
    """_summary_

    Parameters
    ----------
    list_of_dfs : _type_
        _description_
    onderwerp : _type_
        _description_
    start : _type_
        _description_
    end : _type_
        _description_
    list_metrics : _type_
        _description_[calc_r2_score, calc_sum_of_errors, calc_average_error, calc_max_error, calc_MAE, calc_MAPE, calc_RMSE]

    Returns
    -------
    _type_
        _description_
    """
    df = combine_dfs_of_models(list_of_dfs)
    _df = df.loc[start:end].copy()
    print(f"De periode die wordt geanalyseerd is van {start} tot {end.date()}.")
    df_metrics = pd.DataFrame()
    metric_cols = [col for col in _df.columns if onderwerp not in col]

    for col in metric_cols:
        for metric in list_metrics:
            df_metrics.loc[col, metric.__name__] = metric(_df[onderwerp], _df[col])
    df_metrics.rename(columns={
        'calc_r2_score': 'R2-score',
        'calc_sum_of_errors': 'Totale afwijking', 
        'calc_average_error': 'Gemiddelde afwijking', 
        'calc_max_error': 'Maximale afwijking',
        'calc_MAE': 'Mean Absolute Error', 
        'calc_MAPE': 'Mean Absolute Percentage Error', 
        'calc_RMSE': 'Root Mean Squared Error'}, 
        inplace=True)
    return df_metrics

def pas_parameters_toe_en_evalueer(df, 
                     onderwerp, 
                     vanaf_datum_train_periode = '2019-01-01',
                     tot_datum_train_periode = '2023-05-15',
                     vanaf_datum_test_periode = '2023-05-15',
                     tot_datum_test_periode = datetime.datetime.now().strftime('%Y-%m-%d'),
                     window_size = 7,
                     shift_period = 365,
                     yearly_seasonality=False,
                     weekly_seasonality=False,
                     transformation=None, n_bins=4, strategy='uniform', n_knots=10, degree=3,
                     predict=False,
                     shapes=True):
    
    df_X_train, df_y_train, df_X_test, df_y_test = make_X_y(df, 
                                                            onderwerp, 
                                                            vanaf_datum_train_periode, 
                                                            tot_datum_train_periode, 
                                                            vanaf_datum_test_periode, 
                                                            tot_datum_test_periode)

    # Apply average model
    # Calculate the average of the train data
    average = calc_average(df_y_train)
    # Predict the average for the train data
    y_preds_train_avg = predict_with_average(df_y_train, average)
    # Predict the average for the test data
    y_preds_test_avg = predict_with_average(df_y_test, average)
    y_avg = pd.concat([y_preds_train_avg, y_preds_test_avg])
    
    # Apply moving average model
    # Calculate the moving average of the training data
    df_y_train_test = pd.concat([df_y_train, df_y_test])
    y_mov_avg = calc_moving_average(df_y_train_test, 
                                                window_size=window_size, 
                                                shift_period=shift_period, 
                                                predict_to_date=tot_datum_test_periode,
                                                predict=predict)

    # Apply Linear Regression model
    # Get Linear Regression model
    model = LinearRegressionTrain(df_X_train, 
                                  df_y_train, 
                                  yearly_seasonality=yearly_seasonality, 
                                  weekly_seasonality=weekly_seasonality,
                                  transformation=transformation, n_bins=n_bins, strategy=strategy, n_knots=n_knots, degree=degree)
    # Make predictions with the Linear Regression model for the train data
    y_preds_train_lin_reg = LinearRegressionPredict(df_X_train, 
                                                    model, 
                                                    yearly_seasonality=yearly_seasonality, 
                                                    weekly_seasonality=weekly_seasonality)
    # Make predictions with the Linear Regression model for the test data
    y_preds_test_lin_reg = LinearRegressionPredict(df_X_test, 
                                                   model, 
                                                   yearly_seasonality=yearly_seasonality, 
                                                   weekly_seasonality=weekly_seasonality)
    y_lin_reg = pd.concat([y_preds_train_lin_reg, y_preds_test_lin_reg])
    
    df_real = pd.concat([df_y_train, df_y_test], axis=0)
    df_real.name = onderwerp

    df_y_all = pd.DataFrame()
    df_y_all['Gemiddelde'] = y_avg
    df_y_all['Voortschrijdend gemiddelde'] = y_mov_avg
    df_y_all['Lineaire regressie'] = y_lin_reg

    
    df_total = pd.concat([df_real, df_y_all], axis=1)

    plot_prediction_with_shapes(df=df_total, 
                                start_train=vanaf_datum_train_periode, 
                                end_train=tot_datum_train_periode, 
                                start_test=vanaf_datum_test_periode, 
                                end_test=tot_datum_test_periode, 
                                shapes=shapes)

    return df_total

def voorspel(df, 
                onderwerp, 
                vanaf_datum_train_periode = '2019-01-01',
                tot_datum_train_periode = '2023-05-15',
                vanaf_datum_test_periode = '2023-05-15',
                tot_datum_test_periode = datetime.datetime.now().strftime('%Y-%m-%d'),
                window_size = 7,
                shift_period = 365,
                yearly_seasonality=False,
                weekly_seasonality=False,
                transformation=None, n_bins=4, strategy='uniform', n_knots=10, degree=3,
                predict=False,
                shapes=False):

    df_voorspel = pas_parameters_toe_en_evalueer(df=df, 
                onderwerp=onderwerp, 
                vanaf_datum_train_periode = vanaf_datum_train_periode,
                tot_datum_train_periode = tot_datum_train_periode,
                vanaf_datum_test_periode = vanaf_datum_test_periode,
                tot_datum_test_periode = tot_datum_test_periode,
                window_size = window_size,
                shift_period = shift_period,
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                transformation=transformation, n_bins=n_bins, strategy=strategy, n_knots=n_knots, degree=degree,
                predict=predict,
                shapes=shapes)
    return df_voorspel