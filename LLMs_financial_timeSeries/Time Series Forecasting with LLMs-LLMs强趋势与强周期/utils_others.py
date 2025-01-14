import copy
import time
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import numpy as np
import matplotlib.font_manager as fm
import seaborn as sns
import os

os.environ['OMP_NUM_THREADS'] = '4'

from models.utils import grid_iter
from models.promptcast import get_promptcast_predictions_data
from models.darts import get_arima_predictions_data
from models.validation_likelihood_tuning import get_autotuned_predictions_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from models import llmtime
from models.llmtime import get_llmtime_predictions_data

import pathlib
import textwrap
import google.generativeai as genai

import os
os.environ['OMP_NUM_THREADS'] = '4'

from data1.serialize import SerializerSettings
from sklearn import metrics
from data1.small_context import get_datasets, get_memorization_datasets, get_dataset
from models.validation_likelihood_tuning import get_autotuned_predictions_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



def plot_preds_w_train_test(train, test, pred_dict, model_name, ds_name, dir='prediction_w_gemini\prediction_w_train+test', show_samples=False):
    """
    Plot predictions with confidence intervals. (Contain both training and test set)

    Parameters:
        train (pd.Series): Time series of training data.
        test (pd.Series): Time series of testing data (ground truth).
        pred_dict (dict): Dictionary containing predictions and other metrics.
        model_name (str): Name of the predictive model.
        ds_name (str): Name of the dataset.
        show_samples (bool): Whether to plot individual samples along with predictions.

    Returns:
        None
    """
    pred = pred_dict['median']
    pred = pd.Series(pred, index=test.index)
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(train)
    plt.plot(test, label='Truth', color='black')
    plt.plot(pred, label=model_name, color='purple')
    # Shade 90% confidence interval
    samples = pred_dict['samples']
    lower = np.quantile(samples, 0.05, axis=0)
    upper = np.quantile(samples, 0.95, axis=0)
    plt.fill_between(pred.index, lower, upper, alpha=0.3, color='purple')
    if show_samples:
        samples = pred_dict['samples']
        # Convert DataFrame to numpy array
        samples = samples.values if isinstance(samples, pd.DataFrame) else samples
        for i in range(min(10, samples.shape[0])):
            plt.plot(pred.index, samples[i], color='purple', alpha=0.3, linewidth=1)
    plt.legend(loc='upper left')
    if 'NLL/D' in pred_dict:
        nll = pred_dict['NLL/D']
        if nll is not None:
            plt.text(0.03, 0.85, f'NLL/D: {nll:.2f}', transform=plt.gca().transAxes,
                     bbox=dict(facecolor='white', alpha=0.5))
    plt.show()
    plt.savefig(dir + f'/{ds_name}_{model_name}_prediction.pdf', format='pdf')


def plot_preds_w_test(test, pred_dict, model_name, ds_name, dir='prediction_w_gemini\prediction_w_test', show_samples=False):
    """
    Plot predictions with confidence intervals, without training data. (Contain only the test set)

    Parameters:
        train (pd.Series): Time series of training data (not plotted).
        test (pd.Series): Time series of testing data (ground truth).
        pred_dict (dict): Dictionary containing predictions and other metrics.
        model_name (str): Name of the predictive model.
        ds_name (str): Name of the dataset.
        show_samples (bool): Whether to plot individual samples along with predictions.

    Returns:
        None
    """
    pred = pred_dict['median']
    pred = pd.Series(pred, index=test.index)
    plt.figure(figsize=(8, 6), dpi=100)
    # Omit plotting training data
    # plt.plot(train)
    plt.plot(test, label='Truth', color='black')
    plt.plot(pred, label=model_name, color='purple')
    # Shade 90% confidence interval
    samples = pred_dict['samples']
    lower = np.quantile(samples, 0.05, axis=0)
    upper = np.quantile(samples, 0.95, axis=0)
    plt.fill_between(pred.index, lower, upper, alpha=0.3, color='purple')
    if show_samples:
        samples = pred_dict['samples']
        # Convert DataFrame to numpy array
        samples = samples.values if isinstance(samples, pd.DataFrame) else samples
        for i in range(min(10, samples.shape[0])):
            plt.plot(pred.index, samples[i], color='purple', alpha=0.3, linewidth=1)
    plt.legend(loc='upper left')
    if 'NLL/D' in pred_dict:
        nll = pred_dict['NLL/D']
        if nll is not None:
            plt.text(0.03, 0.85, f'NLL/D: {nll:.2f}', transform=plt.gca().transAxes,
                     bbox=dict(facecolor='white', alpha=0.5))
    plt.savefig(dir + f'{ds_name}{model_name}_prediction_meticulous.pdf', format='pdf')


def metrics_used(test, dataset_name, original_pred, num_samples=10):
    '''
    This function defines the metrics used for evaluating model performance.

    Args:
        dataset_name (str): Name of the dataset.
        original_pred (dict): Dictionary containing original predictions and samples.
        num_samples (int): Number of samples to consider.

    Returns:
        tuple: Mean values of MSE, MAE, MAPE, and R².
    '''
    print("dataset_name: ", dataset_name)

    mse_amount = 0.0
    mae_amount = 0.0
    mape_amount = 0.0
    rsquare_amount = 0.0
    for i in range(num_samples):
        seq_pred = original_pred[dataset_name]['samples'].iloc[i, :]

        mse = mean_squared_error(test, seq_pred)
        mae = mean_absolute_error(test, seq_pred)
        mape = metrics.mean_absolute_percentage_error(test, seq_pred) * 100
        r2 = r2_score(test, seq_pred)

        mse_amount += mse
        mae_amount += mae
        mape_amount += mape
        rsquare_amount += r2

    mse_mean = mse_amount / num_samples
    mae_mean = mae_amount / num_samples
    mape_mean = mape_amount / num_samples
    r2_mean = rsquare_amount / num_samples

    # Print and plot values
    print("\n")
    print('Calculating metrics for each prediction and taking the mean:')
    print(f'MSE: {mse_mean}, MAE: {mae_mean}, MAPE: {mape_mean}, R²: {r2_mean}')
    print("\n")

    return mse_mean, mae_mean, mape_mean, r2_mean

def fig_length(metric, metric_name='R^2', dir='length_impact', dataset_name='WineDataset'):
    '''
    This function plots a line chart to visualize the length of the training set.

    Args:
        metric (list): List of metric values.
        metric_name (str): Name of the metric to be displayed on the y-axis.
        dataset_name (str): Name of the dataset.

    Returns:
        None
    '''
    font_path = fm.findfont(fm.FontProperties(family='Times New Roman'))
    font_prop = fm.FontProperties(fname=font_path)
    _font_size = 38
    sns.set_style('whitegrid')
    sns.set(style="whitegrid", rc={"axes.grid.axis": "y", "axes.grid": True})
    fig = plt.figure(figsize=(12,8))
    ax1 = plt.gca()
    x = np.arange(len(metric))
    sns.lineplot(x=x, y=metric, color='#d37981', alpha=1, linewidth=5, marker='o', markerfacecolor='w', markeredgecolor='#d37981', markersize=6)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    plt.xlabel("Index", fontproperties=font_prop, fontsize=_font_size+5)
    plt.ylabel(metric_name, fontproperties=font_prop, fontsize=_font_size+5)
    plt.legend(bbox_to_anchor=(0.5, 1.4), loc='upper center', fontsize=_font_size)
    plt.tight_layout()
    plt.savefig(dir + f"Length_of_Training_Set_Analysis_0407_{dataset_name}.png")
    plt.show()


def plot_scatter_2d_list(data):
    '''
    Plot 2D scatter plot from a two-dimensional list.

    Args:
        data (list): Two-dimensional list of data points.

    Returns:
        None
    '''
    m = len(data)  # Length of the first dimension
    n = len(data[0])  # Length of the second dimension

    # Initialize x and y coordinates
    x_coords = []
    y_coords = []

    # Traverse the two-dimensional list to extract x and y coordinates
    for i in range(m):
        for j in range(n):
            x_coords.append(i)  # x-axis corresponds to the first dimension m
            y_coords.append(data[i][j])  # y-axis corresponds to the values in the two-dimensional list

    # Plot scatter plot
    plt.scatter(x_coords, y_coords)


def fig_counterfactual(metric, metric_name='R^2', dataset_name='WineDataset', dir='counterfactual_analysis'):
    '''
    Plot counterfactual analysis figure.

    Args:
        metric (list): Two-dimensional list of metric values.
        metric_name (str): Name of the metric.
        dataset_name (str): Name of the dataset.

    Returns:
        None
    '''
    font_path = fm.findfont(fm.FontProperties(family='Times New Roman'))
    font_prop = fm.FontProperties(fname=font_path)
    _font_size = 38
    sns.set_style('whitegrid')
    sns.set(style="whitegrid", rc={"axes.grid.axis": "y", "axes.grid": True})
    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.gca()
    x = np.arange(len(metric))
    metric_mean = [sum(sublist) / len(sublist) for sublist in metric]

    sns.lineplot(x=x, y=metric_mean, color='#d37981', alpha=1, linewidth=5, marker='o', markerfacecolor='w',
                 markeredgecolor='#d37981', markersize=6)
    lower = np.quantile(metric, 0.05, axis=1)
    upper = np.quantile(metric, 0.95, axis=1)
    plt.fill_between(x, lower, upper, alpha=0.3, color='purple')
    plot_scatter_2d_list(data=metric)

    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    plt.xlabel("Index", fontproperties=font_prop, fontsize=_font_size + 5)
    plt.ylabel(metric_name, fontproperties=font_prop, fontsize=_font_size + 5)
    plt.legend(bbox_to_anchor=(0.5, 1.4), loc='upper center', fontsize=_font_size)
    plt.tight_layout()
    plt.savefig(dir + f"Counterfactual_Analysis_0407_{dataset_name}.png")
    plt.show()


def prediction_gemini(model_predict_fns, train, test, model_hypers, num_samples=10, whether_blanket=False,
                      dataset_name='WineDataset', genai_key=None):
    '''
    Perform Gemini model predictions.

    Args:
        model_predict_fns (dict): Dictionary of model prediction functions.
        train (pd.Series): Time series of training data.
        test (pd.Series): Time series of testing data.
        model_hypers (dict): Dictionary of model hyperparameters.
        num_samples (int): Number of samples for prediction.
        whether_blanket (bool): Whether to use blanket adjustments.
        dataset_name (str): Name of the dataset.

    Returns:
        tuple: Tuple containing dictionaries of Gemini predictions (gemini-pro and gemini-1.0-pro).
    '''
    model_names = list(model_predict_fns.keys())
    out_gemini_pro = {}  # gemini-pro
    out_gemini_pro_number = {}  # gemini-1.0-pro
    for model in model_names:
        model_hypers[model].update({'dataset_name': dataset_name})  # Add dataset_name to hyperparameters
        hypers = list(grid_iter(model_hypers[model]))  # Generate hyperparameter combinations

        pred_dict = get_autotuned_predictions_data(train, test, hypers, num_samples, model_predict_fns[model],
                                                   verbose=False, parallel=False, whether_blanket=whether_blanket, genai_key=genai_key)
        # This part of the code is yet to be verified, not confirmed if it can run;
        # Automatically autotune hyperparameters based on validation likelihood, as mentioned in the original text (better for trainable models)
        if model == 'gemini-pro':
            out_gemini_pro.update({dataset_name: pred_dict})
        if model == 'gemini-1.0-pro':
            out_gemini_pro_number.update({dataset_name: pred_dict})
    return out_gemini_pro, out_gemini_pro_number

def opt_hyper_gemini(model_predict_fns, train, test, model_hypers, num_samples=10, whether_blanket=False,
                      dataset_name='WineDataset', genai_key=None, temp_list=[0.2, 0.4, 0.6, 0.8, 1.0], prec_list=[2,3]):
    # 创建一个包含序号 i 的字典的列表
    gemini_hypers_list_0 = [{f'temp': temp_val} for i, temp_val in zip(range(len(temp_list)), temp_list)]
    gemini_hypers_list = []

    output_metrics = []

    for prec in prec_list:
        gemini_hypers_list_tmp = copy.deepcopy(gemini_hypers_list_0)
        for dict in gemini_hypers_list_tmp:
            dict.update({
                'alpha': 0.95,
                'beta': 0.3,
                'basic': [False],
                'settings': [SerializerSettings(base=10, prec=prec, signed=True, half_bin_correction=True)],
            })
        gemini_hypers_list.extend(gemini_hypers_list_tmp)

    for index, dict in enumerate(gemini_hypers_list):
        if index > 0:
            time.sleep(60)
        out_gemini_pro, out_gemini_pro_number = prediction_gemini(model_predict_fns=model_predict_fns, train=train, test=test, model_hypers=model_hypers, num_samples=num_samples, whether_blanket=whether_blanket,
                      dataset_name=dataset_name, genai_key=genai_key)
        mse_mean, mae_mean, mape_mean, r2_mean = metrics_used(test=test, dataset_name=dataset_name, original_pred=out_gemini_pro_number, num_samples=num_samples)
        dict.update({'mse': mse_mean, 'mae': mae_mean, 'mape': mape_mean, 'r2': r2_mean})
        output_metrics.append(dict)

    return output_metrics