import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from datetime import datetime

from task_2_3 import (
    rolling_window_analysis,
    expanding_window_analysis
)


def calculate_efficient_frontier(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    n_points: int = 100,
    min_return: float = None,
    max_return: float = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Расчет эффективной границы без ограничений на короткие продажи.

    Parameters:
    -----------
    mean_returns : np.ndarray
        Вектор средних доходностей
    cov_matrix : np.ndarray
        Ковариационная матрица
    n_points : int
        Количество точек на эффективной границе
    min_return : float, optional
        Минимальная доходность для построения границы. Если None, используется GMVP
    max_return : float, optional
        Максимальная доходность для построения границы. Если None, используется макс. доходность отдельного актива

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (массив доходностей, массив стандартных отклонений)
    """
    n_assets = len(mean_returns)

    try:
        L = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        # Добавление небольшой регуляризации
        cov_matrix = cov_matrix + np.eye(n_assets) * 1e-8
        L = np.linalg.cholesky(cov_matrix)

    cov_inv = np.linalg.inv(cov_matrix)


    ones = np.ones(n_assets)
    A = ones.T @ cov_inv @ ones
    B = mean_returns.T @ cov_inv @ mean_returns
    C = mean_returns.T @ cov_inv @ ones
    D = A * B - C ** 2


    w_gmvp = cov_inv @ ones / A
    gmvp_return = w_gmvp.T @ mean_returns

    
    if min_return is None:
        min_return = gmvp_return

    if max_return is None:
        max_return = np.max(mean_returns)

    
    target_returns = np.linspace(min_return, max_return, n_points)
    variances = (A * target_returns ** 2 - 2 * C * target_returns + B) / D
    std_devs = np.sqrt(variances)

    return target_returns, std_devs


def calculate_efficient_frontier_weights(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    target_returns: np.ndarray
) -> np.ndarray:
    """
    Расчет оптимальных весов для заданных уровней доходности.

    Parameters:
    -----------
    mean_returns : np.ndarray
        Вектор средних доходностей
    cov_matrix : np.ndarray
        Ковариационная матрица
    target_returns : np.ndarray
        Массив целевых доходностей

    Returns:
    --------
    np.ndarray
        Матрица весов (n_points x n_assets)
    """
    n_assets = len(mean_returns)
    cov_inv = np.linalg.inv(cov_matrix)

    
    ones = np.ones(n_assets)
    A = ones.T @ cov_inv @ ones
    B = mean_returns.T @ cov_inv @ mean_returns
    C = mean_returns.T @ cov_inv @ ones
    D = A * B - C ** 2

    
    g = (B * cov_inv @ ones - C * cov_inv @ mean_returns) / D
    h = (C * cov_inv @ ones - A * cov_inv @ mean_returns) / D


    weights = np.zeros((len(target_returns), n_assets))
    for i, r in enumerate(target_returns):
        weights[i, :] = g + h * r

    return weights


def calculate_efficient_frontiers_over_time(
    analysis_results: Dict[datetime, dict],
    n_points: int = 100
) -> Dict[datetime, dict]:
    """
    Расчет эффективных границ для каждого окна анализа.

    Parameters:
    -----------
    analysis_results : Dict[datetime, dict]
        Результаты анализа скользящим или расширяющимся окном
    n_points : int
        Количество точек на каждой эффективной границе

    Returns:
    --------
    Dict[datetime, dict]
        Словарь с эффективными границами для каждой даты
    """
    frontiers = {}

    for date, result in analysis_results.items():
        mean_returns = result['mean_returns']
        cov_matrix = result['covariance_matrix']

        returns, stds = calculate_efficient_frontier(mean_returns, cov_matrix, n_points)

        weights = calculate_efficient_frontier_weights(mean_returns, cov_matrix, returns)

        frontiers[date] = {
            'returns': returns,
            'stds': stds,
            'weights': weights,
            'min_std': stds[0],
            'min_std_return': returns[0],
            'max_return': returns[-1],
            'max_return_std': stds[-1]
        }

    return frontiers


def analyze_efficient_frontier_stability(
    frontiers: Dict[datetime, dict]
) -> pd.DataFrame:
    """
    Анализ стабильности эффективных границ во времени.

    Parameters:
    -----------
    frontiers : Dict[datetime, dict]
        Словарь с эффективными границами

    Returns:
    --------
    pd.DataFrame
        DataFrame с метриками стабильности
    """
    dates = sorted(frontiers.keys())

    metrics = []

    for date in dates:
        frontier = frontiers[date]

        sharpe_ratios = frontier['returns'] / frontier['stds']
        max_sharpe = np.max(sharpe_ratios)
        max_sharpe_idx = np.argmax(sharpe_ratios)

        metrics.append({
            'date': date,
            'min_std': frontier['min_std'],
            'min_std_return': frontier['min_std_return'],
            'max_return': frontier['max_return'],
            'max_return_std': frontier['max_return_std'],
            'max_sharpe': max_sharpe,
            'max_sharpe_return': frontier['returns'][max_sharpe_idx],
            'max_sharpe_std': frontier['stds'][max_sharpe_idx],
            'efficiency_ratio': frontier['max_return'] / frontier['min_std'],
            'frontier_range': frontier['max_return'] - frontier['min_std_return']
        })

    df = pd.DataFrame(metrics)
    df.set_index('date', inplace=True)

    if len(df) > 1:
        df['min_std_change'] = df['min_std'].diff()
        df['min_std_return_change'] = df['min_std_return'].diff()
        df['max_return_change'] = df['max_return'].diff()
        df['max_sharpe_change'] = df['max_sharpe'].diff()

    return df


def analyze_portfolio_composition_stability(
    frontiers: Dict[datetime, dict],
    asset_names: List[str],
    percentile: float = 50
) -> pd.DataFrame:
    """
    Анализ стабильности состава портфелей во времени.

    Parameters:
    -----------
    frontiers : Dict[datetime, dict]
        Словарь с эффективными границами
    asset_names : List[str]
        Названия активов
    percentile : float
        Перцентиль для выбора портфеля на границе (0-100)

    Returns:
    --------
    pd.DataFrame
        DataFrame с весами портфелей во времени
    """
    dates = sorted(frontiers.keys())

    weights_data = []

    for date in dates:
        frontier = frontiers[date]

        idx = int(len(frontier['returns']) * percentile / 100)
        weights = frontier['weights'][idx, :]

        weights_dict = {f'w_{asset}': w for asset, w in zip(asset_names, weights)}
        weights_dict['date'] = date
        weights_dict['portfolio_return'] = frontier['returns'][idx]
        weights_dict['portfolio_std'] = frontier['stds'][idx]

        weights_data.append(weights_dict)

    df = pd.DataFrame(weights_data)
    df.set_index('date', inplace=True)

    if len(df) > 1:
        weight_cols = [col for col in df.columns if col.startswith('w_')]
        for col in weight_cols:
            df[f'{col}_change'] = df[col].diff()

        df['total_weight_change'] = df[[f'{col}_change' for col in weight_cols]].abs().sum(axis=1)

    return df


def efficient_frontier_dynamics_rolling(
    returns: pd.DataFrame,
    window_size: str = '1Y',
    step_size: str = '1Y',
    n_points: int = 100
) -> Tuple[Dict[datetime, dict], pd.DataFrame]:
    """
    Задача 9a: Динамика эффективной границы скользящим окном.

    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame с доходностями
    window_size : str
        Размер окна
    step_size : str
        Размер шага
    n_points : int
        Количество точек на эффективной границе

    Returns:
    --------
    Tuple[Dict[datetime, dict], pd.DataFrame]
        (словарь эффективных границ, DataFrame с метриками стабильности)
    """
    analysis_results = rolling_window_analysis(returns, window_size, step_size)
    frontiers = calculate_efficient_frontiers_over_time(analysis_results, n_points)
    stability_metrics = analyze_efficient_frontier_stability(frontiers)
    return frontiers, stability_metrics


def efficient_frontier_dynamics_expanding(
    returns: pd.DataFrame,
    step_size: str = '1Y',
    n_points: int = 100
) -> Tuple[Dict[datetime, dict], pd.DataFrame]:
    """
    Задача 9b: Динамика эффективной границы расширяющимся окном.

    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame с доходностями
    step_size : str
        Размер шага
    n_points : int
        Количество точек на эффективной границе

    Returns:
    --------
    Tuple[Dict[datetime, dict], pd.DataFrame]
        (словарь эффективных границ, DataFrame с метриками стабильности)
    """

    analysis_results = expanding_window_analysis(returns, step_size)
    frontiers = calculate_efficient_frontiers_over_time(analysis_results, n_points)
    stability_metrics = analyze_efficient_frontier_stability(frontiers)

    return frontiers, stability_metrics


def efficient_frontier_dynamics_exponential(
    returns: pd.DataFrame,
    window_size: str = '1Y',
    step_size: str = '1Y',
    lambda_param: float = 0.94,
    n_points: int = 100
) -> Tuple[Dict[datetime, dict], pd.DataFrame]:
    """
    Задача 10: Динамика эффективной границы с экспоненциальным забыванием.

    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame с доходностями
    window_size : str
        Размер окна (для скользящего окна)
    step_size : str
        Размер шага
    lambda_param : float
        Параметр экспоненциального забывания
    n_points : int
        Количество точек на эффективной границе

    Returns:
    --------
    Tuple[Dict[datetime, dict], pd.DataFrame]
        (словарь эффективных границ, DataFrame с метриками стабильности)
    """

    analysis_results = rolling_window_analysis(
        returns, window_size, step_size, lambda_param
    )
    frontiers = calculate_efficient_frontiers_over_time(analysis_results, n_points)
    stability_metrics = analyze_efficient_frontier_stability(frontiers)

    return frontiers, stability_metrics


def exp_efficient_frontier_dynamics_expanding(
    returns: pd.DataFrame,
    step_size: str = '1Y',
    lambda_param: float = 0.94,
    n_points: int = 100
) -> Tuple[Dict[datetime, dict], pd.DataFrame]:
    """
    Задача 10b: Динамика эффективной границы расширяющимся окном с экспоненциальным забыванием.

    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame с доходностями
    step_size : str
        Размер шага
    lambda_param : float
        Параметр экспоненциального забывания
    n_points : int
        Количество точек на эффективной границе

    Returns:
    --------
    Tuple[Dict[datetime, dict], pd.DataFrame]
        (словарь эффективных границ, DataFrame с метриками стабильности)
    """

    analysis_results = expanding_window_analysis(returns, step_size, lambda_param)
    frontiers = calculate_efficient_frontiers_over_time(analysis_results, n_points)
    stability_metrics = analyze_efficient_frontier_stability(frontiers, returns.columns)
    
    return frontiers, stability_metrics


def compare_frontier_methods(
    results_list: List[Tuple[str, Dict[datetime, dict]]]
) -> pd.DataFrame:
    """
    Сравнение эффективных границ для разных методов.

    Parameters:
    -----------
    results_list : List[Tuple[str, Dict[datetime, dict]]]
        Список кортежей (название метода, словарь границ)

    Returns:
    --------
    pd.DataFrame
        DataFrame с сравнительными метриками
    """
    comparison_data = []

    for method_name, frontiers in results_list:
        dates = sorted(frontiers.keys())

        for date in dates:
            frontier = frontiers[date]

            sharpe_ratios = frontier['returns'] / frontier['stds']
            max_sharpe = np.max(sharpe_ratios)

            comparison_data.append({
                'method': method_name,
                'date': date,
                'min_std': frontier['min_std'],
                'min_std_return': frontier['min_std_return'],
                'max_return': frontier['max_return'],
                'max_return_std': frontier['max_return_std'],
                'max_sharpe': max_sharpe,
                'efficiency_ratio': frontier['max_return'] / frontier['min_std']
            })

    df = pd.DataFrame(comparison_data)
    df.set_index(['method', 'date'], inplace=True)

    return df
