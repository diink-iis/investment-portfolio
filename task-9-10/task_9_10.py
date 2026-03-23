"""
Задача 9-10: Анализ стабильности во времени границы эффективных портфелей

Задача 9:
a. Продемонстрировать динамику изменения границы эффективных портфелей (без ограничений)
   скользящим окном с шагом в 1 год
b. То же самое расширяющимся окном с шагом в 1 год

Задача 10: Выполнить задачу 9 для схемы взвешивания наблюдений с экспоненциальным забыванием
"""

import numpy as np
import pandas as pd
import sys
from typing import Dict, Tuple, List
from datetime import datetime

# Импорт функций из task_2_3
sys.path.insert(0, '../task-2-3')
from task_2_3 import (
    load_prices_data,
    calculate_returns,
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

    # Проверка положительной определенности ковариационной матрицы
    try:
        L = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        # Добавление небольшой регуляризации
        cov_matrix = cov_matrix + np.eye(n_assets) * 1e-8
        L = np.linalg.cholesky(cov_matrix)

    # Вычисление матрицы, обратной ковариационной
    cov_inv = np.linalg.inv(cov_matrix)

    # Вычисление коэффициентов A, B, C, D
    ones = np.ones(n_assets)
    A = ones.T @ cov_inv @ ones
    B = mean_returns.T @ cov_inv @ mean_returns
    C = mean_returns.T @ cov_inv @ ones
    D = A * B - C ** 2

    # Портфель с минимальной дисперсией (GMVP)
    w_gmvp = cov_inv @ ones / A
    min_variance_gmvp = w_gmvp.T @ cov_matrix @ w_gmvp
    gmvp_return = w_gmvp.T @ mean_returns

    # Определение диапазона доходностей
    if min_return is None:
        min_return = gmvp_return

    if max_return is None:
        max_return = np.max(mean_returns)

    # Генерация доходностей для построения границы
    target_returns = np.linspace(min_return, max_return, n_points)

    # Расчет дисперсий для каждой целевой доходности
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

    # Коэффициенты
    ones = np.ones(n_assets)
    A = ones.T @ cov_inv @ ones
    B = mean_returns.T @ cov_inv @ mean_returns
    C = mean_returns.T @ cov_inv @ ones
    D = A * B - C ** 2

    # Векторы g и h
    g = (B * cov_inv @ ones - C * cov_inv @ mean_returns) / D
    h = (C * cov_inv @ ones - A * cov_inv @ mean_returns) / D

    # Расчет весов для каждой целевой доходности
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

        # Расчет эффективной границы
        returns, stds = calculate_efficient_frontier(mean_returns, cov_matrix, n_points)

        # Расчет весов для портфелей
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
    frontiers: Dict[datetime, dict],
    asset_names: List[str]
) -> pd.DataFrame:
    """
    Анализ стабильности эффективных границ во времени.

    Parameters:
    -----------
    frontiers : Dict[datetime, dict]
        Словарь с эффективными границами
    asset_names : List[str]
        Названия активов

    Returns:
    --------
    pd.DataFrame
        DataFrame с метриками стабильности
    """
    dates = sorted(frontiers.keys())

    # Метрики для анализа
    metrics = []

    for date in dates:
        frontier = frontiers[date]

        # Шарп-отношение для портфелей (при безрисковой ставке = 0)
        sharpe_ratios = frontier['returns'] / frontier['stds']
        max_sharpe = np.max(sharpe_ratios)
        max_sharpe_idx = np.argmax(sharpe_ratios)

        # Ковариация весов между смежными периодами
        weights = frontier['weights']

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

    # Добавляем метрики изменения между периодами
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

        # Выбор портфеля на границе по перцентилю
        idx = int(len(frontier['returns']) * percentile / 100)
        weights = frontier['weights'][idx, :]

        weights_dict = {f'w_{asset}': w for asset, w in zip(asset_names, weights)}
        weights_dict['date'] = date
        weights_dict['portfolio_return'] = frontier['returns'][idx]
        weights_dict['portfolio_std'] = frontier['stds'][idx]

        weights_data.append(weights_dict)

    df = pd.DataFrame(weights_data)
    df.set_index('date', inplace=True)

    # Расчет изменения весов между периодами
    if len(df) > 1:
        weight_cols = [col for col in df.columns if col.startswith('w_')]
        for col in weight_cols:
            df[f'{col}_change'] = df[col].diff()

        # Сумма абсолютных изменений весов (мера нестабильности)
        df['total_weight_change'] = df[[f'{col}_change' for col in weight_cols]].abs().sum(axis=1)

    return df


def task_9a_efficient_frontier_dynamics_rolling(
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
    print(f"Задача 9a: Анализ эффективной границы скользящим окном ({window_size}, шаг {step_size})...")

    # Анализ скользящим окном
    analysis_results = rolling_window_analysis(returns, window_size, step_size)
    print(f"Получено окон: {len(analysis_results)}")

    # Расчет эффективных границ для каждого окна
    frontiers = calculate_efficient_frontiers_over_time(analysis_results, n_points)
    print(f"Рассчитано эффективных границ: {len(frontiers)}")

    # Анализ стабильности
    stability_metrics = analyze_efficient_frontier_stability(frontiers, returns.columns)
    print(f"Рассчитаны метрики стабильности")

    return frontiers, stability_metrics


def task_9b_efficient_frontier_dynamics_expanding(
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
    print(f"Задача 9b: Анализ эффективной границы расширяющимся окном (шаг {step_size})...")

    # Анализ расширяющимся окном
    analysis_results = expanding_window_analysis(returns, step_size)
    print(f"Получено окон: {len(analysis_results)}")

    # Расчет эффективных границ для каждого окна
    frontiers = calculate_efficient_frontiers_over_time(analysis_results, n_points)
    print(f"Рассчитано эффективных границ: {len(frontiers)}")

    # Анализ стабильности
    stability_metrics = analyze_efficient_frontier_stability(frontiers, returns.columns)
    print(f"Рассчитаны метрики стабильности")

    return frontiers, stability_metrics


def task_10_efficient_frontier_dynamics_exponential(
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
    print(f"Задача 10: Анализ эффективной границы с экспоненциальным забыванием (λ={lambda_param})...")

    # Анализ скользящим окном с экспоненциальным забыванием
    analysis_results = rolling_window_analysis(
        returns, window_size, step_size, lambda_param
    )
    print(f"Получено окон: {len(analysis_results)}")

    # Расчет эффективных границ для каждого окна
    frontiers = calculate_efficient_frontiers_over_time(analysis_results, n_points)
    print(f"Рассчитано эффективных границ: {len(frontiers)}")

    # Анализ стабильности
    stability_metrics = analyze_efficient_frontier_stability(frontiers, returns.columns)
    print(f"Рассчитаны метрики стабильности")

    return frontiers, stability_metrics


def task_10_exp_efficient_frontier_dynamics_expanding(
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
    print(f"Задача 10b: Анализ эффективной границы расширяющимся окном с эксп. забыванием (λ={lambda_param})...")

    # Анализ расширяющимся окном с экспоненциальным забыванием
    analysis_results = expanding_window_analysis(returns, step_size, lambda_param)
    print(f"Получено окон: {len(analysis_results)}")

    # Расчет эффективных границ для каждого окна
    frontiers = calculate_efficient_frontiers_over_time(analysis_results, n_points)
    print(f"Рассчитано эффективных границ: {len(frontiers)}")

    # Анализ стабильности
    stability_metrics = analyze_efficient_frontier_stability(frontiers, returns.columns)
    print(f"Рассчитаны метрики стабильности")

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

            comparison_data.append({
                'method': method_name,
                'date': date,
                'min_std': frontier['min_std'],
                'min_std_return': frontier['min_std_return'],
                'max_return': frontier['max_return'],
                'max_return_std': frontier['max_return_std'],
                'efficiency_ratio': frontier['max_return'] / frontier['min_std']
            })

    df = pd.DataFrame(comparison_data)
    df.set_index(['method', 'date'], inplace=True)

    return df


def main():
    """
    Главная функция для демонстрации работы.
    """
    print("=" * 60)
    print("Задачи 9 и 10: Анализ стабильности эффективной границы")
    print("=" * 60)
    print()

    # Загрузка данных
    print("Загрузка данных...")
    prices = load_prices_data('../data/prices_moex_new.csv')
    print(f"Загружено данных: {prices.shape}")
    print(f"Период: {prices.index.min()} - {prices.index.max()}")
    print(f"Акции: {list(prices.columns)}\n")

    # Расчет доходностей
    print("Расчет доходностей...")
    returns = calculate_returns(prices)
    print(f"Доходности: {returns.shape}\n")

    # Задача 9a: Скользящее окно
    print("-" * 60)
    rolling_frontiers, rolling_stability = task_9a_efficient_frontier_dynamics_rolling(
        returns, window_size='1Y', step_size='1Y'
    )
    print(f"\nМетрики стабильности для скользящего окна:")
    print(rolling_stability.head())
    print()

    # Задача 9b: Расширяющееся окно
    print("-" * 60)
    expanding_frontiers, expanding_stability = task_9b_efficient_frontier_dynamics_expanding(
        returns, step_size='1Y'
    )
    print(f"\nМетрики стабильности для расширяющегося окна:")
    print(expanding_stability.head())
    print()

    # Задача 10: Экспоненциальное забывание (скользящее окно)
    print("-" * 60)
    rolling_exp_frontiers, rolling_exp_stability = task_10_efficient_frontier_dynamics_exponential(
        returns, window_size='1Y', step_size='1Y', lambda_param=0.94
    )
    print(f"\nМетрики стабильности для эксп. забывания (скользящее окно):")
    print(rolling_exp_stability.head())
    print()

    # Сравнение методов
    print("-" * 60)
    comparison = compare_frontier_methods([
        ('Скользящее окно', rolling_frontiers),
        ('Расширяющееся окно', expanding_frontiers),
        ('Скользящее окно (эксп. забывание)', rolling_exp_frontiers)
    ])
    print(f"\nСравнение методов:")
    print(comparison.head(10))
    print()

    return {
        'prices': prices,
        'returns': returns,
        'rolling_frontiers': rolling_frontiers,
        'rolling_stability': rolling_stability,
        'expanding_frontiers': expanding_frontiers,
        'expanding_stability': expanding_stability,
        'rolling_exp_frontiers': rolling_exp_frontiers,
        'rolling_exp_stability': rolling_exp_stability,
        'comparison': comparison
    }


if __name__ == '__main__':
    results = main()
