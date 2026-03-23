"""
Задача 2-3: Расчет векторов доходностей и ковариационных матриц

Задача 2:
a. Рассчитать векторы доходностей и ковариационные матрицы скользящим окном
b. Рассчитать векторы доходностей и ковариационные матрицы расширяющимся окном

Задача 3: Выполнить задачу 2 с экспоненциальным забыванием
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from datetime import datetime, timedelta


def load_prices_data(file_path: str) -> pd.DataFrame:
    """
    Загрузка данных о ценах из CSV файла.

    Parameters:
    -----------
    file_path : str
        Путь к CSV файлу с ценами

    Returns:
    --------
    pd.DataFrame
        DataFrame с датами и ценами акций
    """
    # Загрузка данных
    df = pd.read_csv(file_path, sep=';', decimal=',')

    # Удаление пробелов в названиях столбцов
    df.columns = df.columns.str.strip()

    # Парсинг дат из формата dd.mm.yyyy
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')

    # Установка даты как индекса
    df.set_index('date', inplace=True)

    # Удаление столбцов с пустыми значениями (или заменяем их на NaN)
    df = df.replace('', np.nan).replace(' ', np.nan)

    # Удаление строк где все значения NaN (кроме первой строки)
    df = df.dropna(how='all')

    # Преобразование в числовой формат
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Удаление столбцов с полностью пропущенными данными
    df = df.dropna(axis=1, how='all')

    return df


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Расчет логарифмических доходностей.

    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame с ценами акций

    Returns:
    --------
    pd.DataFrame
        DataFrame с логарифмическими доходностями
    """
    returns = np.log(prices / prices.shift(1))
    returns = returns.dropna()
    return returns


def calculate_exponential_weights(n: int, lambda_param: float = 0.94) -> np.ndarray:
    """
    Расчет экспоненциальных весов для схемы забывания.

    Parameters:
    -----------
    n : int
        Количество наблюдений
    lambda_param : float
        Параметр затухания (обычно 0.94-0.97)

    Returns:
    --------
    np.ndarray
        Массив экспоненциальных весов
    """
    weights = np.array([lambda_param ** (n - i - 1) for i in range(n)])
    weights = weights / weights.sum()  # Нормализация
    return weights


def calculate_weighted_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    """
    Расчет взвешенных доходностей.

    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame с доходностями
    weights : np.ndarray
        Массив весов

    Returns:
    --------
    pd.DataFrame
        DataFrame с взвешенными доходностями
    """
    weighted_returns = returns.mul(weights, axis=0)
    return weighted_returns


def calculate_weighted_covariance(returns: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    """
    Расчет взвешенной ковариационной матрицы.

    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame с доходностями
    weights : np.ndarray
        Массив весов

    Returns:
    --------
    np.ndarray
        Взвешенная ковариационная матрица
    """
    # Центрирование доходностей (вычитание взвешенного среднего)
    weighted_mean = returns.mul(weights, axis=0).sum(axis=0)
    centered_returns = returns - weighted_mean

    # Расчет взвешенной ковариации
    n = len(returns)
    cov_matrix = np.zeros((returns.shape[1], returns.shape[1]))

    for i in range(returns.shape[1]):
        for j in range(returns.shape[1]):
            cov_matrix[i, j] = np.sum(weights * centered_returns.iloc[:, i] * centered_returns.iloc[:, j])

    return cov_matrix


def rolling_window_analysis(
    returns: pd.DataFrame,
    window_size: str = '1Y',
    step_size: str = '1Y',
    lambda_param: float = None
) -> Dict[datetime, Dict[str, np.ndarray]]:
    """
    Расчет векторов доходностей и ковариационных матриц скользящим окном.

    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame с доходностями
    window_size : str
        Размер окна (например, '1Y', '1Q', '1M', '1W', '1D')
    step_size : str
        Размер шага окна
    lambda_param : float, optional
        Параметр для экспоненциального забывания. Если None, используется равномерное взвешивание

    Returns:
    --------
    Dict[datetime, Dict[str, np.ndarray]]
        Словарь с результатами для каждого окна
    """
    results = {}

    # Преобразование размера окна в количество дней
    window_days = _parse_period_to_days(window_size)
    step_days = _parse_period_to_days(step_size)

    # Получаем даты начала и конца
    start_date = returns.index.min()
    end_date = returns.index.max()

    # Проходим по датам с заданным шагом
    current_end = start_date + timedelta(days=window_days)

    while current_end <= end_date:
        current_start = current_end - timedelta(days=window_days)

        # Получаем данные для текущего окна
        window_returns = returns.loc[current_start:current_end]

        if len(window_returns) > 1:  # Нужно минимум 2 наблюдения
            # Расчет вектора средних доходностей
            if lambda_param is None:
                # Равномерное взвешивание
                mean_returns = window_returns.mean().values
                cov_matrix = window_returns.cov().values
            else:
                # Экспоненциальное забывание
                weights = calculate_exponential_weights(len(window_returns), lambda_param)
                mean_returns = calculate_weighted_returns(window_returns, weights).sum(axis=0).values
                cov_matrix = calculate_weighted_covariance(window_returns, weights)

            results[current_end] = {
                'window_start': current_start,
                'mean_returns': mean_returns,
                'covariance_matrix': cov_matrix,
                'window_returns': window_returns.values
            }

        current_end += timedelta(days=step_days)

    return results


def expanding_window_analysis(
    returns: pd.DataFrame,
    step_size: str = '1Y',
    lambda_param: float = None
) -> Dict[datetime, Dict[str, np.ndarray]]:
    """
    Расчет векторов доходностей и ковариационных матриц расширяющимся окном.

    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame с доходностями
    step_size : str
        Размер шага для расширения окна
    lambda_param : float, optional
        Параметр для экспоненциального забывания. Если None, используется равномерное взвешивание

    Returns:
    --------
    Dict[datetime, Dict[str, np.ndarray]]
        Словарь с результатами для каждого окна
    """
    results = {}

    # Преобразование размера шага в количество дней
    step_days = _parse_period_to_days(step_size)

    # Получаем даты начала и конца
    start_date = returns.index.min()
    end_date = returns.index.max()

    # Начальный размер окна (минимум 1 год для корректной оценки)
    initial_window_days = 365

    current_end = start_date + timedelta(days=initial_window_days)

    while current_end <= end_date:
        # Окно от начала данных до текущей даты (расширяющееся)
        window_returns = returns.loc[start_date:current_end]

        if len(window_returns) > 1:
            # Расчет вектора средних доходностей
            if lambda_param is None:
                # Равномерное взвешивание
                mean_returns = window_returns.mean().values
                cov_matrix = window_returns.cov().values
            else:
                # Экспоненциальное забывание
                weights = calculate_exponential_weights(len(window_returns), lambda_param)
                mean_returns = calculate_weighted_returns(window_returns, weights).sum(axis=0).values
                cov_matrix = calculate_weighted_covariance(window_returns, weights)

            results[current_end] = {
                'window_start': start_date,
                'window_end': current_end,
                'window_size': len(window_returns),
                'mean_returns': mean_returns,
                'covariance_matrix': cov_matrix,
                'window_returns': window_returns.values
            }

        current_end += timedelta(days=step_days)

    return results


def _parse_period_to_days(period: str) -> int:
    """
    Преобразование периода в количество дней.

    Parameters:
    -----------
    period : str
        Период (например, '1Y', '1Q', '1M', '1W', '1D')

    Returns:
    --------
    int
        Количество дней
    """
    period_map = {
        'D': 1,
        'W': 7,
        'M': 30,
        'Q': 90,
        'Y': 365
    }

    if len(period) < 2:
        raise ValueError(f"Invalid period format: {period}")

    unit = period[-1]
    try:
        value = int(period[:-1])
    except ValueError:
        raise ValueError(f"Invalid period format: {period}")

    if unit not in period_map:
        raise ValueError(f"Unsupported period unit: {unit}")

    return value * period_map[unit]


def main():
    """
    Главная функция для демонстрации работы.
    """
    # Загрузка данных
    print("Загрузка данных...")
    prices = load_prices_data('data/prices_moex_new.csv')
    print(f"Загружено данных: {prices.shape}")
    print(f"Период: {prices.index.min()} - {prices.index.max()}")
    print(f"Акции: {list(prices.columns)}\n")

    # Расчет доходностей
    print("Расчет доходностей...")
    returns = calculate_returns(prices)
    print(f"Доходности: {returns.shape}\n")

    # Задача 2a: Скользящее окно
    print("Задача 2a: Скользящее окно (1 год, шаг 1 год)...")
    rolling_results = rolling_window_analysis(returns, window_size='1Y', step_size='1Y')
    print(f"Получено окон: {len(rolling_results)}")

    # Пример результатов первого окна
    if rolling_results:
        first_date = list(rolling_results.keys())[0]
        first_result = rolling_results[first_date]
        print(f"Первое окно: {first_result['window_start']} - {first_date}")
        print(f"Размер окна: {len(first_result['window_returns'])} наблюдений")
        print(f"Размер ковариационной матрицы: {first_result['covariance_matrix'].shape}")
        print(f"Пример средних доходностей: {first_result['mean_returns'][:5]}...")
        print()

    # Задача 2b: Расширяющееся окно
    print("Задача 2b: Расширяющееся окно (шаг 1 год)...")
    expanding_results = expanding_window_analysis(returns, step_size='1Y')
    print(f"Получено окон: {len(expanding_results)}")

    if expanding_results:
        first_date = list(expanding_results.keys())[0]
        first_result = expanding_results[first_date]
        print(f"Первое окно: {first_result['window_start']} - {first_result['window_end']}")
        print(f"Размер окна: {first_result['window_size']} наблюдений")
        print()

    # Задача 3: Экспоненциальное забывание
    print("Задача 3: Скользящее окно с экспоненциальным забыванием (lambda=0.94)...")
    rolling_exp_results = rolling_window_analysis(
        returns, window_size='1Y', step_size='1Y', lambda_param=0.94
    )
    print(f"Получено окон: {len(rolling_exp_results)}")

    if rolling_exp_results:
        first_date = list(rolling_exp_results.keys())[0]
        first_result = rolling_exp_results[first_date]
        print(f"Первое окно: {first_result['window_start']} - {first_date}")
        print(f"Пример средних доходностей (эксп. взвешивание): {first_result['mean_returns'][:5]}...")
        print()

    print("Расширяющееся окно с экспоненциальным забыванием (lambda=0.94)...")
    expanding_exp_results = expanding_window_analysis(
        returns, step_size='1Y', lambda_param=0.94
    )
    print(f"Получено окон: {len(expanding_exp_results)}")

    return {
        'prices': prices,
        'returns': returns,
        'rolling': rolling_results,
        'expanding': expanding_results,
        'rolling_exp': rolling_exp_results,
        'expanding_exp': expanding_exp_results
    }


if __name__ == '__main__':
    results = main()
