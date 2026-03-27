"""
Задачи 13-15: Оценка входящих данных для optimizer на основе исторических β

Задача 13: Рассчитать ковариационную матрицу на основе исторических β (historical betas)
Задача 14: Построить границу эффективных портфелей на основе полученной ковариационной матрицы
Задача 15: Построить границу эффективных портфелей для разных исторических окон

Выбор из задач 11-12:
- Индекс: IMOEX (Мосбиржа)
- Окно: Скользящее окно длиной в 1 год (252 торговых дня)
- Схема взвешивания: Равные веса наблюдений
"""

import numpy as np
import pandas as pd
import sys
from typing import Dict, Tuple, List, Optional
from datetime import datetime

# Импорт функций из task_2_3
sys.path.insert(0, '..')
from task_2_3 import (
    calculate_returns,
    rolling_window_analysis
)

# Импорт функций из task_9-10
from task_9_10 import (
    calculate_efficient_frontier,
    analyze_efficient_frontier_stability
)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Предварительная обработка данных о ценах.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame с данными о ценах

    Returns:
    --------
    pd.DataFrame
        Обработанный DataFrame
    """
    df.columns = df.columns.str.strip()
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
    df.set_index('date', inplace=True)

    df = df.replace('', np.nan).replace(' ', np.nan) \
            .dropna(how='all') \
            .dropna(axis=1, how='all')

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def load_prices_data(file_path: str) -> pd.DataFrame:
    """
    Загрузка данных о ценах из CSV файла.

    Parameters:
    -----------
    file_path : str
        Путь к файлу с данными

    Returns:
    --------
    pd.DataFrame
        DataFrame с ценами акций
    """
    df = pd.read_csv(file_path, sep=';', decimal=',')

    return preprocess_data(df)


def calculate_market_model_betas(
    stock_returns: pd.Series,
    market_returns: pd.Series
) -> Tuple[float, float]:
    """
    Расчет параметров рыночной модели (market model).

    Рыночная модель: r_i = α_i + β_i * r_m + ε_i

    Parameters:
    -----------
    stock_returns : pd.Series
        Доходности акции
    market_returns : pd.Series
        Доходности рыночного индекса

    Returns:
    --------
    Tuple[float, float]
        (alpha, beta) - параметры рыночной модели
    """
    # Объединение данных (inner join - только общие даты)
    combined = pd.DataFrame({'stock': stock_returns, 'market': market_returns}).dropna()

    if len(combined) < 2:
        return np.nan, np.nan

    x = combined['market'].values
    y = combined['stock'].values

    # Оценка β и α через OLS
    # β = Cov(r_i, r_m) / Var(r_m)
    # α = E[r_i] - β * E[r_m]
    cov_matrix = np.cov(x, y, ddof=1)
    beta = cov_matrix[0, 1] / cov_matrix[0, 0]
    alpha = np.mean(y) - beta * np.mean(x)

    return alpha, beta


def calculate_all_betas(
    returns: pd.DataFrame,
    market_ticker: str = 'MOEX'
) -> pd.DataFrame:
    """
    Расчет исторических бета для всех акций относительно рыночного индекса.

    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame с доходностями всех акций и индекса
    market_ticker : str
        Тикер рыночного индекса (должен быть в returns.columns)

    Returns:
    --------
    pd.DataFrame
        DataFrame с бета для всех акций
    """
    if market_ticker not in returns.columns:
        raise ValueError(f"Тикер индекса {market_ticker} не найден в данных")

    market_returns = returns[market_ticker]
    stock_tickers = [col for col in returns.columns if col != market_ticker]

    betas_data = []

    for ticker in stock_tickers:
        stock_returns = returns[ticker]
        alpha, beta = calculate_market_model_betas(stock_returns, market_returns)

        betas_data.append({
            'ticker': ticker,
            'alpha': alpha,
            'beta': beta
        })

    df = pd.DataFrame(betas_data)
    df.set_index('ticker', inplace=True)

    return df


def calculate_covariance_from_betas(
    betas: pd.Series,
    market_variance: float,
    residual_variances: Optional[pd.Series] = None
) -> np.ndarray:
    """
    Расчет ковариационной матрицы на основе рыночной модели.

    Согласно рыночной модели:
    Cov(r_i, r_j) = β_i * β_j * σ_m² + Cov(ε_i, ε_j)

    Если предположить, что ошибки ε_i некоррелированы (стандартное предположение):
    Cov(r_i, r_j) = β_i * β_j * σ_m²  для i ≠ j
    Var(r_i) = β_i² * σ_m² + σ_ε_i²

    Parameters:
    -----------
    betas : pd.Series
        Вектор бета-коэффициентов (индекс - тикеры)
    market_variance : float
        Дисперсия рыночного индекса
    residual_variances : pd.Series, optional
        Дисперсии остатков ε (если None, предполагаем некоррелированные остатки с Var(ε_i)=0)

    Returns:
    --------
    np.ndarray
        Ковариационная матрица
    """
    n_assets = len(betas)
    cov_matrix = np.zeros((n_assets, n_assets))

    if residual_variances is None:
        # Простейший случай: предполагаем, что все дисперсии обусловлены рыночным риском
        for i in range(n_assets):
            for j in range(n_assets):
                cov_matrix[i, j] = betas.iloc[i] * betas.iloc[j] * market_variance
    else:
        # Учет остаточного риска (idiosyncratic risk)
        for i in range(n_assets):
            for j in range(n_assets):
                if i == j:
                    # Var(r_i) = β_i² * σ_m² + σ_ε_i²
                    cov_matrix[i, j] = (betas.iloc[i] ** 2) * market_variance + residual_variances.iloc[i]
                else:
                    # Cov(r_i, r_j) = β_i * β_j * σ_m²
                    cov_matrix[i, j] = betas.iloc[i] * betas.iloc[j] * market_variance

    return cov_matrix


def calculate_residual_variances(
    returns: pd.DataFrame,
    betas: pd.DataFrame,
    market_ticker: str = 'MOEX'
) -> pd.Series:
    """
    Расчет дисперсий остатков (residual variances) для каждой акции.

    ε_i = r_i - (α_i + β_i * r_m)

    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame с доходностями
    betas : pd.DataFrame
        DataFrame с бета-коэффициентами (должен содержать 'alpha' и 'beta')
    market_ticker : str
        Тикер рыночного индекса

    Returns:
    --------
    pd.Series
        Дисперсии остатков для каждой акции
    """
    market_returns = returns[market_ticker]
    residuals_var = {}

    for ticker in betas.index:
        if ticker not in returns.columns or ticker == market_ticker:
            residuals_var[ticker] = 0.0
            continue

        stock_returns = returns[ticker]
        combined = pd.DataFrame({'stock': stock_returns, 'market': market_returns}).dropna()

        if len(combined) < 2:
            residuals_var[ticker] = 0.0
            continue

        # Расчет остатков
        alpha = betas.loc[ticker, 'alpha']
        beta = betas.loc[ticker, 'beta']
        residuals = combined['stock'] - (alpha + beta * combined['market'])

        # Дисперсия остатков
        residuals_var[ticker] = residuals.var(ddof=1)

    return pd.Series(residuals_var)


def task_13_covariance_from_historical_betas(
    returns: pd.DataFrame,
    market_ticker: str = 'MOEX',
    include_residuals: bool = True
) -> Dict[str, np.ndarray]:
    """
    Задача 13: Рассчитать ковариационную матрицу на основе исторических β.

    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame с доходностями всех акций и индекса
    market_ticker : str
        Тикер рыночного индекса
    include_residuals : bool
        Учитывать ли остаточные дисперсии

    Returns:
    --------
    Dict[str, np.ndarray]
        Словарь с ковариационной матрицей и метаданными
    """
    
    
    

    # Исключаем индекс из списка акций
    stock_tickers = [col for col in returns.columns if col != market_ticker]
    stock_returns = returns[stock_tickers]

    # Расчет бета для всех акций
    betas_with_alpha = calculate_all_betas(returns, market_ticker)
    betas = betas_with_alpha['beta']

    } бета-коэффициентов")
    
    )
    
    :.4f}")
    :.4f}")

    # Дисперсия рыночного индекса
    market_variance = returns[market_ticker].var(ddof=1)
    

    # Расчет остаточных дисперсий (если нужно)
    residual_variances = None
    if include_residuals:
        residual_variances = calculate_residual_variances(returns, betas_with_alpha, market_ticker)
        
        :.6f}")

    # Расчет ковариационной матрицы на основе бета
    cov_matrix_beta = calculate_covariance_from_betas(betas, market_variance, residual_variances)

    # Сравнение с классической ковариационной матрицей
    cov_matrix_classic = stock_returns.cov().values

    
    : {cov_matrix_beta.mean():.6f}")
    : {cov_matrix_classic.mean():.6f}")

    # Проверка положительной определенности
    try:
        np.linalg.cholesky(cov_matrix_beta)
        
    except np.linalg.LinAlgError:
        
        # Регуляризация
        cov_matrix_beta = cov_matrix_beta + np.eye(cov_matrix_beta.shape[0]) * 1e-8
        

    return {
        'cov_matrix': cov_matrix_beta,
        'betas': betas,
        'alphas': betas_with_alpha['alpha'],
        'residual_variances': residual_variances,
        'market_variance': market_variance,
        'market_ticker': market_ticker,
        'stock_tickers': stock_tickers,
        'cov_matrix_classic': cov_matrix_classic
    }


def task_14_efficient_frontier_from_betas(
    cov_matrix: np.ndarray,
    mean_returns: np.ndarray,
    n_points: int = 100,
    method_name: str = "Historical Betas"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Задача 14: Построить границу эффективных портфелей на основе ковариационной матрицы из β.

    Parameters:
    -----------
    cov_matrix : np.ndarray
        Ковариационная матрица (из задачи 13)
    mean_returns : np.ndarray
        Вектор средних доходностей
    n_points : int
        Количество точек на эффективной границе
    method_name : str
        Название метода для графиков

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (массив доходностей, массив стандартных отклонений)
    """
    

    # Расчет эффективной границы
    returns, stds = calculate_efficient_frontier(mean_returns, cov_matrix, n_points)

    } точек")
    
    

    # Шарп-отношение
    sharpe_ratios = returns / stds
    max_sharpe = np.max(sharpe_ratios)
    max_sharpe_idx = np.argmax(sharpe_ratios)

    
    

    return returns, stds


def task_15_efficient_frontier_dynamics_betas(
    returns: pd.DataFrame,
    market_ticker: str = 'MOEX',
    window_size: str = '1Y',
    step_size: str = '1Y',
    include_residuals: bool = True,
    n_points: int = 100
) -> Tuple[Dict[datetime, dict], pd.DataFrame]:
    """
    Задача 15: Построить границы эффективных портфелей для разных исторических окон
    на основе исторических β и продемонстрировать динамику её изменения.

    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame с доходностями всех акций и индекса
    market_ticker : str
        Тикер рыночного индекса
    window_size : str
        Размер скользящего окна
    step_size : str
        Размер шага
    include_residuals : bool
        Учитывать ли остаточные дисперсии
    n_points : int
        Количество точек на эффективной границе

    Returns:
    --------
    Tuple[Dict[datetime, dict], pd.DataFrame]
        (словарь эффективных границ, DataFrame с метриками стабильности)
    """
    
    

    # Анализ скользящим окном
    analysis_results = rolling_window_analysis(returns, window_size, step_size)
    }")

    # Для каждого окна рассчитываем бета и эффективную границу
    frontiers = {}

    for date, result in analysis_results.items():
        # Получаем window_returns как DataFrame (используя индекс из исходных данных)
        window_start = result['window_start']
        window_end = date
        window_returns = returns.loc[window_start:window_end]

        # Проверяем, есть ли индекс в данных
        if market_ticker not in window_returns.columns:
            
            continue

        # Исключаем индекс из акций
        stock_tickers = [col for col in window_returns.columns if col != market_ticker]
        stock_returns = window_returns[stock_tickers]

        # Расчет бета
        betas_with_alpha = calculate_all_betas(window_returns, market_ticker)
        betas = betas_with_alpha['beta']

        # Дисперсия рынка
        market_variance = window_returns[market_ticker].var(ddof=1)

        # Остаточные дисперсии
        residual_variances = None
        if include_residuals:
            residual_variances = calculate_residual_variances(window_returns, betas_with_alpha, market_ticker)

        # Ковариационная матрица на основе бета
        cov_matrix_beta = calculate_covariance_from_betas(betas, market_variance, residual_variances)

        # Проверка положительной определенности
        try:
            np.linalg.cholesky(cov_matrix_beta)
        except np.linalg.LinAlgError:
            cov_matrix_beta = cov_matrix_beta + np.eye(cov_matrix_beta.shape[0]) * 1e-8

        # Средние доходности (только для акций)
        mean_returns = stock_returns.mean().values

        # Эффективная граница
        ef_returns, ef_stds = calculate_efficient_frontier(mean_returns, cov_matrix_beta, n_points)

        frontiers[date] = {
            'returns': ef_returns,
            'stds': ef_stds,
            'min_std': ef_stds[0],
            'min_std_return': ef_returns[0],
            'max_return': ef_returns[-1],
            'max_return_std': ef_stds[-1]
        }

    }")

    # Анализ стабильности
    if frontiers:
        stability_metrics = analyze_efficient_frontier_stability(frontiers)
        
        return frontiers, stability_metrics
    else:
        return {}, pd.DataFrame()


def compare_covariance_methods(
    returns: pd.DataFrame,
    market_ticker: str = 'MOEX'
) -> Dict[str, np.ndarray]:
    """
    Сравнение ковариационных матриц, рассчитанных разными методами.

    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame с доходностями
    market_ticker : str
        Тикер рыночного индекса

    Returns:
    --------
    Dict[str, np.ndarray]
        Словарь с ковариационными матрицами разных методов
    """
    

    # Классическая матрица
    stock_tickers = [col for col in returns.columns if col != market_ticker]
    cov_classic = returns[stock_tickers].cov().values

    # Матрица на основе бета (без остаточных дисперсий)
    betas_result = task_13_covariance_from_historical_betas(
        returns, market_ticker, include_residuals=False
    )
    cov_beta_simple = betas_result['cov_matrix']

    # Матрица на основе бета (с остаточными дисперсиями)
    betas_result_resid = task_13_covariance_from_historical_betas(
        returns, market_ticker, include_residuals=True
    )
    cov_beta_residuals = betas_result_resid['cov_matrix']

    
    
    :.6f}")
    :.6f}")
    :.6e}")

    :")
    :.6f}")
    :.6f}")
    :.6e}")

    :")
    :.6f}")
    :.6f}")
    :.6e}")

    # Разность матриц
    diff_simple = np.abs(cov_classic - cov_beta_simple)
    diff_residuals = np.abs(cov_classic - cov_beta_residuals)

    
    : {diff_simple.mean():.6f}")
    : {diff_residuals.mean():.6f}")

    return {
        'classic': cov_classic,
        'beta_simple': cov_beta_simple,
        'beta_residuals': cov_beta_residuals
    }


def main():
    """
    Главная функция для демонстрации работы.
    """
    
    
    
    

    # Загрузка данных
    
    prices = load_prices_data('../data/prices_moex_new.csv')
    

    # Расчет доходностей
    
    returns = calculate_returns(prices)
    
    }")
    

    # Задача 13: Расчет ковариационной матрицы на основе бета
    betas_result = task_13_covariance_from_historical_betas(
        returns, market_ticker='MOEX', include_residuals=True
    )
    

    # Задача 14: Эффективная граница на основе бета
    stock_tickers = [col for col in returns.columns if col != 'MOEX']
    stock_returns = returns[stock_tickers]
    mean_returns = stock_returns.mean().values

    task_14_efficient_frontier_from_betas(
        betas_result['cov_matrix'],
        mean_returns,
        n_points=50,
        method_name="Исторические β"
    )
    

    # Задача 15: Динамика эффективных границ
    beta_frontiers, beta_stability = task_15_efficient_frontier_dynamics_betas(
        returns, market_ticker='MOEX', window_size='1Y', step_size='1Y',
        include_residuals=True, n_points=50
    )
    

    # Сравнение методов
    compare_covariance_methods(returns, market_ticker='MOEX')

    return {
        'prices': prices,
        'returns': returns,
        'betas_result': betas_result,
        'beta_frontiers': beta_frontiers,
        'beta_stability': beta_stability
    }


if __name__ == '__main__':
    results = main()
