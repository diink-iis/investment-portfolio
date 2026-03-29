import numpy as np
import pandas as pd
import sys
from typing import Dict, Tuple, List, Optional, Union
from datetime import datetime

# Импорт функций из task_2_3
sys.path.insert(0, '..')
from task_2_3 import (
    calculate_returns,
    rolling_window_analysis
)

# Импорт функций из task_9_10
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


def load_imoex_data(file_path: str) -> pd.DataFrame:
    """
    Загрузка данных индекса IMOEX из CSV файла.

    Parameters:
    -----------
    file_path : str
        Путь к файлу с данными индекса IMOEX

    Returns:
    --------
    pd.DataFrame
        DataFrame с доходностями индекса IMOEX
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
    imoex_returns: Union[pd.Series, pd.DataFrame],
    stock_returns: pd.DataFrame
) -> pd.DataFrame:
    """
    Расчет исторических бета для всех акций относительно рыночного индекса IMOEX.

    Parameters:
    -----------
    imoex_returns : pd.Series or pd.DataFrame
        Доходности индекса IMOEX
    stock_returns : pd.DataFrame
        DataFrame с доходностями акций

    Returns:
    --------
    pd.DataFrame
        DataFrame с бета для всех акций
    """
    # Преобразуем imoex_returns в Series если это DataFrame
    if isinstance(imoex_returns, pd.DataFrame):
        imoex_series = imoex_returns.iloc[:, 0]  # Первая колонка
    else:
        imoex_series = imoex_returns

    # Объединение данных (inner join - только общие даты)
    combined_data = stock_returns.copy()
    combined_data['IMOEX'] = imoex_series

    stock_tickers = [col for col in stock_returns.columns if col != 'IMOEX' and col != 'IMOEX']

    betas_data = []

    for ticker in stock_tickers:
        if ticker not in combined_data.columns:
            continue

        stock_returns_ticker = combined_data[ticker]
        market_returns = combined_data['IMOEX']

        alpha, beta = calculate_market_model_betas(stock_returns_ticker, market_returns)

        if not np.isnan(alpha) and not np.isnan(beta):
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
    Cov(r_i, r_j) = β_i * β_j * σ_m² для i ≠ j
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
    imoex_returns: Union[pd.Series, pd.DataFrame],
    betas: pd.DataFrame,
    stock_returns: pd.DataFrame
) -> pd.Series:
    """
    Расчет дисперсий остатков (residual variances) для каждой акции.

    ε_i = r_i - (α_i + β_i * r_m)

    Parameters:
    -----------
    imoex_returns : pd.Series or pd.DataFrame
        Доходности индекса IMOEX
    betas : pd.DataFrame
        DataFrame с бета-коэффициентами (должен содержать 'alpha' и 'beta')
    stock_returns : pd.DataFrame
        DataFrame с доходностями

    Returns:
    --------
    pd.Series
        Дисперсии остатков для каждой акции
    """
    # Преобразуем imoex_returns в Series если это DataFrame
    if isinstance(imoex_returns, pd.DataFrame):
        imoex_series = imoex_returns.iloc[:, 0]  # Первая колонка
    else:
        imoex_series = imoex_returns

    residuals_var = {}

    for ticker in betas.index:
        if ticker not in stock_returns.columns:
            residuals_var[ticker] = 0.0
            continue

        stock_returns_ticker = stock_returns[ticker]
        combined = pd.DataFrame({'stock': stock_returns_ticker, 'market': imoex_series}).dropna()

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


def covariance_from_historical_betas(
    imoex_returns: Union[pd.Series, pd.DataFrame],
    stock_returns: pd.DataFrame,
    include_residuals: bool = True
) -> Dict[str, np.ndarray]:
    """
    Задача 13: Рассчитать ковариационную матрицу на основе исторических β.

    Parameters:
    -----------
    imoex_returns : pd.Series or pd.DataFrame
        Доходности индекса IMOEX
    stock_returns : pd.DataFrame
        DataFrame с доходностями всех акций
    include_residuals : bool
        Учитывать ли остаточные дисперсии

    Returns:
    --------
    Dict[str, np.ndarray]
        Словарь с ковариационной матрицей и метаданными
    """
    stock_tickers = [col for col in stock_returns.columns if col != 'IMOEX']
    stock_returns_only = stock_returns[stock_tickers]

    betas_with_alpha = calculate_all_betas(imoex_returns, stock_returns_only)
    betas = betas_with_alpha['beta']

    # Дисперсия рынка
    if isinstance(imoex_returns, pd.DataFrame):
        market_variance = imoex_returns.iloc[:, 0].var(ddof=1)
    else:
        market_variance = imoex_returns.var(ddof=1)

    residual_variances = None
    if include_residuals:
        residual_variances = calculate_residual_variances(imoex_returns, betas_with_alpha, stock_returns_only)

    cov_matrix_beta = calculate_covariance_from_betas(betas, market_variance, residual_variances)

    cov_matrix_classic = stock_returns_only.cov().values

    return {
        'cov_matrix': cov_matrix_beta,
        'betas': betas,
        'alphas': betas_with_alpha['alpha'],
        'residual_variances': residual_variances,
        'market_variance': market_variance,
        'stock_tickers': stock_tickers,
        'cov_matrix_classic': cov_matrix_classic
    }


def efficient_frontier_from_betas(
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
    returns, stds = calculate_efficient_frontier(mean_returns, cov_matrix, n_points)

    return returns, stds


def efficient_frontier_dynamics_betas(
    imoex_returns: Union[pd.Series, pd.DataFrame],
    stock_returns: pd.DataFrame,
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
    imoex_returns : pd.Series or pd.DataFrame
        Доходности индекса IMOEX
    stock_returns : pd.DataFrame
        DataFrame с доходностями всех акций
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
    # Подготовка данных: добавляем IMOEX к stock_returns
    combined_returns = stock_returns.copy()

    # Преобразуем imoex_returns в Series если это DataFrame
    if isinstance(imoex_returns, pd.DataFrame):
        imoex_series = imoex_returns.iloc[:, 0]
    else:
        imoex_series = imoex_returns

    combined_returns['IMOEX'] = imoex_series

    # Анализ скользящим окном
    analysis_results = rolling_window_analysis(combined_returns, window_size, step_size)

    # Для каждого окна рассчитываем бета и эффективную границу
    frontiers = {}

    for date, result in analysis_results.items():
        # Получаем window_returns как DataFrame (используя индекс из исходных данных)
        window_start = result['window_start']
        window_end = date
        window_returns = combined_returns.loc[window_start:window_end]

        # Проверяем, есть ли индекс в данных
        if 'IMOEX' not in window_returns.columns:
            continue

        # Исключаем индекс из акций
        stock_tickers = [col for col in window_returns.columns if col != 'IMOEX']
        stock_returns_window = window_returns[stock_tickers]

        # Расчет бета
        betas_with_alpha = calculate_all_betas(window_returns['IMOEX'], stock_returns_window)
        betas = betas_with_alpha['beta']

        # Дисперсия рынка
        market_variance = window_returns['IMOEX'].var(ddof=1)

        # Остаточные дисперсии
        residual_variances = None
        if include_residuals:
            residual_variances = calculate_residual_variances(window_returns['IMOEX'], betas_with_alpha, stock_returns_window)

        # Ковариационная матрица на основе бета
        cov_matrix_beta = calculate_covariance_from_betas(betas, market_variance, residual_variances)

        # Проверка положительной определенности
        try:
            np.linalg.cholesky(cov_matrix_beta)
        except np.linalg.LinAlgError:
            cov_matrix_beta = cov_matrix_beta + np.eye(cov_matrix_beta.shape[0]) * 1e-8

        # Средние доходности (только для акций)
        mean_returns = stock_returns_window.mean().values

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

    # Анализ стабильности
    if frontiers:
        stability_metrics = analyze_efficient_frontier_stability(frontiers)
        return frontiers, stability_metrics
    else:
        return {}, pd.DataFrame()


def compare_covariance_methods(
    imoex_returns: Union[pd.Series, pd.DataFrame],
    stock_returns: pd.DataFrame
) -> Dict[str, np.ndarray]:
    """
    Сравнение ковариационных матриц, рассчитанных разными методами.

    Parameters:
    -----------
    imoex_returns : pd.Series or pd.DataFrame
        Доходности индекса IMOEX
    stock_returns : pd.DataFrame
        DataFrame с доходностями

    Returns:
    --------
    Dict[str, np.ndarray]
        Словарь с ковариационными матрицами разных методов
    """
    # Подготовка данных: добавляем IMOEX к stock_returns
    combined_returns = stock_returns.copy()

    # Преобразуем imoex_returns в Series если это DataFrame
    if isinstance(imoex_returns, pd.DataFrame):
        imoex_series = imoex_returns.iloc[:, 0]
    else:
        imoex_series = imoex_returns

    combined_returns['IMOEX'] = imoex_series

    # Классическая матрица
    stock_tickers = [col for col in combined_returns.columns if col != 'IMOEX']
    cov_classic = combined_returns[stock_tickers].cov().values

    # Матрица на основе бета (без остаточных дисперсий)
    betas_result = covariance_from_historical_betas(
        imoex_series,
        combined_returns,
        include_residuals=False
    )
    cov_beta_simple = betas_result['cov_matrix']

    # Матрица на основе бета (с остаточными дисперсиями)
    betas_result_resid = covariance_from_historical_betas(
        imoex_series,
        combined_returns,
        include_residuals=True
    )
    cov_beta_residuals = betas_result_resid['cov_matrix']

    return {
        'classic': cov_classic,
        'beta_simple': cov_beta_simple,
        'beta_residuals': cov_beta_residuals
    }


# ============ ГЛАВНЫЕ ФУНКЦИИ ЗАДАЧ ============

def task_13_covariance_from_historical_betas(
    imoex_returns_file: str,
    stock_returns_file: str,
    include_residuals: bool = True
) -> Dict[str, np.ndarray]:
    """
    Задача 13: Рассчитать ковариационную матрицу на основе исторических β
    с использованием официального индекса IMOEX из отдельного файла.

    Parameters:
    -----------
    imoex_returns_file : str
        Путь к файлу с данными индекса IMOEX
    stock_returns_file : str
        Путь к файлу с данными о ценах акций
    include_residuals : bool
        Учитывать ли остаточные дисперсии

    Returns:
    --------
    Dict[str, np.ndarray]
        Словарь с ковариационной матрицей и метаданными
    """
    # Загрузка данных
    stock_prices = load_prices_data(stock_returns_file)
    imoex_prices = load_imoex_data(imoex_returns_file)

    # Расчет доходностей
    stock_returns = calculate_returns(stock_prices)
    imoex_returns = calculate_returns(imoex_prices)

    # Расчет ковариационной матрицы
    result = covariance_from_historical_betas(
        imoex_returns,
        stock_returns,
        include_residuals
    )

    return result


def task_14_efficient_frontier_from_betas(
    imoex_returns_file: str,
    stock_returns_file: str,
    n_points: int = 50,
    include_residuals: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Задача 14: Построить эффективную границу на основе бета-коэффициентов.

    Parameters:
    -----------
    imoex_returns_file : str
        Путь к файлу с данными индекса IMOEX
    stock_returns_file : str
        Путь к файлу с данными о ценах акций
    n_points : int
        Количество точек на эффективной границе
    include_residuals : bool
        Учитывать ли остаточные дисперсии

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (массив доходностей, массив стандартных отклонений)
    """
    # Загрузка данных
    stock_prices = load_prices_data(stock_returns_file)
    imoex_prices = load_imoex_data(imoex_returns_file)

    # Расчет доходностей
    stock_returns = calculate_returns(stock_prices)
    imoex_returns = calculate_returns(imoex_prices)

    # Получаем только акции
    stock_tickers = [col for col in stock_returns.columns if col != 'IMOEX']
    stock_returns_only = stock_returns[stock_tickers]

    # Расчет ковариационной матрицы на основе бета
    betas_result = covariance_from_historical_betas(
        imoex_returns,
        stock_returns,
        include_residuals
    )

    cov_matrix = betas_result['cov_matrix']
    mean_returns = stock_returns_only.mean().values

    # Эффективная граница
    ef_returns, ef_stds = calculate_efficient_frontier(mean_returns, cov_matrix, n_points)

    return ef_returns, ef_stds


def task_15_efficient_frontier_dynamics_betas(
    imoex_returns_file: str,
    stock_returns_file: str,
    window_size: str = '1Y',
    step_size: str = '1Y',
    include_residuals: bool = True,
    n_points: int = 50
) -> Tuple[Dict[datetime, dict], pd.DataFrame]:
    """
    Задача 15: Построить динамику эффективных границ на основе бета-коэффициентов.

    Parameters:
    -----------
    imoex_returns_file : str
        Путь к файлу с данными индекса IMOEX
    stock_returns_file : str
        Путь к файлу с данными о ценах акций
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
    # Загрузка данных
    stock_prices = load_prices_data(stock_returns_file)
    imoex_prices = load_imoex_data(imoex_returns_file)

    # Расчет доходностей
    stock_returns = calculate_returns(stock_prices)
    imoex_returns = calculate_returns(imoex_prices)

    # Подготовка данных: добавляем IMOEX к stock_returns
    combined_returns = stock_returns.copy()
    combined_returns['IMOEX'] = imoex_returns

    # Анализ скользящим окном
    frontiers, stability_metrics = efficient_frontier_dynamics_betas(
        combined_returns['IMOEX'],
        combined_returns,
        window_size=window_size,
        step_size=step_size,
        include_residuals=include_residuals,
        n_points=n_points
    )

    return frontiers, stability_metrics


if __name__ == "__main__":
    # Пример использования
    print("=== Тестирование функций задач 13-15 ===\n")

    # Загрузка данных
    stock_prices = load_prices_data('data/prices_moex_new.csv')
    imoex_prices = load_imoex_data('data/imoex_prices.csv')

    print(f"Акции: {stock_prices.shape}")
    print(f"IMOEX: {imoex_prices.shape}")

    # Расчет доходностей
    stock_returns = calculate_returns(stock_prices)
    imoex_returns = calculate_returns(imoex_prices)

    print(f"\nДоходности акций: {stock_returns.shape}")
    print(f"Доходности IMOEX: {imoex_returns.shape}")

    # Тест задачи 13
    print("\n=== Задача 13: Ковариация на основе исторических β ===")
    task13_result = task_13_covariance_from_historical_betas(
        'data/imoex_prices.csv',
        'data/prices_moex_new.csv',
        include_residuals=True
    )

    print(f"Бета-коэффициентов рассчитано: {len(task13_result['betas'])}")
    print(f"Ковариационная матрица: {task13_result['cov_matrix'].shape}")
    print(f"Дисперсия рынка: {task13_result['market_variance']:.6f}")

    # Тест задачи 14
    print("\n=== Задача 14: Эффективная граница на основе β ===")
    ef_returns, ef_stds = task_14_efficient_frontier_from_betas(
        'data/imoex_prices.csv',
        'data/prices_moex_new.csv',
        n_points=50,
        include_residuals=True
    )

    print(f"Эффективная граница: {len(ef_returns)} точек")
    print(f"Мин. доходность: {ef_returns[0]:.6f}, Стд: {ef_stds[0]:.6f}")
    print(f"Макс. доходность: {ef_returns[-1]:.6f}, Стд: {ef_stds[-1]:.6f}")

    # Тест задачи 15
    print("\n=== Задача 15: Динамика эффективных границ ===")
    frontiers, stability = task_15_efficient_frontier_dynamics_betas(
        'data/imoex_prices.csv',
        'data/prices_moex_new.csv',
        window_size='1Y',
        step_size='1Y',
        include_residuals=True,
        n_points=50
    )

    print(f"Получено окон: {len(frontiers)}")
    if len(stability) > 0:
        print(f"Средний мин. риск: {stability['min_std'].mean():.6f}")
        print(f"Средний макс. Шарп: {stability['max_sharpe'].mean():.6f}")