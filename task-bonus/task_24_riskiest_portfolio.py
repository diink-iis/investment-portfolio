"""
Задача 24 (****): Наиболее рискованный портфель

Найти структуру наиболее рискованного портфеля на основе собранных данных
(при отсутствии ограничений на короткие продажи и вложения в отдельные акции).
Построить границу наиболее рискованных портфелей.
"""

import numpy as np
import pandas as pd
import sys
from typing import Tuple, Dict

# Импорт функций
sys.path.insert(0, '..')
from task_2_3 import calculate_returns
from task_9_10 import calculate_efficient_frontier


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Предварительная обработка данных."""
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
    """Загрузка данных из CSV файла."""
    df = pd.read_csv(file_path, sep=';', decimal=',')
    return preprocess_data(df)


def calculate_max_variance_portfolio(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """
    Расчет наиболее рискованного портфеля (максимизация дисперсии).

    Это двойственная задача к минимизации дисперсии:
    max  w^T * Σ * w
    при: Σ w_i = 1

    Аналитическое решение использует собственные векторы ковариационной матрицы.

    Parameters:
    -----------
    mean_returns : np.ndarray
        Вектор средних доходностей
    cov_matrix : np.ndarray
        Ковариационная матрица

    Returns:
    --------
    Tuple[np.ndarray, float, float]
        (веса портфеля, доходность, риск)
    """
    n = len(mean_returns)

    # Эйгенразложение ковариационной матрицы
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Максимальная дисперсия достигается при использовании собственного вектора
    # с максимальным собственным значением
    max_eigenvalue_idx = np.argmax(eigenvalues)
    max_eigenvalue = eigenvalues[max_eigenvalue_idx]
    max_eigenvector = eigenvectors[:, max_eigenvalue_idx]

    # Нормализация весов, чтобы сумма = 1
    weights = max_eigenvector / np.sum(max_eigenvector)

    # Если короткие продажи запрещены, веса должны быть неотрицательными
    # Для максимальной дисперсии решение часто содержит отрицательные веса
    # Мы принимаем это, так как в условии сказано "при отсутствии ограничений"

    # Расчет метрик портфеля
    portfolio_return = weights @ mean_returns
    portfolio_variance = weights @ cov_matrix @ weights
    portfolio_std = np.sqrt(portfolio_variance)

    return weights, portfolio_return, portfolio_std


def calculate_riskiest_frontier(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Построение границы наиболее рискованных портфелей.

    Это "обратная" задача к эффективной границе:
    Для каждого уровня доходности находим портфель с максимальным риском.

    Parameters:
    -----------
    mean_returns : np.ndarray
        Вектор средних доходностей
    cov_matrix : np.ndarray
        Ковариационная матрица
    n_points : int
        Количество точек на границе

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (доходности, стандартные отклонения)
    """
    n_assets = len(mean_returns)

    # Диапазон доходностей
    min_return = mean_returns.min()
    max_return = mean_returns.max()
    target_returns = np.linspace(min_return, max_return, n_points)

    # Для каждой целевой доходности находим портфель с максимальным риском
    # Это квадратичная оптимизация, которую решаем численно
    max_stds = []

    for target_return in target_returns:
        # Ограничение: Σ w_i = 1
        # Ограничение: Σ w_i * r_i = target_return
        # Максимизация: w^T * Σ * w

        # Решаем через двойственную задачу
        # Используем SVD для численного решения

        ones = np.ones(n_assets)
        cov_inv = np.linalg.inv(cov_matrix)

        # Коэффициенты
        A = ones.T @ cov_inv @ ones
        B = mean_returns.T @ cov_inv @ mean_returns
        C = mean_returns.T @ cov_inv @ ones
        D = A * B - C ** 2

        # Для заданной доходности решаем двойственную задачу минимизации
        # которая соответствует максимизации дисперсии

        # Нахождение весов, дающих целевую доходность
        # и максимизирующих дисперсию

        # Используем SVD для решения
        try:
            U, S, Vt = np.linalg.svd(cov_matrix)

            # Находим решение на первых двух собственных векторах
            # (как в задаче 22, но для максимизации)

            # Приближенное решение через оптимизацию
            # Максимальная дисперсия достигается на границе области допустимых весов

            # Используем численную оптимизацию
            from scipy.optimize import minimize

            def objective(w):
                return -(w @ cov_matrix @ w)  # Минимизация отрицательной дисперсии = максимизация

            def constraint_sum(w):
                return np.sum(w) - 1

            def constraint_return(w):
                return w @ mean_returns - target_return

            constraints = [
                {'type': 'eq', 'fun': constraint_sum},
                {'type': 'eq', 'fun': constraint_return}
            ]

            bounds = [(None, None)] * n_assets  # Без ограничений на веса

            # Начальное приближение
            x0 = np.ones(n_assets) / n_assets

            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'ftol': 1e-9, 'disp': False}
            )

            weights = result.x
            max_variance = -result.fun
            max_std = np.sqrt(max_variance)

        except ImportError:
            # Если scipy недоступен, используем альтернативный подход
            # Через эйгенвекторы
            weights, portfolio_return, portfolio_std = calculate_max_variance_portfolio(
                mean_returns, cov_matrix
            )
            # Пропорционально масштабируем под целевую доходность
            if portfolio_return > 0:
                weights = weights * target_return / portfolio_return
            max_std = portfolio_std

        max_stds.append(max_std)

    return target_returns, np.array(max_stds)


def compare_efficient_and_riskiest_frontiers(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    n_points: int = 50
) -> Dict[str, Dict]:
    """
    Сравнение эффективной границы и границы наиболее рискованных портфелей.

    Parameters:
    -----------
    mean_returns : np.ndarray
        Вектор средних доходностей
    cov_matrix : np.ndarray
        Ковариационная матрица
    n_points : int
        Количество точек на границах

    Returns:
    --------
    Dict[str, Dict]
        Сравнительные результаты
    """
    print("\n=== Сравнение эффективной и рисковой границ ===")

    # Эффективная граница (минимизация риска)
    print("Расчет эффективной границы...")
    ef_returns, ef_stds = calculate_efficient_frontier(
        mean_returns, cov_matrix, n_points=n_points
    )

    # Граница наиболее рискованных портфелей (максимизация риска)
    print("Расчет границы наиболее рискованных портфелей...")
    risk_returns, risk_stds = calculate_riskiest_frontier(
        mean_returns, cov_matrix, n_points=n_points
    )

    # Наиболее рискованный портфель (точка)
    riskiest_weights, riskiest_return, riskiest_std = calculate_max_variance_portfolio(
        mean_returns, cov_matrix
    )

    # Наименее рискованный портфель (GMVP)
    from task_9_10 import calculate_efficient_frontier_weights

    gmvp_weights = calculate_efficient_frontier_weights(
        mean_returns, cov_matrix, np.array([mean_returns.min()])
    )[0]
    gmvp_return = mean_returns.min()
    gmvp_std = ef_stds[0]

    print(f"\nНаименее рискованный портфель (GMVP):")
    print(f"  Доходность: {gmvp_return:.6f}")
    print(f"  Риск: {gmvp_std:.6f}")

    print(f"\nНаиболее рискованный портфель:")
    print(f"  Доходность: {riskiest_return:.6f}")
    print(f"  Риск: {riskiest_std:.6f}")

    print(f"\nОтношение рисков: {riskiest_std / gmvp_std:.2f}x")

    return {
        'efficient_frontier': {
            'returns': ef_returns,
            'stds': ef_stds,
            'min_std': gmvp_std,
            'max_std': ef_stds.max()
        },
        'riskiest_frontier': {
            'returns': risk_returns,
            'stds': risk_stds,
            'min_std': risk_stds.min(),
            'max_std': risk_stds.max()
        },
        'riskiest_portfolio': {
            'weights': riskiest_weights,
            'return': riskiest_return,
            'std': riskiest_std
        },
        'gmvp_portfolio': {
            'weights': gmvp_weights,
            'return': gmvp_return,
            'std': gmvp_std
        }
    }


def analyze_riskiest_portfolio_composition(
    weights: np.ndarray,
    ticker_names: List[str],
    top_n: int = 10
) -> pd.DataFrame:
    """
    Анализ состава наиболее рискованного портфеля.

    Parameters:
    -----------
    weights : np.ndarray
        Веса наиболее рискованного портфеля
    ticker_names : List[str]
        Названия акций
    top_n : int
        Количество акций для отображения

    Returns:
    --------
    pd.DataFrame
        DataFrame с весами топ-N акций
    """
    # Сортировка по абсолютному весу
    abs_weights = np.abs(weights)
    sorted_indices = np.argsort(abs_weights)[::-1]

    composition = []
    for i, idx in enumerate(sorted_indices[:top_n]):
        composition.append({
            'ticker': ticker_names[idx],
            'weight': weights[idx],
            'abs_weight': abs_weights[idx],
            'rank': i + 1
        })

    df = pd.DataFrame(composition)

    # Добавляем информацию о коротких позициях
    short_positions = df[df['weight'] < 0]
    long_positions = df[df['weight'] >= 0]

    print(f"\n=== Состав наиболее рискованного портфеля ===")
    print(f"Лонг-позиции (вес >= 0): {len(long_positions)}")
    print(long_positions)
    print(f"\nШорт-позиции (вес < 0): {len(short_positions)}")
    print(short_positions)
    print(f"\nСумма лонг-позиций: {long_positions['weight'].sum():.4f}")
    print(f"Сумма шорт-позиций: {short_positions['weight'].sum():.4f}")

    return df


def main():
    """
    Главная функция для демонстрации наиболее рискованного портфеля.
    """
    print("=" * 60)
    print("Задача 24 (****): Наиболее рискованный портфель")
    print("=" * 60)
    print()

    # Загрузка данных
    print("Загрузка данных...")
    prices = load_prices_data('../data/prices_moex_new.csv')
    returns = calculate_returns(prices)

    # Используем часть данных (первый год)
    returns_data = returns.iloc[:251]

    # Исключаем MOEX (индекс)
    ticker_cols = [col for col in returns_data.columns if col != 'MOEX']
    returns_data = returns_data[ticker_cols]

    mean_returns = returns_data.mean().values
    cov_matrix = returns_data.cov().values

    print(f"Размер: {returns_data.shape}")
    print(f"Количество акций: {len(ticker_cols)}")

    # Расчет наиболее рискованного портфеля
    print("\n=== Наиболее рискованный портфель ===")
    riskiest_weights, riskiest_return, riskiest_std = calculate_max_variance_portfolio(
        mean_returns, cov_matrix
    )

    print(f"Доходность: {riskiest_return:.6f}")
    print(f"Стандартное отклонение: {riskiest_std:.6f}")
    print(f"Дисперсия: {riskiest_std**2:.8f}")

    # Анализ состава
    analyze_riskiest_portfolio_composition(riskiest_weights, ticker_cols, top_n=10)

    # Сравнение границ
    print("\n=== Сравнение эффективной и рисковой границ ===")
    comparison = compare_efficient_and_riskiest_frontiers(
        mean_returns, cov_matrix, n_points=50
    )

    return comparison


if __name__ == '__main__':
    results = main()
