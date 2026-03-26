"""
Задача 23 (***): Метод Монте-Карло для границы эффективных портфелей

Рассчитать границу эффективных портфелей (для отобранных акций)
с помощью метода статистических испытаний (Монте-Карло).
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


def generate_random_portfolios(
    n_assets: int,
    n_simulations: int,
    allow_short_sales: bool = False
) -> np.ndarray:
    """
    Генерация случайных портфелей.

    Parameters:
    -----------
    n_assets : int
        Количество активов
    n_simulations : int
        Количество симуляций
    allow_short_sales : bool
        Разрешены ли короткие продажи

    Returns:
    --------
    np.ndarray
        Матрица весов (n_simulations x n_assets)
    """
    if allow_short_sales:
        # Короткие продажи разрешены: веса могут быть любыми
        weights = np.random.randn(n_simulations, n_assets)
    else:
        # Короткие продажи запрещены: веса должны быть неотрицательными
        weights = np.random.rand(n_simulations, n_assets)

    # Нормализация, чтобы сумма весов = 1
    weights = weights / weights.sum(axis=1, keepdims=True)

    return weights


def calculate_portfolio_metrics(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Расчет доходностей и рисков для портфелей.

    Parameters:
    -----------
    weights : np.ndarray
        Матрица весов (n_simulations x n_assets)
    mean_returns : np.ndarray
        Вектор средних доходностей
    cov_matrix : np.ndarray
        Ковариационная матрица

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (доходности портфелей, риски портфелей)
    """
    # Доходности: r_p = w * r̄
    portfolio_returns = weights @ mean_returns

    # Дисперсии: σ²_p = w * Σ * w^T
    # Ускоренный расчет для многих портфелей
    portfolio_variances = np.einsum('ij,ij->i', weights, cov_matrix @ weights.T)

    return portfolio_returns, portfolio_variances


def find_efficient_portfolios(
    portfolio_returns: np.ndarray,
    portfolio_variances: np.ndarray,
    n_bins: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Отбор недоминируемых портфелей из случайных.

    Портфель A доминирует портфель B, если:
    r_A >= r_B и σ_A <= σ_B

    Parameters:
    -----------
    portfolio_returns : np.ndarray
        Доходности портфелей
    portfolio_variances : np.ndarray
        Риски портфелей
    n_bins : int
        Количество бинов для отбора

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (эффективные доходности, эффективные риски)
    """
    n_portfolios = len(portfolio_returns)
    portfolio_stds = np.sqrt(portfolio_variances)

    # Сортировка по доходности
    sorted_indices = np.argsort(portfolio_returns)
    sorted_returns = portfolio_returns[sorted_indices]
    sorted_stds = portfolio_stds[sorted_indices]

    # Отбор недоминируемых портфелей
    efficient_mask = np.ones(n_portfolios, dtype=bool)

    for i in range(n_portfolios):
        if not efficient_mask[i]:
            continue

        # Портфель i доминирует все портфели j с меньшей доходностью и большим риском
        for j in range(i):
            if efficient_mask[j]:
                if sorted_returns[i] >= sorted_returns[j] and sorted_stds[i] <= sorted_stds[j]:
                    # Портфель i доминирует j
                    efficient_mask[j] = False

    # Фильтрация эффективных портфелей
    efficient_returns = sorted_returns[efficient_mask]
    efficient_stds = sorted_stds[efficient_mask]

    return efficient_returns, efficient_stds


def monte_carlo_efficient_frontier(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    n_simulations: int = 10000,
    allow_short_sales: bool = True,
    n_bins: int = 100
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Расчет эффективной границы методом Монте-Карло.

    Parameters:
    -----------
    mean_returns : np.ndarray
        Вектор средних доходностей
    cov_matrix : np.ndarray
        Ковариационная матрица
    n_simulations : int
        Количество симуляций
    allow_short_sales : bool
        Разрешены ли короткие продажи
    n_bins : int
        Количество бинов для отбора

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]
        (эффективные доходности, эффективные риски, метаданные)
    """
    n_assets = len(mean_returns)

    print(f"Метод Монте-Карло:")
    print(f"  Количество активов: {n_assets}")
    print(f"  Количество симуляций: {n_simulations}")
    print(f"  Короткие продажи: {'разрешены' if allow_short_sales else 'запрещены'}")

    # Генерация случайных портфелей
    weights = generate_random_portfolios(n_assets, n_simulations, allow_short_sales)

    # Расчет метрик
    portfolio_returns, portfolio_variances = calculate_portfolio_metrics(
        weights, mean_returns, cov_matrix
    )

    # Отбор недоминируемых портфелей
    efficient_returns, efficient_stds = find_efficient_portfolios(
        portfolio_returns, portfolio_variances, n_bins
    )

    print(f"  Эффективных портфелей: {len(efficient_returns)} из {n_simulations}")
    print(f"  Мин. доходность: {efficient_returns.min():.6f}, Макс. доходность: {efficient_returns.max():.6f}")
    print(f"  Мин. риск: {efficient_stds.min():.6f}, Макс. риск: {efficient_stds.max():.6f}")

    # Сортировка по риску для построения границы
    sorted_indices = np.argsort(efficient_stds)
    ef_returns = efficient_returns[sorted_indices]
    ef_stds = efficient_stds[sorted_indices]

    # Шарп-отношение
    sharpe_ratios = ef_returns / ef_stds
    max_sharpe_idx = np.argmax(sharpe_ratios)

    metadata = {
        'n_simulations': n_simulations,
        'n_efficient': len(efficient_returns),
        'all_weights': weights,
        'all_returns': portfolio_returns,
        'all_variances': portfolio_variances,
        'max_sharpe_return': ef_returns[max_sharpe_idx],
        'max_sharpe_std': ef_stds[max_sharpe_idx],
        'max_sharpe': sharpe_ratios[max_sharpe_idx]
    }

    return ef_returns, ef_stds, metadata


def compare_monte_carlo_vs_analytical(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    n_simulations: int = 10000,
    n_analytical_points: int = 100
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Сравнение метода Монте-Карло с аналитическим решением.

    Parameters:
    -----------
    mean_returns : np.ndarray
        Вектор средних доходностей
    cov_matrix : np.ndarray
        Ковариационная матрица
    n_simulations : int
        Количество симуляций
    n_analytical_points : int
        Количество точек аналитической границы

    Returns:
    --------
    Dict[str, Dict[str, np.ndarray]]
        Сравнительные результаты
    """
    print("\n=== Сравнение методов ===")

    # Аналитическое решение
    print("Аналитическое решение...")
    analytical_returns, analytical_stds = calculate_efficient_frontier(
        mean_returns, cov_matrix, n_points=n_analytical_points
    )

    # Метод Монте-Карло
    print("Метод Монте-Карло...")
    mc_returns, mc_stds, mc_metadata = monte_carlo_efficient_frontier(
        mean_returns, cov_matrix, n_simulations=n_simulations,
        allow_short_sales=True, n_bins=50
    )

    # Сравнение по ключевым метрикам
    # Минимальный риск
    min_risk_analytical = analytical_stds.min()
    min_risk_mc = mc_stds.min()

    # Максимальное Шарп-отношение
    max_sharpe_analytical = np.max(analytical_returns / analytical_stds)
    max_sharpe_mc = np.max(mc_returns / mc_stds)

    print(f"\n=== Сравнение ===")
    print(f"Минимальный риск:")
    print(f"  Аналитическое: {min_risk_analytical:.6f}")
    print(f"  Монте-Карло: {min_risk_mc:.6f}")
    print(f"  Разница: {abs(min_risk_mc - min_risk_analytical):.6f}")

    print(f"\nМаксимальное Шарп-отношение:")
    print(f"  Аналитическое: {max_sharpe_analytical:.6f}")
    print(f"  Монте-Карло: {max_sharpe_mc:.6f}")
    print(f"  Разница: {abs(max_sharpe_mc - max_sharpe_analytical):.6f}")

    return {
        'analytical': {
            'returns': analytical_returns,
            'stds': analytical_stds,
            'min_std': min_risk_analytical,
            'max_sharpe': max_sharpe_analytical
        },
        'monte_carlo': {
            'returns': mc_returns,
            'stds': mc_stds,
            'metadata': mc_metadata,
            'min_std': min_risk_mc,
            'max_sharpe': max_sharpe_mc
        },
        'difference': {
            'min_std': abs(min_risk_mc - min_risk_analytical),
            'max_sharpe': abs(max_sharpe_mc - max_sharpe_analytical)
        }
    }


def main():
    """
    Главная функция для демонстрации метода Монте-Карло.
    """
    print("=" * 60)
    print("Задача 23 (***): Метод Монте-Карло для эффективной границы")
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

    # Метод Монте-Карло с разным количеством симуляций
    print("\n=== Тестирование с разным количеством симуляций ===")

    n_simulations_list = [1000, 5000, 10000, 50000]
    results = {}

    for n_sim in n_simulations_list:
        print(f"\n--- {n_sim} симуляций ---")
        mc_returns, mc_stds, metadata = monte_carlo_efficient_frontier(
            mean_returns, cov_matrix, n_simulations=n_sim,
            allow_short_sales=True, n_bins=100
        )
        results[n_sim] = {
            'returns': mc_returns,
            'stds': mc_stds,
            'metadata': metadata
        }

    # Сравнение с аналитическим решением
    print("\n=== Сравнение с аналитическим решением ===")
    comparison = compare_monte_carlo_vs_analytical(
        mean_returns, cov_matrix, n_simulations=10000,
        n_analytical_points=100
    )

    return results, comparison


if __name__ == '__main__':
    results, comparison = main()
