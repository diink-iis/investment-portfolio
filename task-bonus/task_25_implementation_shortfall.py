"""
Задача 25 (*****): Implementation Shortfall в портфельной оптимизации

Встроить Implementation Shortfall, рассчитанный для дневных цен закрытия,
в оптимизационную задачу.
"""

import numpy as np
import pandas as pd
import sys
from typing import Tuple, Dict
from scipy import optimize

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


def calculate_implementation_shortfall(
    returns: pd.DataFrame,
    weights: np.ndarray,
    benchmark: float = 0.0
) -> float:
    """
    Расчет Implementation Shortfall (IS) для портфеля.

    IS = E[min(R - B, 0)] = average of negative deviations from benchmark

    Parameters:
    -----------
    returns : pd.DataFrame
        Исторические доходности всех активов
    weights : np.ndarray
        Веса портфеля
    benchmark : float
        Пороговый уровень (бенчмарк), обычно 0

    Returns:
    --------
    float
        Implementation Shortfall
    """
    # Доходность портфеля для каждого наблюдения
    portfolio_returns = returns @ weights

    # IS = среднее из max(B - R, 0)
    # Это среднее отрицательных отклонений от бенчмарка
    is_values = np.maximum(benchmark - portfolio_returns, 0)
    is_mean = np.mean(is_values)

    return is_mean


def calculate_is_efficient_frontier(
    returns: pd.DataFrame,
    benchmark: float = 0.0,
    n_points: int = 100,
    allow_short_sales: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Построение IS-эффективной границы.

    Для каждого уровня доходности находим портфель с минимальным IS.

    Parameters:
    -----------
    returns : pd.DataFrame
        Исторические доходности всех активов
    benchmark : float
        Пороговый уровень
    n_points : int
        Количество точек на границе
    allow_short_sales : bool
        Разрешены ли короткие продажи

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (доходности, стандартные отклонения, веса портфелей)
    """
    n_assets = len(returns.columns)
    mean_returns = returns.mean().values

    # Диапазон целевых доходностей
    min_return = mean_returns.min()
    max_return = mean_returns.max()
    target_returns = np.linspace(min_return, max_return, n_points)

    frontier_weights = []
    frontier_returns = []
    frontier_stds = []

    print(f"IS-эффективная граница:")
    print(f"  Количество активов: {n_assets}")
    print(f"  Бенчмарк: {benchmark:.4f}")
    print(f"  Короткие продажи: {'разрешены' if allow_short_sales else 'запрещены'}")

    for i, target_return in enumerate(target_returns):
        # Ограничения
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Сумма весов = 1
            {'type': 'eq', 'fun': lambda w: w @ mean_returns - target_return}  # Доходность = target_return
        ]

        bounds = [(None, None)] * n_assets if allow_short_sales else [(0, None)] * n_assets

        # Начальное приближение
        x0 = np.ones(n_assets) / n_assets

        # Целевая функция: минимизировать IS
        def objective(w):
            return calculate_implementation_shortfall(returns, w, benchmark)

        # Оптимизация
        result = optimize.minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )

        if result.success:
            w = result.x
            is_value = result.fun

            # Расчет стандартного отклонения
            portfolio_var = w @ (returns.cov() @ w)
            portfolio_std = np.sqrt(portfolio_var)

            frontier_weights.append(w)
            frontier_returns.append(target_return)
            frontier_stds.append(portfolio_std)
        else:
            print(f"⚠ Оптимизация не сошлась для точки {i}")

    return np.array(frontier_returns), np.array(frontier_stds), np.array(frontier_weights)


def compare_risk_measures(
    returns: pd.DataFrame,
    weights: np.ndarray,
    benchmark: float = 0.0
) -> Dict[str, float]:
    """
    Сравнение различных мер риска для одного портфеля.

    Parameters:
    -----------
    returns : pd.DataFrame
        Исторические доходности
    weights : np.ndarray
        Веса портфеля
    benchmark : float
        Бенчмарк для IS

    Returns:
    --------
    Dict[str, float]
        Словарь с различными мерами риска
    """
    portfolio_returns = returns @ weights

    # Дисперсия (классическая мера)
    variance = np.var(portfolio_returns, ddof=1)
    std = np.sqrt(variance)

    # Implementation Shortfall
    is_value = calculate_implementation_shortfall(returns, weights, benchmark)

    # Conditional Value-at-Risk (CVaR) на уровне 95%
    var_95 = np.percentile(portfolio_returns, 5)
    cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()

    # Semi-variance (учитывает только отрицательные доходности)
    negative_returns = portfolio_returns[portfolio_returns < 0]
    semi_variance = np.mean(negative_returns ** 2) if len(negative_returns) > 0 else 0
    semi_std = np.sqrt(semi_variance)

    return {
        'variance': variance,
        'std': std,
        'implementation_shortfall': is_value,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'semi_variance': semi_variance,
        'semi_std': semi_std
    }


def compare_frontiers_by_risk_measure(
    returns: pd.DataFrame,
    benchmark: float = 0.0,
    n_points: int = 50,
    allow_short_sales: bool = True
) -> Dict[str, Dict]:
    """
    Сравнение эффективных границ, рассчитанных разными мерами риска.

    Parameters:
    -----------
    returns : pd.DataFrame
        Исторические доходности
    benchmark : float
        Бенчмарк для IS
    n_points : int
        Количество точек на границах
    allow_short_sales : bool
        Разрешены ли короткие продажи

    Returns:
    --------
    Dict[str, Dict]
        Сравнительные результаты
    """
    print("\n=== Сравнение границ по мерам риска ===")

    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values

    # Классическая эффективная граница (минимизация дисперсии)
    print("Расчет классической эффективной границы (дисперсия)...")
    ef_variance_returns, ef_variance_stds, ef_variance_weights = calculate_efficient_frontier(
        mean_returns, cov_matrix, n_points=n_points
    )

    # IS-эффективная граница (минимизация IS)
    print("Расчет IS-эффективной границы...")
    ef_is_returns, ef_is_stds, ef_is_weights = calculate_is_efficient_frontier(
        returns, benchmark=benchmark, n_points=n_points,
        allow_short_sales=allow_short_sales
    )

    # Сравнение ключевых метрик
    print("\n=== Сравнение ключевых метрик ===")

    # GMVP для обеих границ
    gmvp_variance_weight = calculate_efficient_frontier_weights(
        mean_returns, cov_matrix, np.array([ef_variance_returns.min()])
    )[0]
    gmvp_is_weight = ef_is_weights[0]

    risk_comparison_variance = compare_risk_measures(returns, gmvp_variance_weight, benchmark)
    risk_comparison_is = compare_risk_measures(returns, gmvp_is_weight, benchmark)

    print(f"\nGMVP (дисперсия):")
    print(f"  Std: {risk_comparison_variance['std']:.6f}")
    print(f"  IS: {risk_comparison_variance['implementation_shortfall']:.6f}")
    print(f"  Semi-std: {risk_comparison_variance['semi_std']:.6f}")

    print(f"\nGMVP (IS):")
    print(f"  Std: {risk_comparison_is['std']:.6f}")
    print(f"  IS: {risk_comparison_is['implementation_shortfall']:.6f}")
    print(f"  Semi-std: {risk_comparison_is['semi_std']:.6f}")

    return {
        'variance_frontier': {
            'returns': ef_variance_returns,
            'stds': ef_variance_stds,
            'weights': ef_variance_weights,
            'gmvp_risk': risk_comparison_variance
        },
        'is_frontier': {
            'returns': ef_is_returns,
            'stds': ef_is_stds,
            'weights': ef_is_weights,
            'gmvp_risk': risk_comparison_is
        }
    }


def test_is_sensitivity_to_benchmark(
    returns: pd.DataFrame,
    benchmark_values: np.ndarray = np.array([-0.01, 0.0, 0.01, 0.02]),
    n_points: int = 30
) -> Dict[float, Dict]:
    """
    Тестирование чувствительности IS-эффективной границы к выбору бенчмарка.

    Parameters:
    -----------
    returns : pd.DataFrame
        Исторические доходности
    benchmark_values : np.ndarray
        Различные уровни бенчмарка
    n_points : int
        Количество точек на границе

    Returns:
    --------
    Dict[float, Dict]
        Результаты для разных бенчмарков
    """
    print("\n=== Тестирование чувствительности к бенчмарку ===")

    results = {}

    for benchmark in benchmark_values:
        print(f"\nБенчмарк: {benchmark:.4f}")

        # IS-эффективная граница
        ef_returns, ef_stds, ef_weights = calculate_is_efficient_frontier(
            returns, benchmark=benchmark, n_points=n_points, allow_short_sales=True
        )

        # GMVP
        gmvp_weight = ef_weights[0]
        gmvp_is = calculate_implementation_shortfall(returns, gmvp_weight, benchmark)

        results[benchmark] = {
            'returns': ef_returns,
            'stds': ef_stds,
            'gmvp_is': gmvp_is,
            'min_is': np.min([calculate_implementation_shortfall(returns, w, benchmark)
                               for w in ef_weights]),
            'max_is': np.max([calculate_implementation_shortfall(returns, w, benchmark)
                               for w in ef_weights])
        }

        print(f"  GMVP IS: {gmvp_is:.6f}")
        print(f"  Мин. IS: {results[benchmark]['min_is']:.6f}")
        print(f"  Макс. IS: {results[benchmark]['max_is']:.6f}")

    return results


def main():
    """
    Главная функция для демонстрации Implementation Shortfall.
    """
    print("=" * 60)
    print("Задача 25 (*****): Implementation Shortfall в оптимизации")
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

    print(f"Размер: {returns_data.shape}")
    print(f"Количество акций: {len(ticker_cols)}")

    # Тестирование IS для разных бенчмарков
    print("\n=== Тестирование IS для разных бенчмарков ===")
    benchmark_results = test_is_sensitivity_to_benchmark(
        returns_data, benchmark_values=np.array([-0.01, 0.0, 0.01, 0.02]),
        n_points=30
    )

    # Сравнение границ по мерам риска
    print("\n=== Сравнение границ ===")
    frontier_comparison = compare_frontiers_by_risk_measure(
        returns_data, benchmark=0.0, n_points=50, allow_short_sales=True
    )

    return benchmark_results, frontier_comparison


if __name__ == '__main__':
    results, comparison = main()
