"""
Задача 22 (**): Проверить Black's (1972) Two-Fund Theorem

Теорема: все портфели на границе портфелей с минимальной дисперсией
являются линейной комбинацией любых двух других портфелей на этой границе
(при условии, что короткие продажи разрешены).
"""

import numpy as np
import pandas as pd
import sys
from typing import Tuple, Dict, List

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


def calculate_portfolio_weights(
    returns: np.ndarray,
    cov_matrix: np.ndarray,
    target_return: float
) -> np.ndarray:
    """
    Расчет оптимальных весов портфеля для заданной доходности
    без ограничений на короткие продажи.

    Parameters:
    -----------
    returns : np.ndarray
        Вектор средних доходностей
    cov_matrix : np.ndarray
        Ковариационная матрица
    target_return : float
        Целевая доходность

    Returns:
    --------
    np.ndarray
        Оптимальные веса портфеля
    """
    n = len(returns)

    # Вычисление коэффициентов
    ones = np.ones(n)
    cov_inv = np.linalg.inv(cov_matrix)

    A = ones.T @ cov_inv @ ones
    B = returns.T @ cov_inv @ returns
    C = returns.T @ cov_inv @ ones
    D = A * B - C ** 2

    # Веса для заданной доходности
    g = (B * cov_inv @ ones - C * cov_inv @ returns) / D
    h = (C * cov_inv @ ones - A * cov_inv @ returns) / D

    w = g + h * target_return

    return w


def verify_two_fund_theorem(
    w1: np.ndarray,
    w2: np.ndarray,
    w3: np.ndarray,
    tolerance: float = 1e-6
) -> Tuple[float, bool]:
    """
    Проверка Two-Fund Theorem: портфель 3 должен быть линейной комбинацией
    портфелей 1 и 2.

    Решает систему уравнений:
    w3 = lambda * w1 + (1 - lambda) * w2

    Parameters:
    -----------
    w1, w2, w3 : np.ndarray
        Веса трех портфелей
    tolerance : float
        Допустимая ошибка

    Returns:
    --------
    Tuple[float, bool]
        (коэффициент lambda, совпадает ли)
    """
    n = len(w1)

    # Решение системы: w3 = lambda * w1 + (1 - lambda) * w2
    # lambda * w1 + (1 - lambda) * w2 = w3
    # lambda * (w1 - w2) = w3 - w2
    # lambda * (w1 - w2) = w3 - w2

    diff = w1 - w2

    # Если diff близок к 0, то w1 ≈ w2
    if np.allclose(diff, 0, atol=tolerance):
        # Любое lambda подойдет
        return 0.5, True

    # Решаем для каждого веса
    lambdas = []

    for i in range(n):
        if abs(diff[i]) > tolerance:
            lambda_i = (w3[i] - w2[i]) / diff[i]
            lambdas.append(lambda_i)

    if len(lambdas) == 0:
        return 0.5, True

    # Проверяем, что все lambda примерно равны
    lambda_mean = np.mean(lambdas)
    lambda_max = np.max(lambdas)
    lambda_min = np.min(lambdas)

    matches = np.allclose(lambdas, lambda_mean, atol=tolerance)

    return lambda_mean, matches


def test_two_fund_theorem(
    returns: np.ndarray,
    cov_matrix: np.ndarray,
    n_test_portfolios: int = 5,
    tolerance: float = 1e-6
) -> Dict[str, List]:
    """
    Проверка Two-Fund Theorem на нескольких портфелях.

    Parameters:
    -----------
    returns : np.ndarray
        Вектор средних доходностей
    cov_matrix : np.ndarray
        Ковариационная матрица
    n_test_portfolios : int
        Количество проверяемых портфелей
    tolerance : float
        Допустимая ошибка

    Returns:
    --------
    Dict[str, List]
        Словарь с результатами проверок
    """
    print(f"Проверка Black's Two-Fund Theorem...")
    print(f"Количество тестовых портфелей: {n_test_portfolios}")

    # Расчет эффективной границы
    ef_returns, ef_stds = calculate_efficient_frontier(
        returns, cov_matrix, n_points=50
    )

    # Выбираем два портфеля на границе
    idx1 = len(ef_returns) // 3  # Нижний левый угол (GMVP)
    idx2 = 2 * len(ef_returns) // 3  # Верхний правый угол

    # Веса для этих двух портфелей
    w1 = calculate_portfolio_weights(returns, cov_matrix, ef_returns[idx1])
    w2 = calculate_portfolio_weights(returns, cov_matrix, ef_returns[idx2])

    # Тестируем несколько других портфелей
    results = []

    test_indices = np.linspace(0, len(ef_returns) - 1, n_test_portfolios, dtype=int)

    for i, test_idx in enumerate(test_indices):
        if test_idx == idx1 or test_idx == idx2:
            continue

        # Веса для тестового портфеля
        w3 = calculate_portfolio_weights(returns, cov_matrix, ef_returns[test_idx])

        # Проверяем Two-Fund Theorem
        lambda_coef, matches = verify_two_fund_theorem(w1, w2, w3, tolerance)

        # Проверяем, что комбинация действительно дает портфель
        w_combined = lambda_coef * w1 + (1 - lambda_coef) * w2
        combined_correct = np.allclose(w_combined, w3, atol=tolerance)

        results.append({
            'test_portfolio': i + 1,
            'idx': test_idx,
            'target_return': ef_returns[test_idx],
            'target_std': ef_stds[test_idx],
            'lambda': lambda_coef,
            'matches_theorem': matches,
            'combined_correct': combined_correct,
            'w1_mean_return': ef_returns[idx1],
            'w1_std': ef_stds[idx1],
            'w2_mean_return': ef_returns[idx2],
            'w2_std': ef_stds[idx2]
        })

    print(f"\nРезультаты проверки:")
    for result in results:
        print(f"Портфель {result['test_portfolio']}: ")
        print(f"  Доходность: {result['target_return']:.6f}, Риск: {result['target_std']:.6f}")
        print(f"  Lambda: {result['lambda']:.6f}")
        print(f"  Совпадает с теоремой: {result['matches_theorem']}")
        print(f"  Комбинация корректна: {result['combined_correct']}")

    # Статистика по проверкам
    matches_count = sum(1 for r in results if r['matches_theorem'])
    print(f"\n=== Статистика ===")
    print(f"Успешных проверок: {matches_count}/{len(results)}")
    print(f"Процент успеха: {100*matches_count/len(results):.1f}%")

    return {
        'w1': w1,
        'w2': w2,
        'w1_return': ef_returns[idx1],
        'w1_std': ef_stds[idx1],
        'w2_return': ef_returns[idx2],
        'w2_std': ef_stds[idx2],
        'ef_returns': ef_returns,
        'ef_stds': ef_stds,
        'test_results': results,
        'success_rate': 100*matches_count/len(results)
    }


def two_fund_theorem_proof(
    returns: np.ndarray,
    cov_matrix: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Математическое доказательство Two-Fund Theorem.

    Любые три портфеля на эффективной границе P1, P2, P3
    могут быть представлены как P3 = w * P1 + (1-w) * P2.

    Parameters:
    -----------
    returns : np.ndarray
        Вектор средних доходностей
    cov_matrix : np.ndarray
        Ковариационная матрица

    Returns:
    --------
    Dict[str, np.ndarray]
        Веса для демонстрации теоремы
    """
    n = len(returns)

    # Эффективная граница
    ef_returns, ef_stds = calculate_efficient_frontier(
        returns, cov_matrix, n_points=3
    )

    # Веса для трех портфелей
    w1 = calculate_portfolio_weights(returns, cov_matrix, ef_returns[0])
    w2 = calculate_portfolio_weights(returns, cov_matrix, ef_returns[1])
    w3 = calculate_portfolio_weights(returns, cov_matrix, ef_returns[2])

    # Решаем для lambda
    # w3 = lambda * w1 + (1 - lambda) * w2
    diff = w1 - w2

    if np.allclose(diff, 0):
        lambda_coef = 0.5
    else:
        # Решаем уравнение: lambda * diff = w3 - w2
        lambda_coef = np.mean((w3 - w2) / diff[~np.isclose(diff, 0)])

    # Проверяем
    w_combined = lambda_coef * w1 + (1 - lambda_coef) * w2

    # Расчет метрик
    w1_return = w1 @ returns
    w1_var = w1 @ cov_matrix @ w1
    w1_std = np.sqrt(w1_var)

    w2_return = w2 @ returns
    w2_var = w2 @ cov_matrix @ w2
    w2_std = np.sqrt(w2_var)

    w3_return = w3 @ returns
    w3_var = w3 @ cov_matrix @ w3
    w3_std = np.sqrt(w3_var)

    return {
        'w1': w1,
        'w2': w2,
        'w3': w3,
        'lambda': lambda_coef,
        'w_combined': w_combined,
        'w3_matches': np.allclose(w_combined, w3),
        'w1_return': w1_return,
        'w1_std': w1_std,
        'w2_return': w2_return,
        'w2_std': w2_std
    }


def main():
    """
    Главная функция для демонстрации Two-Fund Theorem.
    """
    print("=" * 60)
    print("Задача 22 (**): Проверка Black's Two-Fund Theorem")
    print("=" * 60)
    print()

    # Загрузка данных
    print("Загрузка данных...")
    prices = load_prices_data('../data/prices_moex_new.csv')
    returns = calculate_returns(prices)

    # Используем часть данных (первый год)
    returns_data = returns.iloc[:251]
    mean_returns = returns_data.mean().values
    cov_matrix = returns_data.cov().values

    # Исключаем MOEX (индекс)
    ticker_cols = [col for col in returns_data.columns if col != 'MOEX']
    returns_data = returns_data[ticker_cols]
    mean_returns = returns_data.mean().values
    cov_matrix = returns_data.cov().values

    print(f"Размер: {returns_data.shape}")
    print(f"Количество акций: {len(ticker_cols)}")

    # Тестирование Two-Fund Theorem
    print("\n=== Тестирование Two-Fund Theorem ===")
    results = test_two_fund_theorem(
        mean_returns, cov_matrix,
        n_test_portfolios=7,
        tolerance=1e-6
    )

    # Математическое доказательство
    print("\n=== Математическое доказательство ===")
    proof = two_fund_theorem_proof(mean_returns, cov_matrix)

    print(f"\nПортфель 1 (GMVP):")
    print(f"  Доходность: {proof['w1_return']:.6f}, Риск: {proof['w1_std']:.6f}")
    print(f"  Веса (первые 5): {proof['w1'][:5]}")

    print(f"\nПортфель 2 (макс. доходность):")
    print(f"  Доходность: {proof['w2_return']:.6f}, Риск: {proof['w2_std']:.6f}")
    print(f"  Веса (первые 5): {proof['w2'][:5]}")

    print(f"\nПортфель 3 (промежуточный):")
    print(f"  Доходность: {proof['w3'] @ mean_returns:.6f}, Риск: {np.sqrt(proof['w3'] @ cov_matrix @ proof['w3']):.6f}")
    print(f"  Веса (первые 5): {proof['w3'][:5]}")

    print(f"\nКоэффициент lambda: {proof['lambda']:.6f}")
    print(f"Совпадение: {proof['w3_matches']}")

    if proof['w3_matches']:
        print("\n✓ Two-Fund Theorem подтверждена!")
    else:
        print("\n⚠ Two-Fund Theorem не подтверждена")

    return results


if __name__ == '__main__':
    results = main()
