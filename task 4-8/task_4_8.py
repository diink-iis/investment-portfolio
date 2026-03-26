"""
Задачи 4-8: Анализ влияния ограничений в optimizer на границу эффективных портфелей.

Пункт 4:
Выбрать одно историческое окно и схему взвешивания наблюдений
для дальнейшего расчета эффективных границ. По умолчанию:
- скользящее окно длиной 1 год,
- шаг 1 год,
- экспоненциальное забывание с lambda = 0.94,
- последнее доступное окно по дате в выборке.

Пункты 5-8:
Построить эффективные границы для четырех сценариев ограничений:
5. Short allowed without limits
6. Short allowed, but each short position <= 25% of capital
7. No short selling
8. Each asset weight >= 2%
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from typing import Dict, Tuple, List, Optional


# ============================================================
# БЛОК 1. БАЗОВЫЕ ФУНКЦИИ ДЛЯ РАБОТЫ С ДАННЫМИ
# ============================================================

def load_prices_data(file_path: str) -> pd.DataFrame:
    """
    Загружает таблицу цен акций из CSV.

    Ожидаемый формат:
    - разделитель ';'
    - десятичная запятая ','
    - столбец date в формате dd.mm.yyyy
    """
    df = pd.read_csv(file_path, sep=';', decimal=',')
    df.columns = df.columns.str.strip()
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
    df = df.set_index('date')
    df = df.replace('', np.nan).replace(' ', np.nan)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')
    df = df.sort_index()

    return df


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Считает логарифмические дневные доходности.
    """
    returns = np.log(prices / prices.shift(1))
    returns = returns.dropna(how='all')
    return returns


def calculate_exponential_weights(n: int, lambda_param: float = 0.94) -> np.ndarray:
    """
    Считает экспоненциальные веса для схемы forgetting.
    Чем ближе наблюдение к концу окна, тем выше его вес.
    """
    weights = np.array([lambda_param ** (n - i - 1) for i in range(n)], dtype=float)
    weights = weights / weights.sum()
    return weights


def calculate_weighted_covariance(returns: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    """
    Считает взвешенную ковариационную матрицу.
    """
    values = returns.values
    weighted_mean = np.average(values, axis=0, weights=weights)
    centered = values - weighted_mean
    cov = (centered * weights[:, None]).T @ centered
    return cov


def _period_to_days(period: str) -> int:
    """
    Переводит строку периода в календарные дни.
    """
    mapping = {
        '1Y': 365,
        '1Q': 91,
        '1M': 30,
        '1W': 7,
        '1D': 1,
    }
    if period not in mapping:
        raise ValueError(f"Неизвестный период: {period}")
    return mapping[period]


def _prepare_window_returns(window_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Готовит данные внутри конкретного окна:
    - удаляет даты, где хотя бы по одной бумаге нет доходности,
      чтобы mean/cov были рассчитаны на согласованной матрице наблюдений;
    - не выбрасывает лишние даты глобально по всей выборке.
    """
    if window_returns.empty:
        return window_returns
    return window_returns.dropna(how='any')


def rolling_window_analysis(
    returns: pd.DataFrame,
    window_size: str = '1Y',
    step_size: str = '1Y',
    lambda_param: Optional[float] = None
) -> Dict[pd.Timestamp, dict]:
    """
    Считает mean returns и covariance matrix на скользящих окнах.
    """
    window_days = _period_to_days(window_size)
    step_days = _period_to_days(step_size)

    results: Dict[pd.Timestamp, dict] = {}
    dates = returns.index.sort_values()

    if len(dates) == 0:
        return results

    current_end = dates.min() + pd.Timedelta(days=window_days)

    while current_end <= dates.max():
        current_start = current_end - pd.Timedelta(days=window_days)
        window_returns = returns.loc[(returns.index > current_start) & (returns.index <= current_end)]
        window_returns = _prepare_window_returns(window_returns)

        if len(window_returns) >= 2:
            if lambda_param is None:
                mean_returns = window_returns.mean().values
                cov_matrix = window_returns.cov().values
            else:
                weights = calculate_exponential_weights(len(window_returns), lambda_param)
                mean_returns = np.average(window_returns.values, axis=0, weights=weights)
                cov_matrix = calculate_weighted_covariance(window_returns, weights)

            results[current_end] = {
                'window_start': current_start,
                'window_end': current_end,
                'window_returns': window_returns,
                'window_size': len(window_returns),
                'mean_returns': mean_returns,
                'covariance_matrix': cov_matrix
            }

        current_end = current_end + pd.Timedelta(days=step_days)

    return results


def expanding_window_analysis(
    returns: pd.DataFrame,
    step_size: str = '1Y',
    lambda_param: Optional[float] = None
) -> Dict[pd.Timestamp, dict]:
    """
    Считает mean returns и covariance matrix на расширяющихся окнах.
    """
    step_days = _period_to_days(step_size)
    results: Dict[pd.Timestamp, dict] = {}
    dates = returns.index.sort_values()

    if len(dates) == 0:
        return results

    start_date = dates.min()
    current_end = start_date + pd.Timedelta(days=365)

    while current_end <= dates.max():
        window_returns = returns.loc[(returns.index >= start_date) & (returns.index <= current_end)]
        window_returns = _prepare_window_returns(window_returns)

        if len(window_returns) >= 2:
            if lambda_param is None:
                mean_returns = window_returns.mean().values
                cov_matrix = window_returns.cov().values
            else:
                weights = calculate_exponential_weights(len(window_returns), lambda_param)
                mean_returns = np.average(window_returns.values, axis=0, weights=weights)
                cov_matrix = calculate_weighted_covariance(window_returns, weights)

            results[current_end] = {
                'window_start': start_date,
                'window_end': current_end,
                'window_returns': window_returns,
                'window_size': len(window_returns),
                'mean_returns': mean_returns,
                'covariance_matrix': cov_matrix
            }

        current_end = current_end + pd.Timedelta(days=step_days)

    return results


def _build_trailing_window_result(
    returns: pd.DataFrame,
    window_method: str,
    window_size: str,
    lambda_param: Optional[float]
) -> dict:
    """
    Строит действительно последнее доступное окно по последней дате выборки.
    Это исправляет ситуацию, когда шаг 1Y оставлял неиспользованной часть конца выборки.
    """
    dates = returns.index.sort_values()

    if len(dates) == 0:
        raise ValueError("Пустой набор доходностей.")

    end_date = dates.max()

    if window_method == 'rolling':
        window_days = _period_to_days(window_size)
        start_date = end_date - pd.Timedelta(days=window_days)
        window_returns = returns.loc[(returns.index > start_date) & (returns.index <= end_date)]
    elif window_method == 'expanding':
        start_date = dates.min()
        window_returns = returns.loc[(returns.index >= start_date) & (returns.index <= end_date)]
    else:
        raise ValueError("window_method должен быть 'rolling' или 'expanding'")

    window_returns = _prepare_window_returns(window_returns)

    if len(window_returns) < 2:
        raise ValueError("Недостаточно наблюдений в последнем доступном окне.")

    if lambda_param is None:
        mean_returns = window_returns.mean().values
        cov_matrix = window_returns.cov().values
    else:
        weights = calculate_exponential_weights(len(window_returns), lambda_param)
        mean_returns = np.average(window_returns.values, axis=0, weights=weights)
        cov_matrix = calculate_weighted_covariance(window_returns, weights)

    return {
        'window_start': window_returns.index.min(),
        'window_end': window_returns.index.max(),
        'window_returns': window_returns,
        'window_size': len(window_returns),
        'mean_returns': mean_returns,
        'covariance_matrix': cov_matrix
    }


# ============================================================
# БЛОК 2. ВЫБОР ОКНА ДЛЯ ПУНКТА 4
# ============================================================

def select_estimation_window(
    returns: pd.DataFrame,
    window_method: str = 'rolling',
    window_size: str = '1Y',
    step_size: str = '1Y',
    lambda_param: Optional[float] = 0.94,
    selection_mode: str = 'latest'
) -> dict:
    """
    Выбирает одно окно и одну схему взвешивания для дальнейшей оптимизации.

    По умолчанию:
    - rolling,
    - 1Y,
    - 1Y step,
    - lambda = 0.94,
    - реально последнее доступное окно по последней дате выборки.
    """
    if window_method == 'rolling':
        analysis_results = rolling_window_analysis(
            returns=returns,
            window_size=window_size,
            step_size=step_size,
            lambda_param=lambda_param
        )
    elif window_method == 'expanding':
        analysis_results = expanding_window_analysis(
            returns=returns,
            step_size=step_size,
            lambda_param=lambda_param
        )
    else:
        raise ValueError("window_method должен быть 'rolling' или 'expanding'")

    if not analysis_results:
        raise ValueError("Не удалось получить ни одного окна для оптимизации.")

    if selection_mode == 'latest':
        selected_result = _build_trailing_window_result(
            returns=returns,
            window_method=window_method,
            window_size=window_size,
            lambda_param=lambda_param
        )
    elif selection_mode == 'last_from_grid':
        selected_end_date = sorted(analysis_results.keys())[-1]
        selected_result = analysis_results[selected_end_date]
    else:
        raise ValueError("selection_mode должен быть 'latest' или 'last_from_grid'")

    return {
        'window_method': window_method,
        'window_size': window_size,
        'step_size': step_size,
        'lambda_param': lambda_param,
        'selection_mode': selection_mode,
        'window_start': selected_result['window_start'],
        'window_end': selected_result['window_end'],
        'window_n_obs': selected_result['window_size'],
        'mean_returns_daily': selected_result['mean_returns'],
        'covariance_matrix_daily': selected_result['covariance_matrix']
    }


def annualize_inputs(
    mean_returns_daily: np.ndarray,
    covariance_matrix_daily: np.ndarray,
    trading_days: int = 252
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Переводит дневные оценки в годовой масштаб.
    """
    mean_returns_annual = mean_returns_daily * trading_days
    covariance_matrix_annual = covariance_matrix_daily * trading_days
    return mean_returns_annual, covariance_matrix_annual


def regularize_covariance(covariance_matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Добавляет небольшую регуляризацию к ковариационной матрице,
    если она численно неудобна для оптимизации.
    """
    cov = np.asarray(covariance_matrix, dtype=float)
    cov = 0.5 * (cov + cov.T)
    cov = cov + np.eye(cov.shape[0]) * eps
    return cov


# ============================================================
# БЛОК 3. ФУНКЦИИ ПОРТФЕЛЯ
# ============================================================

def portfolio_return(weights: np.ndarray, mean_returns: np.ndarray) -> float:
    return float(weights @ mean_returns)


def portfolio_variance(weights: np.ndarray, covariance_matrix: np.ndarray) -> float:
    return float(weights @ covariance_matrix @ weights)


def portfolio_volatility(weights: np.ndarray, covariance_matrix: np.ndarray) -> float:
    return float(np.sqrt(max(portfolio_variance(weights, covariance_matrix), 0.0)))


# ============================================================
# БЛОК 4. РЕШЕНИЕ ЗАДАЧ ОПТИМИЗАЦИИ
# ============================================================

def _build_initial_weights(n_assets: int, bounds: List[Tuple[Optional[float], Optional[float]]]) -> np.ndarray:
    """
    Строит стартовую точку, совместимую с bounds.
    """
    x0 = np.repeat(1.0 / n_assets, n_assets)

    lower_bounds = np.array([(-np.inf if b[0] is None else b[0]) for b in bounds], dtype=float)
    upper_bounds = np.array([(np.inf if b[1] is None else b[1]) for b in bounds], dtype=float)

    x0 = np.maximum(x0, np.where(np.isfinite(lower_bounds), lower_bounds, x0))
    x0 = np.minimum(x0, np.where(np.isfinite(upper_bounds), upper_bounds, x0))

    finite_lower_sum = np.nansum(np.where(np.isfinite(lower_bounds), lower_bounds, 0.0))
    if finite_lower_sum > 1 + 1e-10:
        raise ValueError("Сумма минимальных ограничений превышает 1.")

    total = x0.sum()
    if np.isfinite(total) and abs(total) > 1e-12:
        x0 = x0 / total

    return x0


def solve_gmv_portfolio(
    mean_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    bounds: List[Tuple[Optional[float], Optional[float]]]
) -> dict:
    """
    Находит глобальный портфель минимальной дисперсии для заданных ограничений.
    """
    covariance_matrix = regularize_covariance(covariance_matrix)
    n_assets = len(mean_returns)
    x0 = _build_initial_weights(n_assets, bounds)

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    result = minimize(
        fun=lambda w: portfolio_variance(w, covariance_matrix),
        x0=x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-12}
    )

    if not result.success:
        raise RuntimeError(f"Не удалось найти GMV-портфель: {result.message}")

    weights = result.x
    return {
        'weights': weights,
        'portfolio_return': portfolio_return(weights, mean_returns),
        'portfolio_volatility': portfolio_volatility(weights, covariance_matrix)
    }


def solve_max_return_portfolio(
    mean_returns: np.ndarray,
    bounds: List[Tuple[Optional[float], Optional[float]]]
) -> dict:
    """
    Находит максимально доходный допустимый портфель.
    """
    n_assets = len(mean_returns)

    result = linprog(
        c=-np.asarray(mean_returns, dtype=float),
        A_eq=np.ones((1, n_assets), dtype=float),
        b_eq=np.array([1.0], dtype=float),
        bounds=bounds,
        method='highs'
    )

    if result.status == 3:
        return {
            'success': False,
            'is_unbounded': True,
            'weights': None,
            'portfolio_return': np.inf,
            'message': result.message
        }

    if not result.success:
        return {
            'success': False,
            'is_unbounded': False,
            'weights': None,
            'portfolio_return': np.nan,
            'message': result.message
        }

    weights = result.x
    return {
        'success': True,
        'is_unbounded': False,
        'weights': weights,
        'portfolio_return': float(weights @ mean_returns),
        'message': result.message
    }


def solve_markowitz_for_target_return(
    mean_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    target_return: float,
    bounds: List[Tuple[Optional[float], Optional[float]]],
    x0: Optional[np.ndarray] = None
) -> dict:
    """
    Для заданной целевой доходности находит портфель минимального риска.
    """
    covariance_matrix = regularize_covariance(covariance_matrix)
    n_assets = len(mean_returns)

    if x0 is None:
        x0 = _build_initial_weights(n_assets, bounds)
    else:
        x0 = np.asarray(x0, dtype=float)

    objective = lambda w: float(w @ covariance_matrix @ w)
    objective_jac = lambda w: 2.0 * (covariance_matrix @ w)

    constraints = [
        {
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0,
            'jac': lambda w: np.ones(n_assets)
        },
        {
            'type': 'eq',
            'fun': lambda w: float(w @ mean_returns) - float(target_return),
            'jac': lambda w: mean_returns
        }
    ]

    result = minimize(
        fun=objective,
        jac=objective_jac,
        x0=x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 300, 'ftol': 1e-9, 'disp': False}
    )

    if not result.success:
        return {
            'success': False,
            'weights': None,
            'portfolio_return': np.nan,
            'portfolio_volatility': np.nan,
            'message': result.message
        }

    weights = result.x

    return {
        'success': True,
        'weights': weights,
        'portfolio_return': portfolio_return(weights, mean_returns),
        'portfolio_volatility': portfolio_volatility(weights, covariance_matrix),
        'message': result.message
    }


# ============================================================
# БЛОК 5. ПОСТРОЕНИЕ ЭФФЕКТИВНОЙ ГРАНИЦЫ
# ============================================================

def _choose_visual_max_target_for_unbounded_case(
    mean_returns: np.ndarray,
    gmv_return: float
) -> float:
    """
    Для неограниченной сверху эффективной границы выбирает конечный
    диапазон доходностей только для визуализации.

    Это нужно для пункта 5:
    при unlimited short true max-return portfolio не существует,
    поэтому строим корректный конечный фрагмент верхней ветви.
    """
    mu = np.asarray(mean_returns, dtype=float)
    spread = max(float(mu.max() - mu.min()), float(np.std(mu)), 1e-4)
    visual_max = max(float(mu.max()), float(gmv_return + 1.5 * spread))

    if visual_max <= gmv_return + 1e-10:
        visual_max = gmv_return + spread

    return visual_max


def build_efficient_frontier(
    mean_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    bounds: List[Tuple[Optional[float], Optional[float]]],
    n_points: int = 50
) -> pd.DataFrame:
    """
    Строит границу для одного сценария ограничений.

    - для bounded - строим верхнюю ветвь от GMV до max-return portfolio;
    - для unbounded case max-return portfolio не существует,
      поэтому строим конечный визуальный фрагмент верхней ветви от GMV до разумной
      конечной целевой доходности.
    """
    gmv = solve_gmv_portfolio(mean_returns, covariance_matrix, bounds)
    max_return_solution = solve_max_return_portfolio(mean_returns, bounds)

    min_target = gmv['portfolio_return']

    if max_return_solution.get('is_unbounded', False):
        max_target = _choose_visual_max_target_for_unbounded_case(
            mean_returns=mean_returns,
            gmv_return=min_target
        )
        frontier_mode = 'visual_fragment_of_unbounded_frontier'
    else:
        if not max_return_solution.get('success', False):
            raise RuntimeError(
                "Не удалось построить верхнюю границу: не найден max-return portfolio "
                f"({max_return_solution.get('message', 'unknown error')})."
            )

        max_target = float(max_return_solution['portfolio_return'])
        frontier_mode = 'bounded_upper_branch'

    if max_target < min_target:
        min_target, max_target = max_target, min_target

    target_returns = np.linspace(min_target, max_target, n_points)
    frontier_points = []
    warm_start = gmv['weights']

    for target in target_returns:
        solution = solve_markowitz_for_target_return(
            mean_returns=mean_returns,
            covariance_matrix=covariance_matrix,
            target_return=float(target),
            bounds=bounds,
            x0=warm_start
        )

        if solution['success']:
            frontier_points.append({
                'target_return': target,
                'portfolio_return': solution['portfolio_return'],
                'portfolio_volatility': solution['portfolio_volatility'],
                'weights': solution['weights'],
                'frontier_mode': frontier_mode
            })
            warm_start = solution['weights']

    frontier_df = pd.DataFrame(frontier_points)

    if frontier_df.empty:
        raise RuntimeError("Не удалось построить ни одной точки эффективной границы.")

    frontier_df = frontier_df.sort_values(['portfolio_return', 'portfolio_volatility']).reset_index(drop=True)
    return frontier_df


# ============================================================
# БЛОК 6. ОГРАНИЧЕНИЯ ДЛЯ ПУНКТОВ 5-8
# ============================================================

def get_bounds_for_points_5_8(n_assets: int) -> Dict[str, dict]:
    """
    Возвращает ограничения для пунктов 5-8.
    """
    return {
        'point_5_unrestricted_short': {
            'title': 'Пункт 5 - short allowed without limits',
            'bounds': [(None, None)] * n_assets
        },
        'point_6_short_limit_25': {
            'title': 'Пункт 6 - short allowed, but each weight >= -25%',
            'bounds': [(-0.25, None)] * n_assets
        },
        'point_7_no_short': {
            'title': 'Пункт 7 - no short selling',
            'bounds': [(0.0, 1.0)] * n_assets
        },
        'point_8_min_2_percent_each': {
            'title': 'Пункт 8 - each asset weight >= 2%',
            'bounds': [(0.02, 1.0)] * n_assets
        }
    }


# ============================================================
# БЛОК 7. КРАТКИЕ СВОДКИ ПО РЕЗУЛЬТАТАМ
# ============================================================

def extract_gmv_portfolio(frontier_df: pd.DataFrame, asset_names: List[str]) -> dict:
    """
    Находит GMV-портфель внутри уже построенной границы.
    """
    gmv_idx = frontier_df['portfolio_volatility'].idxmin()
    gmv_row = frontier_df.loc[gmv_idx]

    gmv_weights = pd.Series(gmv_row['weights'], index=asset_names).sort_values(ascending=False)

    return {
        'return': gmv_row['portfolio_return'],
        'volatility': gmv_row['portfolio_volatility'],
        'weights': gmv_weights
    }


def compare_constraint_scenarios(frontiers: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Собирает сравнительную таблицу по сценариям ограничений.
    """
    rows = []

    for scenario_title, frontier_df in frontiers.items():
        gmv_idx = frontier_df['portfolio_volatility'].idxmin()
        gmv_row = frontier_df.loc[gmv_idx]

        rows.append({
            'scenario': scenario_title,
            'frontier_mode': frontier_df['frontier_mode'].iloc[0] if 'frontier_mode' in frontier_df.columns else 'unknown',
            'n_frontier_points': len(frontier_df),
            'gmv_return': gmv_row['portfolio_return'],
            'gmv_volatility': gmv_row['portfolio_volatility'],
            'frontier_min_return': frontier_df['portfolio_return'].min(),
            'frontier_max_return': frontier_df['portfolio_return'].max(),
            'frontier_min_volatility': frontier_df['portfolio_volatility'].min(),
            'frontier_max_volatility': frontier_df['portfolio_volatility'].max()
        })

    comparison_df = pd.DataFrame(rows).sort_values('gmv_volatility').reset_index(drop=True)
    return comparison_df


# ============================================================
# БЛОК 8. ГЛАВНЫЙ ЗАПУСК ДЛЯ ПУНКТОВ 4-8
# ============================================================

def run_task_4_8(
    returns: pd.DataFrame,
    asset_names: List[str],
    window_method: str = 'rolling',
    window_size: str = '1Y',
    step_size: str = '1Y',
    lambda_param: Optional[float] = 0.94,
    n_points: int = 50,
    selection_mode: str = 'latest'
) -> dict:
    """
    Выполняет пункты 4-8:
    - выбирает одно окно и одну схему взвешивания,
    - annualize input parameters,
    - строит 4 efficient frontier,
    - выделяет GMV-портфели,
    - делает сравнительную таблицу.
    """
    selected_window = select_estimation_window(
        returns=returns,
        window_method=window_method,
        window_size=window_size,
        step_size=step_size,
        lambda_param=lambda_param,
        selection_mode=selection_mode
    )

    mean_returns_annual, covariance_matrix_annual = annualize_inputs(
        mean_returns_daily=selected_window['mean_returns_daily'],
        covariance_matrix_daily=selected_window['covariance_matrix_daily'],
        trading_days=252
    )

    n_assets = len(asset_names)
    bounds_config = get_bounds_for_points_5_8(n_assets)

    frontiers = {}
    summaries = {}

    for _, scenario_info in bounds_config.items():
        frontier_df = build_efficient_frontier(
            mean_returns=mean_returns_annual,
            covariance_matrix=covariance_matrix_annual,
            bounds=scenario_info['bounds'],
            n_points=n_points
        )
        frontiers[scenario_info['title']] = frontier_df
        summaries[scenario_info['title']] = extract_gmv_portfolio(frontier_df, asset_names)

    comparison = compare_constraint_scenarios(frontiers)

    return {
        'selected_window': selected_window,
        'mean_returns_annual': mean_returns_annual,
        'covariance_matrix_annual': covariance_matrix_annual,
        'frontiers': frontiers,
        'summaries': summaries,
        'comparison': comparison
    }

