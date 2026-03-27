
"""
Задачи 16-20: Ковариационные матрицы на основе скорректированных beta
и сравнение трех подходов к оценке входных данных для optimizer.

Пункты 16-17:
- рассчитываем covariance matrix на основе adjusted beta;
- строим efficient frontier на выбранном окне.

Пункт 18:
- показываем, как frontier на adjusted beta меняется по разным историческим окнам.

Пункт 19:
- сравниваем frontier, построенные тремя способами:
  1) на основе исторических доходностей;
  2) на основе historical beta;
  3) на основе adjusted beta.

Пункт 20:
- повторяем сравнение по разным окнам и показываем динамику.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from task_4_8 import (
    build_efficient_frontier,
    extract_gmv_portfolio
)


# ============================================================
# БЛОК 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ОКОН И РЫНКА
# ============================================================

def _period_to_days(period: str) -> int:
    """
    Переводит строку периода в число календарных дней.
    """
    mapping = {
        '1Y': 365,
        '1Q': 91,
        '1M': 30,
        '1W': 7,
        '1D': 1,
    }
    if period not in mapping:
        raise ValueError(f'Неизвестный период: {period}')
    return mapping[period]


def _prepare_window_returns(window_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Очищает окно только внутри выбранного периода.
    """
    if window_returns.empty:
        return window_returns
    return window_returns.dropna(how='any')


def prepare_market_returns(
    returns: pd.DataFrame,
    market_returns: Optional[pd.Series] = None,
    market_label: Optional[str] = None
) -> Tuple[pd.Series, str, str]:
    """
    Готовит рыночный ряд.

    Если отдельный ряд индекса не передан, используется proxy:
    равновзвешенная средняя доходность всех бумаг в выборке.
    """
    if market_returns is None:
        proxy_market = returns.mean(axis=1)
        resolved_label = market_label or 'Proxy market - equal-weighted average return of 30 stocks'
        market_source = 'proxy_market'
        return proxy_market, resolved_label, market_source

    market_series = pd.Series(market_returns).copy()
    market_series = market_series.sort_index()
    market_series = market_series.dropna()

    resolved_label = market_label or 'User-provided market series'
    market_source = 'external_market_series'
    return market_series, resolved_label, market_source


def get_selected_window_returns(
    returns: pd.DataFrame,
    window_method: str = 'rolling',
    window_size: str = '1Y',
    step_size: str = '1Y',
    selection_mode: str = 'latest'
) -> dict:
    """
    Возвращает одно выбранное окно как DataFrame доходностей.
    """
    returns = returns.sort_index()

    if returns.empty:
        raise ValueError('Пустой набор доходностей.')

    if selection_mode == 'latest':
        end_date = returns.index.max()

        if window_method == 'rolling':
            window_days = _period_to_days(window_size)
            start_date = end_date - pd.Timedelta(days=window_days)
            window_returns = returns.loc[(returns.index > start_date) & (returns.index <= end_date)]
        elif window_method == 'expanding':
            start_date = returns.index.min()
            window_returns = returns.loc[(returns.index >= start_date) & (returns.index <= end_date)]
        else:
            raise ValueError("window_method должен быть 'rolling' или 'expanding'")

        window_returns = _prepare_window_returns(window_returns)

        if len(window_returns) < 2:
            raise ValueError('Недостаточно наблюдений в выбранном окне.')

        return {
            'window_start': window_returns.index.min(),
            'window_end': window_returns.index.max(),
            'window_returns': window_returns,
            'window_size': len(window_returns)
        }

    if selection_mode != 'last_from_grid':
        raise ValueError("selection_mode должен быть 'latest' или 'last_from_grid'")

    windows = get_dynamic_windows(
        returns=returns,
        window_method=window_method,
        window_size=window_size,
        step_size=step_size
    )

    if not windows:
        raise ValueError('Не удалось построить сетку окон.')

    selected_end = sorted(windows.keys())[-1]
    selected_window = windows[selected_end]

    return {
        'window_start': selected_window['window_start'],
        'window_end': selected_window['window_end'],
        'window_returns': selected_window['window_returns'],
        'window_size': len(selected_window['window_returns'])
    }


def get_dynamic_windows(
    returns: pd.DataFrame,
    window_method: str = 'rolling',
    window_size: str = '1Y',
    step_size: str = '1Y'
) -> Dict[pd.Timestamp, dict]:
    """
    Строит набор окон для динамического анализа.
    """
    returns = returns.sort_index()
    dates = returns.index

    if len(dates) == 0:
        return {}

    step_days = _period_to_days(step_size)
    results: Dict[pd.Timestamp, dict] = {}

    if window_method == 'rolling':
        window_days = _period_to_days(window_size)
        current_end = dates.min() + pd.Timedelta(days=window_days)

        while current_end <= dates.max():
            current_start = current_end - pd.Timedelta(days=window_days)
            window_returns = returns.loc[(returns.index > current_start) & (returns.index <= current_end)]
            window_returns = _prepare_window_returns(window_returns)

            if len(window_returns) >= 2:
                results[current_end] = {
                    'window_start': window_returns.index.min(),
                    'window_end': window_returns.index.max(),
                    'window_returns': window_returns
                }

            current_end = current_end + pd.Timedelta(days=step_days)

    elif window_method == 'expanding':
        start_date = dates.min()
        current_end = start_date + pd.Timedelta(days=365)

        while current_end <= dates.max():
            window_returns = returns.loc[(returns.index >= start_date) & (returns.index <= current_end)]
            window_returns = _prepare_window_returns(window_returns)

            if len(window_returns) >= 2:
                results[current_end] = {
                    'window_start': window_returns.index.min(),
                    'window_end': window_returns.index.max(),
                    'window_returns': window_returns
                }

            current_end = current_end + pd.Timedelta(days=step_days)

    else:
        raise ValueError("window_method должен быть 'rolling' или 'expanding'")

    return results


def _align_assets_and_market(
    window_returns: pd.DataFrame,
    market_returns: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Выравнивает доходности акций и рынка по общим датам.
    """
    aligned = window_returns.copy()
    aligned['__market__'] = market_returns.reindex(aligned.index)
    aligned = aligned.dropna(how='any')

    if len(aligned) < 2:
        raise ValueError('После выравнивания с рыночным рядом осталось слишком мало наблюдений.')

    market_aligned = aligned.pop('__market__')
    return aligned, market_aligned


# ============================================================
# БЛОК 2. ОЦЕНКА MARKET MODEL И BETA
# ============================================================

def calculate_exponential_weights(n: int, lambda_param: float = 0.94) -> np.ndarray:
    """
    Считает экспоненциальные веса.
    """
    weights = np.array([lambda_param ** (n - i - 1) for i in range(n)], dtype=float)
    weights = weights / weights.sum()
    return weights


def calculate_weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * values))


def calculate_weighted_cov(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    x_mean = calculate_weighted_mean(x, weights)
    y_mean = calculate_weighted_mean(y, weights)
    return float(np.sum(weights * (x - x_mean) * (y - y_mean)))


def calculate_weighted_var(x: np.ndarray, weights: np.ndarray) -> float:
    x_mean = calculate_weighted_mean(x, weights)
    return float(np.sum(weights * (x - x_mean) ** 2))


def estimate_market_model_betas(
    window_returns: pd.DataFrame,
    market_returns: pd.Series,
    lambda_param: Optional[float] = None,
    beta_adjustment_weight: float = 2 / 3
) -> dict:
    """
    Оценивает historical beta и adjusted beta по market model.

    Для adjusted beta используется классическая Blume-style correction:
    beta_adj = w * beta_hist + (1 - w) * 1
    где по умолчанию w = 2/3.
    """
    asset_returns, market_aligned = _align_assets_and_market(window_returns, market_returns)

    market_values = market_aligned.values.astype(float)
    asset_matrix = asset_returns.values.astype(float)

    n_obs = len(asset_returns)

    if lambda_param is None:
        weights = np.full(n_obs, 1.0 / n_obs, dtype=float)
    else:
        weights = calculate_exponential_weights(n_obs, lambda_param=lambda_param)

    mean_market = calculate_weighted_mean(market_values, weights)
    var_market = calculate_weighted_var(market_values, weights)

    if var_market <= 1e-15:
        raise ValueError('Дисперсия рыночного ряда слишком мала для оценки beta.')

    asset_rows = []

    for i, asset_name in enumerate(asset_returns.columns):
        asset_values = asset_matrix[:, i]

        mean_asset = calculate_weighted_mean(asset_values, weights)
        cov_im = calculate_weighted_cov(asset_values, market_values, weights)

        beta_hist = cov_im / var_market
        alpha = mean_asset - beta_hist * mean_market

        residuals = asset_values - alpha - beta_hist * market_values
        residual_var = float(np.sum(weights * residuals ** 2))

        beta_adj = beta_adjustment_weight * beta_hist + (1 - beta_adjustment_weight) * 1.0

        asset_rows.append({
            'asset': asset_name,
            'mean_return_daily': mean_asset,
            'alpha_daily': alpha,
            'historical_beta': beta_hist,
            'adjusted_beta': beta_adj,
            'residual_variance_daily': residual_var
        })

    beta_table = pd.DataFrame(asset_rows).set_index('asset')

    return {
        'beta_table': beta_table,
        'market_mean_daily': mean_market,
        'market_variance_daily': var_market,
        'weights': weights,
        'window_returns_aligned': asset_returns,
        'market_returns_aligned': market_aligned
    }


def build_single_index_covariance_matrix(
    beta_values: np.ndarray,
    market_variance: float,
    residual_variances: np.ndarray
) -> np.ndarray:
    """
    Строит covariance matrix в одноиндексной модели:
    Sigma = beta beta' * sigma_m^2 + D
    """
    systematic_part = np.outer(beta_values, beta_values) * market_variance
    idiosyncratic_part = np.diag(residual_variances)
    covariance_matrix = systematic_part + idiosyncratic_part
    covariance_matrix = 0.5 * (covariance_matrix + covariance_matrix.T)
    return covariance_matrix


def calculate_historical_inputs(
    window_returns: pd.DataFrame,
    lambda_param: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Считает historical mean returns и historical covariance matrix.
    """
    values = window_returns.values.astype(float)
    n_obs = len(window_returns)

    if lambda_param is None:
        mean_returns = values.mean(axis=0)
        cov_matrix = window_returns.cov().values
        return mean_returns, cov_matrix

    weights = calculate_exponential_weights(n_obs, lambda_param=lambda_param)
    mean_returns = np.average(values, axis=0, weights=weights)

    centered = values - mean_returns
    cov_matrix = (centered * weights[:, None]).T @ centered
    return mean_returns, cov_matrix


def annualize_inputs(
    mean_returns_daily: np.ndarray,
    covariance_matrix_daily: np.ndarray,
    trading_days: int = 252
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Переводит дневные оценки в годовой масштаб.
    """
    mean_annual = mean_returns_daily * trading_days
    cov_annual = covariance_matrix_daily * trading_days
    return mean_annual, cov_annual


# ============================================================
# БЛОК 3. FRONTIER ДЛЯ ВЫБРАННОГО ОКНА
# ============================================================

def build_method_inputs_for_window(
    window_returns: pd.DataFrame,
    market_returns: pd.Series,
    lambda_param: Optional[float] = None
) -> dict:
    """
    Готовит все входные данные для трех методов:
    - historical returns covariance;
    - historical beta covariance;
    - adjusted beta covariance.
    """
    aligned_returns, aligned_market = _align_assets_and_market(window_returns, market_returns)

    mean_returns_daily, cov_hist_daily = calculate_historical_inputs(
        window_returns=aligned_returns,
        lambda_param=lambda_param
    )

    beta_results = estimate_market_model_betas(
        window_returns=aligned_returns,
        market_returns=aligned_market,
        lambda_param=lambda_param
    )

    beta_table = beta_results['beta_table']
    market_variance_daily = beta_results['market_variance_daily']

    cov_hist_beta_daily = build_single_index_covariance_matrix(
        beta_values=beta_table['historical_beta'].values,
        market_variance=market_variance_daily,
        residual_variances=beta_table['residual_variance_daily'].values
    )

    cov_adj_beta_daily = build_single_index_covariance_matrix(
        beta_values=beta_table['adjusted_beta'].values,
        market_variance=market_variance_daily,
        residual_variances=beta_table['residual_variance_daily'].values
    )

    mean_returns_annual, cov_hist_annual = annualize_inputs(mean_returns_daily, cov_hist_daily)
    _, cov_hist_beta_annual = annualize_inputs(mean_returns_daily, cov_hist_beta_daily)
    _, cov_adj_beta_annual = annualize_inputs(mean_returns_daily, cov_adj_beta_daily)

    method_inputs = {
        'Historical returns': {
            'mean_returns_annual': mean_returns_annual,
            'covariance_matrix_annual': cov_hist_annual
        },
        'Historical betas': {
            'mean_returns_annual': mean_returns_annual,
            'covariance_matrix_annual': cov_hist_beta_annual
        },
        'Adjusted betas': {
            'mean_returns_annual': mean_returns_annual,
            'covariance_matrix_annual': cov_adj_beta_annual
        }
    }

    return {
        'aligned_window_returns': aligned_returns,
        'aligned_market_returns': aligned_market,
        'beta_table': beta_table,
        'method_inputs': method_inputs,
        'cov_hist_daily': cov_hist_daily,
        'cov_hist_beta_daily': cov_hist_beta_daily,
        'cov_adj_beta_daily': cov_adj_beta_daily,
        'market_variance_daily': market_variance_daily
    }


def compare_frontier_methods(frontiers: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Делает компактную таблицу сравнения методов.
    """
    rows = []

    for method_name, frontier_df in frontiers.items():
        gmv_idx = frontier_df['portfolio_volatility'].idxmin()
        gmv_row = frontier_df.loc[gmv_idx]

        rows.append({
            'method': method_name,
            'n_points': len(frontier_df),
            'gmv_return': gmv_row['portfolio_return'],
            'gmv_volatility': gmv_row['portfolio_volatility'],
            'frontier_min_return': frontier_df['portfolio_return'].min(),
            'frontier_max_return': frontier_df['portfolio_return'].max(),
            'frontier_min_volatility': frontier_df['portfolio_volatility'].min(),
            'frontier_max_volatility': frontier_df['portfolio_volatility'].max(),
            'frontier_mode': frontier_df['frontier_mode'].iloc[0]
        })

    return pd.DataFrame(rows).sort_values('gmv_volatility').reset_index(drop=True)


def build_frontiers_for_window(
    window_returns: pd.DataFrame,
    market_returns: pd.Series,
    asset_names: List[str],
    lambda_param: Optional[float] = None,
    n_points: int = 12,
    bounds: Optional[List[Tuple[Optional[float], Optional[float]]]] = None
) -> dict:
    """
    Строит frontier для трех методов на одном и том же окне.
    """
    prepared = build_method_inputs_for_window(
        window_returns=window_returns,
        market_returns=market_returns,
        lambda_param=lambda_param
    )

    if bounds is None:
        bounds = [(None, None)] * len(asset_names)

    frontiers = {}
    summaries = {}

    for method_name, method_info in prepared['method_inputs'].items():
        frontier_df = build_efficient_frontier(
            mean_returns=method_info['mean_returns_annual'],
            covariance_matrix=method_info['covariance_matrix_annual'],
            bounds=bounds,
            n_points=n_points
        )
        frontiers[method_name] = frontier_df
        summaries[method_name] = extract_gmv_portfolio(frontier_df, asset_names)

    comparison = compare_frontier_methods(frontiers)

    return {
        'beta_table': prepared['beta_table'],
        'method_inputs': prepared['method_inputs'],
        'frontiers': frontiers,
        'summaries': summaries,
        'comparison': comparison,
        'cov_matrices_daily': {
            'Historical returns': prepared['cov_hist_daily'],
            'Historical betas': prepared['cov_hist_beta_daily'],
            'Adjusted betas': prepared['cov_adj_beta_daily']
        },
        'aligned_window_returns': prepared['aligned_window_returns'],
        'aligned_market_returns': prepared['aligned_market_returns']
    }


# ============================================================
# БЛОК 4. ДИНАМИКА ПО ОКНАМ
# ============================================================

def build_dynamic_method_comparison(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    asset_names: List[str],
    window_method: str = 'rolling',
    window_size: str = '1Y',
    step_size: str = '1Y',
    lambda_param: Optional[float] = None,
    n_points: int = 12,
    bounds: Optional[List[Tuple[Optional[float], Optional[float]]]] = None
) -> dict:
    """
    Строит frontier по разным окнам и собирает динамику.
    """
    if bounds is None:
        bounds = [(None, None)] * len(asset_names)

    windows = get_dynamic_windows(
        returns=returns,
        window_method=window_method,
        window_size=window_size,
        step_size=step_size
    )

    dynamic_frontiers_by_method = {
        'Historical returns': {},
        'Historical betas': {},
        'Adjusted betas': {}
    }

    dynamic_rows = []

    for window_end, window_info in windows.items():
        window_returns = window_info['window_returns']

        try:
            window_result = build_frontiers_for_window(
                window_returns=window_returns,
                market_returns=market_returns,
                asset_names=asset_names,
                lambda_param=lambda_param,
                n_points=n_points,
                bounds=bounds
            )
        except Exception:
            continue

        for method_name, frontier_df in window_result['frontiers'].items():
            dynamic_frontiers_by_method[method_name][window_end] = frontier_df

            gmv_idx = frontier_df['portfolio_volatility'].idxmin()
            gmv_row = frontier_df.loc[gmv_idx]

            dynamic_rows.append({
                'window_end': window_end,
                'window_start': window_info['window_start'],
                'method': method_name,
                'gmv_return': gmv_row['portfolio_return'],
                'gmv_volatility': gmv_row['portfolio_volatility'],
                'frontier_max_return': frontier_df['portfolio_return'].max(),
                'frontier_max_volatility': frontier_df['portfolio_volatility'].max()
            })

    dynamic_summary = pd.DataFrame(dynamic_rows)
    if not dynamic_summary.empty:
        dynamic_summary = dynamic_summary.sort_values(['window_end', 'method']).reset_index(drop=True)

    return {
        'windows': windows,
        'dynamic_frontiers_by_method': dynamic_frontiers_by_method,
        'dynamic_summary': dynamic_summary
    }


# ============================================================
# БЛОК 5. ГЛАВНАЯ ФУНКЦИЯ ДЛЯ ПУНКТОВ 16-20
# ============================================================

def run_task_16_20(
    returns: pd.DataFrame,
    asset_names: List[str],
    market_returns: Optional[pd.Series] = None,
    market_label: Optional[str] = None,
    window_method: str = 'rolling',
    window_size: str = '1Y',
    step_size: str = '1Y',
    lambda_param: Optional[float] = 0.94,
    n_points: int = 12,
    selection_mode: str = 'latest',
    dynamic_window_method: str = 'rolling',
    dynamic_window_size: str = '1Y',
    dynamic_step_size: str = '1Y'
) -> dict:
    """
    Выполняет пункты 16-20 в одном пайплайне.
    """
    market_series, resolved_market_label, market_source = prepare_market_returns(
        returns=returns,
        market_returns=market_returns,
        market_label=market_label
    )

    selected_window = get_selected_window_returns(
        returns=returns,
        window_method=window_method,
        window_size=window_size,
        step_size=step_size,
        selection_mode=selection_mode
    )

    selected_result = build_frontiers_for_window(
        window_returns=selected_window['window_returns'],
        market_returns=market_series,
        asset_names=asset_names,
        lambda_param=lambda_param,
        n_points=n_points,
        bounds=[(None, None)] * len(asset_names)
    )

    dynamic_result = build_dynamic_method_comparison(
        returns=returns,
        market_returns=market_series,
        asset_names=asset_names,
        window_method=dynamic_window_method,
        window_size=dynamic_window_size,
        step_size=dynamic_step_size,
        lambda_param=lambda_param,
        n_points=n_points,
        bounds=[(None, None)] * len(asset_names)
    )

    return {
        'config': {
            'window_method': window_method,
            'window_size': window_size,
            'step_size': step_size,
            'lambda_param': lambda_param,
            'selection_mode': selection_mode,
            'dynamic_window_method': dynamic_window_method,
            'dynamic_window_size': dynamic_window_size,
            'dynamic_step_size': dynamic_step_size
        },
        'market_label': resolved_market_label,
        'market_source': market_source,
        'selected_window': {
            'window_start': selected_window['window_start'],
            'window_end': selected_window['window_end'],
            'window_n_obs': selected_window['window_size']
        },
        'beta_table_selected': selected_result['beta_table'],
        'frontiers_selected': selected_result['frontiers'],
        'summaries_selected': selected_result['summaries'],
        'comparison_selected': selected_result['comparison'],
        'cov_matrices_selected_daily': selected_result['cov_matrices_daily'],
        'dynamic_frontiers_by_method': dynamic_result['dynamic_frontiers_by_method'],
        'dynamic_summary': dynamic_result['dynamic_summary']
    }
