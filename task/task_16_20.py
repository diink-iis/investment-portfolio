import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

from task_9_10 import (
    calculate_efficient_frontier,
    analyze_efficient_frontier_stability
)


# ============================================================
# БЛОК 1. ЗАГРУЗКА И ПОДГОТОВКА РЫНОЧНОГО ИНДЕКСА
# ============================================================


def load_market_index_data(file_path: str, index_name: str = 'IMOEX') -> pd.DataFrame:
    """
    Загружает исторические цены рыночного индекса из CSV.

    Ожидаемый формат:
    - разделитель ';'
    - десятичная запятая ','
    - столбец date в формате dd.mm.yyyy
    - столбец с названием индекса, например IMOEX
    """
    df = pd.read_csv(file_path, sep=';', decimal=',')
    df.columns = df.columns.str.strip()

    if 'date' not in df.columns:
        raise ValueError("В файле индекса не найден столбец 'date'.")

    if index_name not in df.columns:
        raise ValueError(
            f"В файле индекса не найден столбец {index_name}. "
            f"Доступные столбцы: {list(df.columns)}"
        )

    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
    df = df.set_index('date').sort_index()
    df[index_name] = pd.to_numeric(df[index_name], errors='coerce')
    df = df[[index_name]].dropna(how='all')

    return df



def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Считает логарифмические доходности.
    """
    returns = np.log(prices / prices.shift(1))
    returns = returns.dropna(how='all')
    return returns



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



def align_stock_and_market_returns(
    stock_returns: pd.DataFrame,
    market_returns: pd.DataFrame,
    market_column: str = 'IMOEX'
) -> pd.DataFrame:
    """
    Объединяет доходности акций и доходности рыночного индекса по общим датам.
    """
    if market_column not in market_returns.columns:
        raise ValueError(f"Столбец {market_column} не найден в market_returns")

    combined = stock_returns.join(market_returns[[market_column]], how='inner')
    combined = combined.dropna(how='any')

    if combined.empty:
        raise ValueError("После объединения акций и индекса не осталось общих наблюдений.")

    return combined



def prepare_selected_beta_window(
    stock_returns: pd.DataFrame,
    market_returns: pd.DataFrame,
    market_column: str = 'IMOEX',
    window_size: str = '1Y',
    selection_mode: str = 'latest'
) -> Dict[str, object]:
    """
    Готовит окно для пунктов 16-19.

    По умолчанию берется реально последнее доступное годовое окно,
    чтобы использовать весь доступный хвост выборки до последней даты.
    """
    combined = align_stock_and_market_returns(stock_returns, market_returns, market_column)
    dates = combined.index.sort_values()

    if selection_mode != 'latest':
        raise ValueError("В task_16_20 поддерживается selection_mode='latest'.")

    window_days = _period_to_days(window_size)
    window_end = dates.max()
    window_start = window_end - pd.Timedelta(days=window_days)

    window_data = combined.loc[(combined.index > window_start) & (combined.index <= window_end)]
    window_data = window_data.dropna(how='any')

    if len(window_data) < 2:
        raise ValueError("Недостаточно наблюдений в выбранном окне для расчета beta.")

    stock_cols = [col for col in window_data.columns if col != market_column]

    return {
        'window_start': window_data.index.min(),
        'window_end': window_data.index.max(),
        'window_size': len(window_data),
        'combined_returns': window_data,
        'stock_returns': window_data[stock_cols],
        'market_returns': window_data[market_column],
        'stock_tickers': stock_cols,
        'market_column': market_column,
    }


# ============================================================
# БЛОК 2. РЫНОЧНАЯ МОДЕЛЬ, HISTORICAL BETA И ADJUSTED BETA
# ============================================================


def calculate_market_model_betas(
    stock_returns: pd.DataFrame,
    market_returns: pd.Series
) -> pd.DataFrame:
    """
    Оценивает alpha и historical beta для каждой акции по рыночной модели.

    r_i = alpha_i + beta_i * r_m + eps_i
    """
    betas_data = []

    for ticker in stock_returns.columns:
        combined = pd.DataFrame({
            'stock': stock_returns[ticker],
            'market': market_returns
        }).dropna()

        if len(combined) < 2:
            alpha = np.nan
            beta = np.nan
        else:
            x = combined['market'].values
            y = combined['stock'].values
            cov_matrix = np.cov(x, y, ddof=1)
            beta = cov_matrix[0, 1] / cov_matrix[0, 0]
            alpha = np.mean(y) - beta * np.mean(x)

        betas_data.append({
            'ticker': ticker,
            'alpha': alpha,
            'beta': beta
        })

    result = pd.DataFrame(betas_data).set_index('ticker')
    return result



def calculate_adjusted_betas(
    raw_betas: pd.Series,
    weight_raw: float = 0.67,
    weight_market: float = 0.33,
    market_beta: float = 1.0
) -> pd.Series:
    """
    Считает adjusted beta по формуле Blume/Bloomberg-типа:

    beta_adjusted = weight_raw * beta_historical + weight_market * market_beta
    """
    adjusted = weight_raw * raw_betas + weight_market * market_beta
    adjusted.name = 'adjusted_beta'
    return adjusted



def calculate_residual_variances(
    stock_returns: pd.DataFrame,
    market_returns: pd.Series,
    alpha_beta_df: pd.DataFrame,
    beta_column: str = 'beta'
) -> pd.Series:
    """
    Считает дисперсии остатков рыночной модели для каждой акции.

    Для adjusted beta здесь сохраняется логика single-index model:
    residual variance оценивается по исторической регрессии,
    а затем комбинируется с прогнозной beta.
    """
    residuals_var = {}

    for ticker in stock_returns.columns:
        combined = pd.DataFrame({
            'stock': stock_returns[ticker],
            'market': market_returns
        }).dropna()

        if len(combined) < 2:
            residuals_var[ticker] = 0.0
            continue

        alpha = alpha_beta_df.loc[ticker, 'alpha']
        beta = alpha_beta_df.loc[ticker, beta_column]

        residuals = combined['stock'] - (alpha + beta * combined['market'])
        residuals_var[ticker] = residuals.var(ddof=1)

    result = pd.Series(residuals_var)
    result.name = 'residual_variance'
    return result



def calculate_covariance_from_betas(
    betas: pd.Series,
    market_variance: float,
    residual_variances: Optional[pd.Series] = None
) -> np.ndarray:
    """
    Считает ковариационную матрицу single-index model.

    Cov(r_i, r_j) = beta_i * beta_j * sigma_m^2,  i != j
    Var(r_i) = beta_i^2 * sigma_m^2 + sigma_eps_i^2
    """
    betas = betas.astype(float)
    beta_vector = betas.values.reshape(-1, 1)
    cov_matrix = (beta_vector @ beta_vector.T) * float(market_variance)

    if residual_variances is not None:
        residual_variances = residual_variances.reindex(betas.index).fillna(0.0)
        cov_matrix = cov_matrix + np.diag(residual_variances.values)

    cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
    return cov_matrix



def covariance_from_adjusted_betas(
    stock_returns: pd.DataFrame,
    market_returns: pd.Series,
    include_residuals: bool = True,
    weight_raw: float = 0.67,
    weight_market: float = 0.33,
    market_beta: float = 1.0
) -> Dict[str, object]:
    """
    Задача 16.
    Рассчитывает ковариационную матрицу на основе adjusted beta.
    """
    raw_beta_df = calculate_market_model_betas(stock_returns, market_returns)
    raw_betas = raw_beta_df['beta']
    adjusted_betas = calculate_adjusted_betas(
        raw_betas=raw_betas,
        weight_raw=weight_raw,
        weight_market=weight_market,
        market_beta=market_beta
    )

    market_variance = market_returns.var(ddof=1)

    residual_variances = None
    if include_residuals:
        residual_variances = calculate_residual_variances(
            stock_returns=stock_returns,
            market_returns=market_returns,
            alpha_beta_df=raw_beta_df,
            beta_column='beta'
        )

    cov_matrix_adjusted = calculate_covariance_from_betas(
        betas=adjusted_betas,
        market_variance=market_variance,
        residual_variances=residual_variances
    )

    try:
        np.linalg.cholesky(cov_matrix_adjusted)
        is_pd = True
    except np.linalg.LinAlgError:
        cov_matrix_adjusted = cov_matrix_adjusted + np.eye(cov_matrix_adjusted.shape[0]) * 1e-8
        is_pd = False

    return {
        'cov_matrix': cov_matrix_adjusted,
        'raw_betas': raw_betas,
        'adjusted_betas': adjusted_betas,
        'alphas': raw_beta_df['alpha'],
        'residual_variances': residual_variances,
        'market_variance': market_variance,
        'include_residuals': include_residuals,
        'is_positive_definite_before_regularization': is_pd,
    }


# ============================================================
# БЛОК 3. ЭФФЕКТИВНАЯ ГРАНИЦА ДЛЯ ADJUSTED BETA
# ============================================================


def efficient_frontier_from_adjusted_betas(
    cov_matrix: np.ndarray,
    mean_returns: np.ndarray,
    n_points: int = 100,
    method_name: str = 'Adjusted Betas'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Задача 17.
    Строит эффективную границу на основе covariance matrix из adjusted beta.
    """
    returns, stds = calculate_efficient_frontier(
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
        n_points=n_points
    )

    return returns, stds


# ============================================================
# БЛОК 4. ДИНАМИКА ADJUSTED BETA ВО ВРЕМЕНИ
# ============================================================


def _build_rolling_windows(
    combined_returns: pd.DataFrame,
    window_size: str = '1Y',
    step_size: str = '1Y'
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    Строит rolling windows для объединенного набора акций и индекса.
    """
    dates = combined_returns.index.sort_values()
    if len(dates) == 0:
        return {}

    window_days = _period_to_days(window_size)
    step_days = _period_to_days(step_size)

    current_end = dates.min() + pd.Timedelta(days=window_days)
    windows = {}

    while current_end <= dates.max():
        current_start = current_end - pd.Timedelta(days=window_days)
        window_data = combined_returns.loc[
            (combined_returns.index > current_start) & (combined_returns.index <= current_end)
        ].dropna(how='any')

        if len(window_data) >= 2:
            windows[current_end] = window_data

        current_end = current_end + pd.Timedelta(days=step_days)

    return windows



def efficient_frontier_dynamics_adjusted_betas(
    stock_returns: pd.DataFrame,
    market_returns: pd.DataFrame,
    market_column: str = 'IMOEX',
    window_size: str = '1Y',
    step_size: str = '1Y',
    include_residuals: bool = True,
    n_points: int = 100,
    weight_raw: float = 0.67,
    weight_market: float = 0.33,
    market_beta: float = 1.0
) -> Tuple[Dict[pd.Timestamp, dict], pd.DataFrame]:
    """
    Задача 18.
    Строит динамику эффективных границ на основе adjusted beta по rolling windows.
    """
    combined = align_stock_and_market_returns(stock_returns, market_returns, market_column)
    windows = _build_rolling_windows(combined, window_size=window_size, step_size=step_size)

    frontiers = {}

    for date, window_data in windows.items():
        stock_window = window_data.drop(columns=[market_column])
        market_window = window_data[market_column]

        adjusted_result = covariance_from_adjusted_betas(
            stock_returns=stock_window,
            market_returns=market_window,
            include_residuals=include_residuals,
            weight_raw=weight_raw,
            weight_market=weight_market,
            market_beta=market_beta
        )

        mean_returns = stock_window.mean().values
        ef_returns, ef_stds = efficient_frontier_from_adjusted_betas(
            cov_matrix=adjusted_result['cov_matrix'],
            mean_returns=mean_returns,
            n_points=n_points
        )

        frontiers[date] = {
            'returns': ef_returns,
            'stds': ef_stds,
            'min_std': ef_stds[0],
            'min_std_return': ef_returns[0],
            'max_return': ef_returns[-1],
            'max_return_std': ef_stds[-1]
        }

    if frontiers:
        stability = analyze_efficient_frontier_stability(frontiers)
    else:
        stability = pd.DataFrame()

    return frontiers, stability


# ============================================================
# БЛОК 5. СРАВНЕНИЕ ТРЕХ МЕТОДОВ НА ОДНОМ ОКНЕ
# ============================================================


def _frontier_summary(method_name: str, returns: np.ndarray, stds: np.ndarray) -> Dict[str, float]:
    """
    Краткие метрики для одной эффективной границы.
    """
    sharpe_ratios = returns / stds
    max_sharpe_idx = int(np.argmax(sharpe_ratios))

    return {
        'method': method_name,
        'gmv_return': float(returns[0]),
        'gmv_std': float(stds[0]),
        'max_return': float(returns[-1]),
        'max_return_std': float(stds[-1]),
        'max_sharpe': float(sharpe_ratios[max_sharpe_idx]),
        'max_sharpe_return': float(returns[max_sharpe_idx]),
        'max_sharpe_std': float(stds[max_sharpe_idx]),
        'frontier_range': float(returns[-1] - returns[0]),
        'efficiency_ratio': float(returns[-1] / stds[0]),
    }



def compare_three_methods_on_selected_window(
    stock_returns: pd.DataFrame,
    market_returns: pd.Series,
    include_residuals: bool = True,
    n_points: int = 100,
    weight_raw: float = 0.67,
    weight_market: float = 0.33,
    market_beta: float = 1.0
) -> Dict[str, object]:
    """
    Задача 19.
    Сравнивает три подхода на одном и том же окне:
    - исторические доходности,
    - historical beta,
    - adjusted beta.
    """
    mean_returns = stock_returns.mean().values
    cov_hist = stock_returns.cov().values

    raw_beta_df = calculate_market_model_betas(stock_returns, market_returns)
    raw_betas = raw_beta_df['beta']
    adjusted_betas = calculate_adjusted_betas(
        raw_betas,
        weight_raw=weight_raw,
        weight_market=weight_market,
        market_beta=market_beta
    )

    residual_variances = None
    if include_residuals:
        residual_variances = calculate_residual_variances(
            stock_returns=stock_returns,
            market_returns=market_returns,
            alpha_beta_df=raw_beta_df,
            beta_column='beta'
        )

    market_variance = market_returns.var(ddof=1)
    cov_hist_beta = calculate_covariance_from_betas(raw_betas, market_variance, residual_variances)
    cov_adj_beta = calculate_covariance_from_betas(adjusted_betas, market_variance, residual_variances)

    hist_returns, hist_stds = calculate_efficient_frontier(mean_returns, cov_hist, n_points=n_points)
    raw_returns, raw_stds = calculate_efficient_frontier(mean_returns, cov_hist_beta, n_points=n_points)
    adj_returns, adj_stds = calculate_efficient_frontier(mean_returns, cov_adj_beta, n_points=n_points)

    summaries = pd.DataFrame([
        _frontier_summary('Historical returns', hist_returns, hist_stds),
        _frontier_summary('Historical betas', raw_returns, raw_stds),
        _frontier_summary('Adjusted betas', adj_returns, adj_stds),
    ]).set_index('method')

    return {
        'frontiers': {
            'Historical returns': {'returns': hist_returns, 'stds': hist_stds},
            'Historical betas': {'returns': raw_returns, 'stds': raw_stds},
            'Adjusted betas': {'returns': adj_returns, 'stds': adj_stds},
        },
        'summary_table': summaries,
        'raw_betas': raw_betas,
        'adjusted_betas': adjusted_betas,
        'residual_variances': residual_variances,
        'cov_matrices': {
            'Historical returns': cov_hist,
            'Historical betas': cov_hist_beta,
            'Adjusted betas': cov_adj_beta,
        },
        'market_variance': market_variance,
    }


# ============================================================
# БЛОК 6. СРАВНЕНИЕ ТРЕХ МЕТОДОВ ВО ВРЕМЕНИ
# ============================================================


def compare_three_methods_over_time(
    stock_returns: pd.DataFrame,
    market_returns: pd.DataFrame,
    market_column: str = 'IMOEX',
    window_size: str = '1Y',
    step_size: str = '1Y',
    include_residuals: bool = True,
    n_points: int = 100,
    weight_raw: float = 0.67,
    weight_market: float = 0.33,
    market_beta: float = 1.0
) -> Dict[str, object]:
    """
    Задача 20.
    Сравнивает три метода на наборе rolling windows.
    """
    combined = align_stock_and_market_returns(stock_returns, market_returns, market_column)
    windows = _build_rolling_windows(combined, window_size=window_size, step_size=step_size)

    frontiers_by_method = {
        'Historical returns': {},
        'Historical betas': {},
        'Adjusted betas': {}
    }

    metrics_rows = []

    for date, window_data in windows.items():
        stock_window = window_data.drop(columns=[market_column])
        market_window = window_data[market_column]

        comparison = compare_three_methods_on_selected_window(
            stock_returns=stock_window,
            market_returns=market_window,
            include_residuals=include_residuals,
            n_points=n_points,
            weight_raw=weight_raw,
            weight_market=weight_market,
            market_beta=market_beta
        )

        for method_name, frontier in comparison['frontiers'].items():
            frontiers_by_method[method_name][date] = {
                'returns': frontier['returns'],
                'stds': frontier['stds'],
                'min_std': frontier['stds'][0],
                'min_std_return': frontier['returns'][0],
                'max_return': frontier['returns'][-1],
                'max_return_std': frontier['stds'][-1]
            }

        summary_table = comparison['summary_table'].reset_index()
        summary_table['date'] = date
        metrics_rows.append(summary_table)

    if metrics_rows:
        metrics_table = pd.concat(metrics_rows, ignore_index=True)
        metrics_table = metrics_table[['date', 'method', 'gmv_return', 'gmv_std', 'max_return',
                                       'max_return_std', 'max_sharpe', 'max_sharpe_return',
                                       'max_sharpe_std', 'frontier_range', 'efficiency_ratio']]
        metrics_table = metrics_table.sort_values(['date', 'method']).reset_index(drop=True)
    else:
        metrics_table = pd.DataFrame()

    stability_tables = {}
    for method_name, frontiers in frontiers_by_method.items():
        if frontiers:
            stability_tables[method_name] = analyze_efficient_frontier_stability(frontiers)
        else:
            stability_tables[method_name] = pd.DataFrame()

    if not metrics_table.empty:
        summary_by_method = metrics_table.groupby('method').agg(
            avg_gmv_std=('gmv_std', 'mean'),
            std_gmv_std=('gmv_std', 'std'),
            avg_max_sharpe=('max_sharpe', 'mean'),
            std_max_sharpe=('max_sharpe', 'std'),
            avg_efficiency_ratio=('efficiency_ratio', 'mean'),
            avg_frontier_range=('frontier_range', 'mean')
        ).sort_values('avg_gmv_std')
    else:
        summary_by_method = pd.DataFrame()

    return {
        'frontiers_by_method': frontiers_by_method,
        'metrics_table': metrics_table,
        'stability_tables': stability_tables,
        'summary_by_method': summary_by_method,
    }
