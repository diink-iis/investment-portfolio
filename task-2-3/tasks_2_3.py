"""
Задачи 2 и 3: Анализ инвестиционного портфеля
Задача 2: Расчет показателей эффективности портфелей и построение эффективной границы
Задача 3: Анализ динамики портфелей во времени
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Функции должны быть доступны в среде (определены в финальном ноутбуке):
# - load_data(filepath)
# - calculate_returns(prices_df)
# - estimate_parameters(returns_df)
# - portfolio_performance(weights, expected_returns, cov_matrix)
# - optimize_portfolio(expected_returns, cov_matrix, risk_free_rate, method, target_return)
# - annualize_metrics(daily_return, daily_volatility)
# - plot_efficient_frontier(...)
# - plot_portfolio_weights(...)


def task_2(returns_df, expected_returns, cov_matrix, risk_free_rate):
    """Выполнение задачи 2"""

    print("\n" + "="*80)
    print("ЗАДАЧА 2: Расчет показателей эффективности портфелей")
    print("="*80)

    n_assets = len(expected_returns)

    # Отладка ковариационной матрицы
    print(f"\nКовариационная матрица: размер {cov_matrix.shape}")
    print(f"Диагональные элементы (дисперсии) - первые 10:")
    diag = np.diag(cov_matrix)
    print(diag[:10])
    print(f"\nСредняя дисперсия: {np.trace(cov_matrix) / n_assets:.8f}")
    print(f"Стандартные отклонения (годовые, %) - первые 10:")
    std_devs = np.sqrt(diag) * np.sqrt(252) * 100
    print(np.round(std_devs[:10], 2))

    # Проверим корреляции
    corr_matrix = returns_df.corr()
    avg_correlation = (corr_matrix.values.sum() - np.trace(corr_matrix.values)) / (n_assets * (n_assets - 1))
    print(f"\nСредняя корреляция между активами: {avg_correlation:.4f}")
    print(f"Корреляционная матрица (срез 10x10):")
    print(np.round(corr_matrix.iloc[:10, :10].values, 3))

    # Оптимизация портфелей
    print("\n1. Оптимизация портфелей...")

    # Портфель минимальной дисперсии
    min_var_weights = optimize_portfolio(expected_returns, cov_matrix, risk_free_rate,
                                         method='min_variance')
    print(f"Минимальная дисперсия - сумма весов: {min_var_weights.sum():.6f}")
    print(f"Минимальная дисперсия - ненулевых весов: {(min_var_weights > 0.01).sum()}")
    print(f"Минимальная дисперсия - первые 5 весов: {min_var_weights[:5]}")

    # Портфель максимального коэффициента Шарпа
    max_sharpe_weights = optimize_portfolio(expected_returns, cov_matrix, risk_free_rate,
                                           method='max_sharpe')
    print(f"Максимальный Шарп - сумма весов: {max_sharpe_weights.sum():.6f}")
    print(f"Максимальный Шарп - ненулевых весов: {(max_sharpe_weights > 0.01).sum()}")

    # Равнонагруженный портфель
    equal_weights = optimize_portfolio(expected_returns, cov_matrix, risk_free_rate,
                                      method='equal_weight')

    weights_dict = {
        'min_variance': min_var_weights,
        'max_sharpe': max_sharpe_weights,
        'equal_weight': equal_weights
    }

    # Расчет характеристик портфелей
    print("\n2. Расчет характеристик портфелей...")

    results = []

    for name, weights in weights_dict.items():
        daily_return, daily_vol = portfolio_performance(weights, expected_returns, cov_matrix)
        annual_return, annual_vol = annualize_metrics(daily_return, daily_vol)

        sharpe_ratio = (annual_return - risk_free_rate * 252) / annual_vol if annual_vol > 0 else 0

        results.append({
            'Portfolio': name,
            'Annual Return (%)': annual_return * 100,
            'Annual Volatility (%)': annual_vol * 100,
            'Sharpe Ratio': sharpe_ratio
        })

    results_df = pd.DataFrame(results)
    print("\nСравнение портфелей:")
    print(results_df.round(4))

    # Детальный анализ весов
    print("\n3. Анализ весов портфелей...")

    tickers = expected_returns.index

    for name, weights in weights_dict.items():
        print(f"\n--- {name.upper()} ---")
        nonzero = weights > 0.001
        print(f"Количество активов в портфеле: {nonzero.sum()} из {len(weights)}")

        # Топ-10 активов
        top_10_idx = np.argsort(weights)[-10:][::-1]
        print("Топ-10 активов:")
        for idx in top_10_idx:
            if weights[idx] > 0:
                print(f"  {tickers[idx]:8s}: {weights[idx]*100:6.2f}%")

    # Построение эффективной границы
    print("\n4. Построение эффективной границы...")
    plot_efficient_frontier(expected_returns, cov_matrix, risk_free_rate,
                            min_var_weights, max_sharpe_weights)

    # Визуализация весов
    print("\n5. Визуализация весов портфелей...")
    plot_portfolio_weights(tickers, weights_dict)

    return results_df, weights_dict


def task_3(prices_df, window_years=1, step_months=6, risk_free_rate=0.16/252):
    """Выполнение задачи 3"""

    print("\n" + "="*80)
    print("ЗАДАЧА 3: Анализ динамики портфелей во времени")
    print("="*80)

    window_days = int(window_years * 252)
    step_days = int(step_months * 21)

    total_days = len(prices_df)
    tickers = prices_df.columns

    weights_history = {
        'min_variance': [],
        'max_sharpe': [],
        'equal_weight': []
    }

    dates_history = []

    print(f"\nПараметры анализа:")
    print(f"  - Размер окна: {window_years} года ({window_days} дней)")
    print(f"  - Шаг: {step_months} месяцев ({step_days} дней)")
    print(f"  - Всего окон: {(total_days - window_days) // step_days}")

    start_idx = 0

    while start_idx + window_days <= total_days:
        window_prices = prices_df.iloc[start_idx:start_idx + window_days]

        if len(window_prices) < window_days * 0.8:  # Пропускаем окна с недостатком данных
            start_idx += step_days
            continue

        # Расчет доходностей
        window_returns = np.log(window_prices / window_prices.shift(1)).dropna()

        # Фильтрация активов с большим количеством пропусков
        missing_pct = window_returns.isnull().sum() / len(window_returns)
        valid_assets = missing_pct[missing_pct < 0.1].index
        window_returns = window_returns[valid_assets].dropna()

        if len(window_returns) < 10 or len(valid_assets) < 5:
            start_idx += step_days
            continue

        # Оценка параметров
        exp_returns = window_returns.mean()
        cov_matrix = window_returns.cov()

        # Оптимизация для каждого типа портфеля
        try:
            # Равнонагруженный
            n = len(exp_returns)
            equal_w = np.ones(n) / n
            weights_history['equal_weight'].append(equal_w)

            # Минимальная дисперсия
            min_var_w = optimize_portfolio(exp_returns, cov_matrix, risk_free_rate, method='min_variance')
            weights_history['min_variance'].append(min_var_w)

            # Максимальный Шарп
            max_sharpe_w = optimize_portfolio(exp_returns, cov_matrix, risk_free_rate, method='max_sharpe')
            weights_history['max_sharpe'].append(max_sharpe_w)

            dates_history.append(prices_df.index[start_idx + window_days - 1])

        except Exception as e:
            print(f"Ошибка для окна {start_idx}: {e}")

        start_idx += step_days

    if not dates_history:
        print("\nНедостаточно данных для анализа динамики")
        return None

    print(f"\nУспешно обработано окон: {len(dates_history)}")

    # Анализ стабильности весов
    print("\n1. Анализ стабильности весов...")

    # Нормализуем весовые векторы к фиксированному набору тикеров
    all_tickers = set()
    for method_weights in weights_history.values():
        for w in method_weights:
            if hasattr(w, 'index'):
                all_tickers.update(w.index)

    all_tickers = sorted(list(all_tickers))

    stability_metrics = {}

    for method_name, weights_list in weights_history.items():
        # Расчет средних весов и стандартных отклонений
        weights_array = np.array(weights_list)

        mean_weights = weights_array.mean(axis=0)
        std_weights = weights_array.std(axis=0)
        cv_weights = std_weights / (mean_weights + 1e-10) * 100  # Коэффициент вариации в %

        # Метрики стабильности
        avg_assets = (weights_array > 0.01).sum(axis=1).mean()  # Среднее кол-во активов
        turnover = np.mean([np.abs(weights_array[i] - weights_array[i-1]).sum() / 2
                           for i in range(1, len(weights_array))]) if len(weights_array) > 1 else 0

        stability_metrics[method_name] = {
            'avg_num_assets': avg_assets,
            'turnover': turnover,
            'mean_std_weight': np.mean(std_weights),
            'mean_cv': np.mean(cv_weights[cv_weights < 1000])  # Исключаем выбросы
        }

    print("\nМетрики стабильности портфелей:")
    stability_df = pd.DataFrame(stability_metrics).T
    stability_df['turnover'] = stability_df['turnover'] * 100
    print(stability_df.round(4))

    # Визуализация динамики весов
    print("\n2. Визуализация динамики весов...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (method, weights_list) in enumerate(weights_history.items()):
        if idx >= 4:
            break

        ax = axes[idx]

        weights_array = np.array(weights_list)

        # Берем топ-5 активов по среднему весу
        mean_weights = weights_array.mean(axis=0)
        top_5_idx = np.argsort(mean_weights)[-5:][::-1]

        # Строим временной ряд для каждого актива
        for asset_idx in top_5_idx:
            if mean_weights[asset_idx] > 0:
                ax.plot(dates_history, weights_array[:, asset_idx],
                       marker='o', markersize=4, label=tickers[asset_idx],
                       linewidth=2, alpha=0.8)

        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Weight', fontsize=11)
        ax.set_title(f'{method.upper()} - Dynamics of Top 5 Assets',
                     fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('rebalancing_comparison.png', dpi=300, bbox_inches='tight')
    print("График динамики весов сохранен: rebalancing_comparison.png")
    plt.close()

    # Тепловая карта весов портфеля с максимальным Шарпом
    print("\n3. Построение тепловой карты...")
    max_sharpe_weights_array = np.array(weights_history['max_sharpe'])

    # Берем топ-10 активов
    mean_weights = max_sharpe_weights_array.mean(axis=0)
    top_10_idx = np.argsort(mean_weights)[-10:][::-1]
    top_10_tickers = [tickers[i] for i in top_10_idx if mean_weights[i] > 0]

    if len(top_10_tickers) > 0:
        top_10_weights = max_sharpe_weights_array[:, top_10_idx]

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(top_10_weights.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')

        ax.set_xticks(range(len(dates_history))[::max(1, len(dates_history)//10)])
        ax.set_xticklabels([d.strftime('%Y-%m') for d in dates_history][::max(1, len(dates_history)//10)],
                          rotation=45, ha='right')
        ax.set_yticks(range(len(top_10_tickers)))
        ax.set_yticklabels(top_10_tickers)
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Asset', fontsize=11)
        ax.set_title('Max Sharpe Portfolio - Weights Heatmap',
                    fontsize=12, fontweight='bold')

        plt.colorbar(im, ax=ax, label='Weight')
        plt.tight_layout()
        plt.savefig('portfolio_weights_heatmap.png', dpi=300, bbox_inches='tight')
        print("Тепловая карта сохранена: portfolio_weights_heatmap.png")
        plt.close()

    # Выводы по стабильности
    print("\n4. Выводы по стабильности:")
    for method, metrics in stability_metrics.items():
        print(f"\n--- {method.upper()} ---")
        print(f"  Среднее количество активов: {metrics['avg_num_assets']:.1f}")
        print(f"  Средний оборот (turnover): {metrics['turnover']:.2f}%")
        print(f"  Среднее СКО весов: {metrics['mean_std_weight']:.4f}")
        print(f"  Средний коэффициент вариации: {metrics['mean_cv']:.1f}%")

        if method == 'equal_weight':
            print("  -> Равнонагруженный портфель: максимально стабильный, нулевой оборот")
        elif method == 'min_variance':
            print(f"  -> Портфель мин. дисперсии: {'стабильный' if metrics['turnover'] < 0.2 else 'нестабильный'}")
        elif method == 'max_sharpe':
            print(f"  -> Портфель макс. Шарпа: {'стабильный' if metrics['turnover'] < 0.3 else 'нестабильный'}")

    return stability_df, weights_history, dates_history


def main():
    """Главная функция выполнения всех задач"""

    print("="*80)
    print("АНАЛИЗ ИНВЕСТИЦИОННОГО ПОРТФЕЛЯ")
    print("Задачи 2 и 3")
    print("="*80)

    # Загрузка данных
    print("\nШаг 1: Загрузка данных")
    prices_df = load_data('data/prices_moex_new.csv')

    # Расчет доходностей
    print("\nШаг 2: Расчет доходностей")
    returns_df = calculate_returns(prices_df)

    # Оценка параметров
    print("\nШаг 3: Оценка параметров")
    expected_returns, annual_returns, cov_matrix, annual_cov = estimate_parameters(returns_df)

    # Выполнение задачи 2
    print("\n" + "="*80)
    results_df, weights_dict = task_2(returns_df, expected_returns, cov_matrix, RISK_FREE_RATE)

    # Выполнение задачи 3
    print("\n" + "="*80)
    stability_df, weights_history, dates_history = task_3(
        prices_df, window_years=1, step_months=6, risk_free_rate=RISK_FREE_RATE
    )

    print("\n" + "="*80)
    print("ЗАВЕРШЕНО!")
    print("Созданные файлы:")
    print("  - efficient_frontier.png")
    print("  - portfolio_weights.png")
    print("  - rebalancing_comparison.png")
    print("  - portfolio_weights_heatmap.png")
    print("="*80)

    return results_df, stability_df


if __name__ == "__main__":
    results, stability = main()
