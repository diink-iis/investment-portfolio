# Задачи 13-15: Оценка входящих данных для optimizer на основе исторических β

## Описание

Этот модуль реализует оценку входящих данных для портфельного оптимизатора на основе исторических β (бета-коэффициентов).

### Задача 13
Рассчитать ковариационную матрицу на основе исторических β, которые оцениваются согласно рыночной модели (market model).

### Задача 14
Построить границу эффективных портфелей на основе полученной ковариационной матрицы.

### Задача 15
Построить границу эффективных портфелей для разных исторических окон и продемонстрировать динамику её изменения.

## Выбор из задач 11-12

**Индекс:** IMOEX (Мосбиржа) — используется как рыночный индекс

**Историческое окно:** Скользящее окно длиной в 1 год (252 торговых дня)

**Схема взвешивания:** Равные веса наблюдений (простое скользящее окно)

## Структура

- [task_13_14_15.py](task_13_14_15.py) - основной модуль с реализацией
- [test_task_13_14_15.ipynb](test_task_13_14_15.ipynb) - проверочный Jupyter Notebook

## Использование

### Основные функции

#### `calculate_market_model_betas(stock_returns, market_returns)`
Расчет параметров рыночной модели для одной акции.

**Параметры:**
- `stock_returns` - доходности акции
- `market_returns` - доходности рыночного индекса

**Возвращает:** кортеж (alpha, beta)

**Рыночная модель:** `r_i = α_i + β_i * r_m + ε_i`

#### `calculate_all_betas(returns, market_ticker='MOEX')`
Расчет исторических бета для всех акций относительно рыночного индекса.

**Параметры:**
- `returns` - DataFrame с доходностями всех акций и индекса
- `market_ticker` - тикер рыночного индекса

**Возвращает:** DataFrame с бета для всех акций

#### `calculate_covariance_from_betas(betas, market_variance, residual_variances=None)`
Расчет ковариационной матрицы на основе рыночной модели.

**Параметры:**
- `betas` - вектор бета-коэффициентов
- `market_variance` - дисперсия рыночного индекса
- `residual_variances` - дисперсии остатков (опционально)

**Возвращает:** ковариационная матрица

**Формула:**
- `Cov(r_i, r_j) = β_i * β_j * σ_m²` (при некоррелированных остатках)
- `Var(r_i) = β_i² * σ_m² + σ_ε_i²` (с остаточными дисперсиями)

#### `task_13_covariance_from_historical_betas(returns, market_ticker='MOEX', include_residuals=True)`
Задача 13: Рассчитать ковариационную матрицу на основе исторических β.

**Возвращает:** словарь с ковариационной матрицей и метаданными

#### `task_14_efficient_frontier_from_betas(cov_matrix, mean_returns, n_points=100, method_name='Historical Betas')`
Задача 14: Построить границу эффективных портфелей на основе β.

**Возвращает:** кортеж (массив доходностей, массив стандартных отклонений)

#### `task_15_efficient_frontier_dynamics_betas(returns, market_ticker='MOEX', window_size='1Y', step_size='1Y', include_residuals=True, n_points=100)`
Задача 15: Динамика эффективных границ на основе β.

**Возвращает:** кортеж (словарь границ, DataFrame с метриками стабильности)

### Пример использования

```python
from task_13_14_15 import (
    task_13_covariance_from_historical_betas,
    task_14_efficient_frontier_from_betas,
    task_15_efficient_frontier_dynamics_betas
)

# Загрузка данных
prices = load_prices_data('../data/prices_moex_new.csv')
returns = calculate_returns(prices)

# Задача 13: Ковариационная матрица на основе β
betas_result = task_13_covariance_from_historical_betas(
    returns, market_ticker='MOEX', include_residuals=True
)

# Задача 14: Эффективная граница
ef_returns, ef_stds = task_14_efficient_frontier_from_betas(
    betas_result['cov_matrix'],
    mean_returns,
    n_points=50
)

# Задача 15: Динамика границ
beta_frontiers, beta_stability = task_15_efficient_frontier_dynamics_betas(
    returns, market_ticker='MOEX', window_size='1Y', step_size='1Y',
    include_residuals=True, n_points=50
)
```

## Проверка

Для проверки работоспособности запустите:

```bash
cd task-13-14-15
python3 task_13_14_15.py
```

Или откройте [test_task_13_14_15.ipynb](test_task_13_14_15.ipynb) в Jupyter Notebook.

## Зависимости

- numpy
- pandas
- matplotlib
- seaborn

Модуль использует функции из:
- [task_2_3](../task_2_3/): calculate_returns, rolling_window_analysis
- [task_9-10](../task_9-10/): calculate_efficient_frontier, analyze_efficient_frontier_stability

## Результаты

### Задача 13
- Вектор бета-коэффициентов для всех акций (29 акций, исключая индекс MOEX)
- Ковариационная матрица (29x29) на основе рыночной модели
- Опционально: учет остаточных дисперсий (idiosyncratic risk)

### Задача 14
- Эффективная граница портфелей на основе β-модели
- Минимальный риск, максимальная доходность
- Максимальное Шарп-отношение

### Задача 15
- Эффективные границы для 10 окон (скользящее окно 1 год)
- Метрики стабильности: минимальный риск, доходность, Шарп-отношение
- Динамика изменений во времени

## Интерпретация бета

| Значение β | Интерпретация |
|-------------|----------------|
| β > 1 | Агрессивная акция (более волатильна, чем рынок) |
| β = 1 | Соразмерна рынку |
| 0 < β < 1 | Защитная акция (менее волатильна, чем рынок) |
| β < 0 | Обратная корреляция с рынком |
| β = 0 | Нет корреляции с рынком |

## Теоретическая основа

### Рыночная модель (Market Model)

$$r_i = \alpha_i + \beta_i \cdot r_m + \varepsilon_i$$

где:
- $r_i$ — доходность акции i
- $r_m$ — доходность рыночного индекса
- $\alpha_i$ — альфа (аномальная доходность)
- $\beta_i$ — бета-коэффициент (чувствительность к рынку)
- $\varepsilon_i$ — ошибка модели (остаточный риск)

### Бета-коэффициент

$$\beta_i = \frac{Cov(r_i, r_m)}{Var(r_m)}$$

### Ковариационная матрица на основе β

$$Cov(r_i, r_j) = \beta_i \cdot \beta_j \cdot \sigma_m^2$$

$$Var(r_i) = \beta_i^2 \cdot \sigma_m^2 + \sigma_{\varepsilon_i}^2$$

где:
- $\sigma_m^2$ — дисперсия рыночного индекса
- $\sigma_{\varepsilon_i}^2$ — остаточная дисперсия (idiosyncratic risk)
