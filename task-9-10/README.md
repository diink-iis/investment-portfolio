# Задачи 9 и 10: Анализ стабильности во времени границы эффективных портфелей

## Описание

Этот модуль реализует анализ динамики эффективной границы портфелей без ограничений на короткие продажи.

### Задача 9
- **9a**: Демонстрация динамики эффективной границы скользящим окном
- **9b**: Демонстрация динамики эффективной границы расширяющимся окном

### Задача 10
- Выполнение задачи 9 с использованием схемы взвешивания наблюдений с экспоненциальным забыванием

## Структура

- [task_9_10.py](task_9_10.py) - основной модуль с реализацией
- [test_task_9_10.ipynb](test_task_9_10.ipynb) - проверочный Jupyter Notebook

## Использование

### Основные функции

#### `calculate_efficient_frontier(mean_returns, cov_matrix, n_points=100)`
Расчет эффективной границы для заданных параметров.

**Параметры:**
- `mean_returns` - вектор средних доходностей
- `cov_matrix` - ковариационная матрица
- `n_points` - количество точек на границе

**Возвращает:** кортеж (массив доходностей, массив стандартных отклонений)

#### `task_9a_efficient_frontier_dynamics_rolling(returns, window_size='1Y', step_size='1Y', n_points=100)`
Анализ динамики эффективной границы скользящим окном.

**Возвращает:** кортеж (словарь границ, DataFrame с метриками стабильности)

#### `task_9b_efficient_frontier_dynamics_expanding(returns, step_size='1Y', n_points=100)`
Анализ динамики эффективной границы расширяющимся окном.

**Возвращает:** кортеж (словарь границ, DataFrame с метриками стабильности)

#### `task_10_efficient_frontier_dynamics_exponential(returns, window_size='1Y', step_size='1Y', lambda_param=0.94, n_points=100)`
Анализ динамики эффективной границы с экспоненциальным забыванием.

**Возвращает:** кортеж (словарь границ, DataFrame с метриками стабильности)

### Пример использования

```python
from task_9_10 import task_9a_efficient_frontier_dynamics_rolling

# Загрузка данных
prices = load_prices_data('../data/prices_moex_new.csv')
returns = calculate_returns(prices)

# Анализ скользящим окном
frontiers, stability = task_9a_efficient_frontier_dynamics_rolling(
    returns, window_size='1Y', step_size='1Y'
)

# Визуализация
import matplotlib.pyplot as plt
for date, frontier in sorted(frontiers.items()):
    plt.plot(frontier['stds'], frontier['returns'], label=date)
plt.legend()
plt.show()
```

## Проверка

Для проверки работоспособности запустите:

```bash
cd task-9-10
python3 task_9_10.py
```

Или откройте [test_task_9_10.ipynb](test_task_9_10.ipynb) в Jupyter Notebook.

## Зависимости

- numpy
- pandas
- matplotlib
- seaborn

Модуль использует функции из [task_2_3](../task-2-3/):
- load_prices_data
- calculate_returns
- rolling_window_analysis
- expanding_window_analysis

## Результаты

Анализ возвращает:
- Словарь эффективных границ для каждого окна с доходностями, рисками и весами
- DataFrame с метриками стабильности:
  - min_std - минимальный риск (GMVP)
  - min_std_return - доходность GMVP
  - max_return - максимальная доходность
  - max_return_std - риск максимальной доходности
  - max_sharpe - максимальное Шарп-отношение
  - efficiency_ratio - коэффициент эффективности (max_return / min_std)
  - ... и изменения этих метрик между периодами
