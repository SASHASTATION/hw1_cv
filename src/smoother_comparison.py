"""
Утилита для сравнения трёх сглаживателей на одних и тех же данных.

Прогоняет покадровые дельты через Kalman, Moving Average
и Exponential Smoothing, возвращает сглаженные траектории.
"""

from src.trajectory import create_smoother


def compare_smoothers(deltas_x, deltas_y, deltas_a, configs=None):
    """
    Сравнивает несколько сглаживателей на одних данных.

    Args:
        deltas_x, deltas_y, deltas_a: покадровые смещения (списки float)
        configs: dict {"name": {**kwargs}} для каждого сглаживателя
                 По умолчанию: kalman, moving_avg, exponential

    Returns:
        dict вида:
        {
            "kalman": {"x": [...], "y": [...], "a": [...]},
            "moving_avg": {"x": [...], "y": [...], "a": [...]},
            ...
        }
    """
    if configs is None:
        configs = {
            "kalman": {"process_noise": 1e-4, "measurement_noise": 1e-1},
            "moving_avg": {"window": 30},
            "exponential": {"alpha": 0.05},
        }

    results = {}

    for name, kwargs in configs.items():
        smoother_x = create_smoother(name, **kwargs)
        smoother_y = create_smoother(name, **kwargs)
        smoother_a = create_smoother(name, **kwargs)

        smooth_x, smooth_y, smooth_a = [], [], []
        cum_x, cum_y, cum_a = 0.0, 0.0, 0.0

        for dx, dy, da in zip(deltas_x, deltas_y, deltas_a):
            cum_x += dx
            cum_y += dy
            cum_a += da

            smooth_x.append(smoother_x.smooth(cum_x))
            smooth_y.append(smoother_y.smooth(cum_y))
            smooth_a.append(smoother_a.smooth(cum_a))

        results[name] = {"x": smooth_x, "y": smooth_y, "a": smooth_a}

    # также вернём raw для удобства
    raw_x, raw_y, raw_a = [], [], []
    cx, cy, ca = 0.0, 0.0, 0.0
    for dx, dy, da in zip(deltas_x, deltas_y, deltas_a):
        cx += dx
        cy += dy
        ca += da
        raw_x.append(cx)
        raw_y.append(cy)
        raw_a.append(ca)

    results["_raw"] = {"x": raw_x, "y": raw_y, "a": raw_a}

    return results
