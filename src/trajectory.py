"""
Модуль траектории движения камеры.

Содержит:
- Trajectory: накопление сырой траектории и сглаженной
- Три сглаживателя с единым интерфейсом:
    - KalmanSmoother1D
    - MovingAverageSmoother
    - ExponentialSmoother
"""

import cv2
import numpy as np
from collections import deque


# ═══════════════════════════════════════════════════════════════════
#  СГЛАЖИВАТЕЛИ  (единый интерфейс: smooth(value) → float)
# ═══════════════════════════════════════════════════════════════════

class KalmanSmoother1D:
    """
    Фильтр Калмана для одномерного сигнала.

    Состояние: [значение, скорость].
    Хорошо подходит для плавного сглаживания с учётом динамики.
    """

    def __init__(self, process_noise=1e-4, measurement_noise=1e-1):
        kf = cv2.KalmanFilter(2, 1)

        kf.transitionMatrix = np.array([
            [1, 1],
            [0, 1],
        ], dtype=np.float32)

        kf.measurementMatrix = np.array([[1, 0]], dtype=np.float32)

        kf.processNoiseCov = np.eye(2, dtype=np.float32) * process_noise
        kf.measurementNoiseCov = np.array([[measurement_noise]], dtype=np.float32)

        kf.errorCovPost = np.eye(2, dtype=np.float32)
        kf.statePost = np.zeros((2, 1), dtype=np.float32)

        self._kf = kf

    def smooth(self, value):
        """Принимает измерение, возвращает сглаженное значение."""
        self._kf.predict()
        measurement = np.array([[np.float32(value)]])
        estimated = self._kf.correct(measurement)
        return float(estimated[0, 0])


class MovingAverageSmoother:
    """
    Скользящее среднее по окну фиксированного размера.

    Простой и предсказуемый фильтр, но вносит задержку.
    """

    def __init__(self, window=30):
        self._window = window
        self._buffer = deque(maxlen=window)

    def smooth(self, value):
        self._buffer.append(value)
        return float(np.mean(self._buffer))


class ExponentialSmoother:
    """
    Экспоненциальное сглаживание (EMA).

    s_t = α * x_t + (1 − α) * s_{t-1}

    Маленький α → сильное сглаживание, большая задержка.
    """

    def __init__(self, alpha=0.05):
        self._alpha = alpha
        self._value = None

    def smooth(self, value):
        if self._value is None:
            self._value = value
        else:
            self._value = self._alpha * value + (1 - self._alpha) * self._value
        return float(self._value)


# ═══════════════════════════════════════════════════════════════════
#  ФАБРИКА СГЛАЖИВАТЕЛЕЙ
# ═══════════════════════════════════════════════════════════════════

def create_smoother(name, **kwargs):
    """
    Создаёт сглаживатель по имени.

    Args:
        name: "kalman", "moving_avg" или "exponential"
        **kwargs: параметры для конкретного сглаживателя

    Returns:
        экземпляр сглаживателя с методом smooth(value) → float
    """
    if name == "kalman":
        return KalmanSmoother1D(
            process_noise=kwargs.get("process_noise", 1e-4),
            measurement_noise=kwargs.get("measurement_noise", 1e-1),
        )
    elif name == "moving_avg":
        return MovingAverageSmoother(
            window=kwargs.get("window", 30),
        )
    elif name == "exponential":
        return ExponentialSmoother(
            alpha=kwargs.get("alpha", 0.05),
        )
    else:
        raise ValueError(f"Неизвестный сглаживатель: {name}")


# ═══════════════════════════════════════════════════════════════════
#  ТРАЕКТОРИЯ
# ═══════════════════════════════════════════════════════════════════

class Trajectory:
    """
    Накапливает траекторию камеры (x, y, angle) покадрово.

    Хранит:
    - raw: исходная (кумулятивная) траектория
    - smoothed: сглаженная траектория
    - deltas: покадровые смещения
    """

    def __init__(self, smoother_name="kalman", **smoother_kwargs):
        # три независимых сглаживателя для x, y, angle
        self._smoother_x = create_smoother(smoother_name, **smoother_kwargs)
        self._smoother_y = create_smoother(smoother_name, **smoother_kwargs)
        self._smoother_a = create_smoother(smoother_name, **smoother_kwargs)

        # кумулятивная (сырая) траектория
        self._x = 0.0
        self._y = 0.0
        self._a = 0.0

        # история
        self.raw_x = []
        self.raw_y = []
        self.raw_a = []

        self.smooth_x = []
        self.smooth_y = []
        self.smooth_a = []

        self.deltas_x = []
        self.deltas_y = []
        self.deltas_a = []

    def update(self, dx, dy, da):
        """
        Добавляет покадровое смещение и возвращает стабилизационную поправку.

        Args:
            dx, dy, da: смещение между предыдущим и текущим кадром

        Returns:
            (diff_x, diff_y, diff_a): разность между сглаженной и сырой траекторией
        """
        # сохраняем дельты
        self.deltas_x.append(dx)
        self.deltas_y.append(dy)
        self.deltas_a.append(da)

        # обновляем кумулятивную траекторию
        self._x += dx
        self._y += dy
        self._a += da

        self.raw_x.append(self._x)
        self.raw_y.append(self._y)
        self.raw_a.append(self._a)

        # сглаживаем
        sx = self._smoother_x.smooth(self._x)
        sy = self._smoother_y.smooth(self._y)
        sa = self._smoother_a.smooth(self._a)

        self.smooth_x.append(sx)
        self.smooth_y.append(sy)
        self.smooth_a.append(sa)

        # поправка = сглаженная − сырая
        diff_x = sx - self._x
        diff_y = sy - self._y
        diff_a = sa - self._a

        return diff_x, diff_y, diff_a
