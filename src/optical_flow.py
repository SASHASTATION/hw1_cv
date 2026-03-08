"""
Модуль оптического потока.

Содержит:
- вычисление плотного потока (Farnebäck)
- сэмплирование потока на регулярной сетке
- среднюю величину потока
- визуализации: стрелки и HSV-карта
"""

import cv2
import numpy as np


# ─────────────────────── вычисление потока ───────────────────────

def compute_farneback(prev_gray, curr_gray, params):
    """
    Плотный оптический поток Farnebäck.

    Args:
        prev_gray: предыдущий кадр (grayscale, uint8)
        curr_gray: текущий кадр (grayscale, uint8)
        params: словарь параметров (см. config.FARNEBACK_PARAMS)

    Returns:
        flow: массив (H, W, 2) — dx, dy для каждого пикселя
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        params["pyr_scale"],
        params["levels"],
        params["winsize"],
        params["iterations"],
        params["poly_n"],
        params["poly_sigma"],
        params["flags"],
    )
    return flow


# ─────────────────────── сэмплирование ──────────────────────────

def sample_flow_on_grid(flow, step):
    """
    Сэмплирует оптический поток на регулярной сетке.

    Args:
        flow: (H, W, 2) поток
        step: шаг сетки в пикселях

    Returns:
        pts1: (N, 2) координаты точек на предыдущем кадре
        pts2: (N, 2) координаты точек на текущем кадре (pts1 + flow)
    """
    h, w = flow.shape[:2]

    y_coords, x_coords = np.mgrid[step:h:step, step:w:step]

    fx = flow[y_coords, x_coords, 0]
    fy = flow[y_coords, x_coords, 1]

    pts1 = np.stack((x_coords, y_coords), axis=-1).astype(np.float32)
    pts2 = pts1.copy()
    pts2[..., 0] += fx
    pts2[..., 1] += fy

    pts1 = pts1.reshape(-1, 2)
    pts2 = pts2.reshape(-1, 2)

    return pts1, pts2


# ─────────────────────── метрики ─────────────────────────────────

def mean_flow_magnitude(flow):
    """Средняя величина потока по всему кадру."""
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(np.mean(mag))


# ─────────────────────── визуализации ────────────────────────────

def draw_flow_arrows(frame, pts1, pts2, color=(0, 255, 0), thickness=1):
    """
    Рисует стрелки оптического потока на кадре.

    Args:
        frame: кадр BGR (будет скопирован)
        pts1, pts2: массивы точек (N, 2)
        color: цвет стрелок
        thickness: толщина

    Returns:
        vis: кадр с нарисованными стрелками
    """
    vis = frame.copy()
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        cv2.arrowedLine(
            vis,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color, thickness, tipLength=0.3,
        )
    return vis


def flow_to_hsv(flow):
    """
    Цветовая HSV-карта оптического потока.

    Направление → Hue, величина → Value.

    Args:
        flow: (H, W, 2)

    Returns:
        bgr: (H, W, 3) BGR-изображение для отображения
    """
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2        # Hue: направление
    hsv[..., 1] = 255                            # Saturation: максимум
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value: величина

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr
