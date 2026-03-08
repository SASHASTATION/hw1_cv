"""
Модуль глобальной модели движения камеры.

Поддерживает две модели:
- Аффинная : estimateAffine2D
- Гомография : findHomography

Каждая модель разбирается на компоненты dx, dy, da (угол)
и собирается обратно в матрицу трансформации.
"""

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════════
#  ОЦЕНКА МОДЕЛИ
# ═══════════════════════════════════════════════════════════════════

def estimate_affine(pts1, pts2):
    """
    Оценка аффинной трансформации с RANSAC.

    Returns:
        M: матрица 2×3 (или None при ошибке)
        inliers: маска инлайеров
    """
    M, inliers = cv2.estimateAffine2D(pts1, pts2)
    return M, inliers


def estimate_homography(pts1, pts2):
    """
    Оценка гомографии  с RANSAC.

    Returns:
        H: матрица 3×3 (или None при ошибке)
        inliers: маска инлайеров
    """
    H, inliers = cv2.findHomography(pts1, pts2, cv2.RANSAC)
    return H, inliers


def estimate_motion(pts1, pts2, model="affine"):
    """
    Универсальная точка входа для оценки модели.

    Args:
        pts1, pts2: массивы точек (N, 2)
        model: "affine" или "homography"

    Returns:
        transform: матрица трансформации (2×3 или 3×3)
        inliers: маска инлайеров
    """
    if model == "affine":
        return estimate_affine(pts1, pts2)
    elif model == "homography":
        return estimate_homography(pts1, pts2)
    else:
        raise ValueError(f"Неизвестная модель: {model}. Используйте 'affine' или 'homography'.")


# ═══════════════════════════════════════════════════════════════════
#  РАЗЛОЖЕНИЕ НА КОМПОНЕНТЫ (dx, dy, da)
# ═══════════════════════════════════════════════════════════════════

def affine_to_motion(M):
    """
    Извлекает dx, dy, da (угол поворота) из аффинной матрицы 2×3.

    | cos(a)  -sin(a)  dx |
    | sin(a)   cos(a)  dy |
    """
    if M is None:
        return 0.0, 0.0, 0.0

    dx = float(M[0, 2])
    dy = float(M[1, 2])
    da = float(np.arctan2(M[1, 0], M[0, 0]))

    return dx, dy, da


def homography_to_motion(H):
    """
    Извлекает приблизительные dx, dy, da из гомографии 3×3.

    Гомография содержит больше параметров,
    но для стабилизации нам нужны основные: сдвиг и поворот.
    """
    if H is None:
        return 0.0, 0.0, 0.0

    dx = float(H[0, 2])
    dy = float(H[1, 2])
    da = float(np.arctan2(H[1, 0], H[0, 0]))

    return dx, dy, da


def transform_to_motion(transform, model="affine"):
    """Универсальное разложение трансформации на dx, dy, da."""
    if model == "affine":
        return affine_to_motion(transform)
    elif model == "homography":
        return homography_to_motion(transform)
    else:
        raise ValueError(f"Неизвестная модель: {model}")


# ═══════════════════════════════════════════════════════════════════
#  СБОРКА МАТРИЦЫ ИЗ КОМПОНЕНТОВ
# ═══════════════════════════════════════════════════════════════════

def motion_to_affine(dx, dy, da):
    """Собирает аффинную матрицу 2×3 из dx, dy, da."""
    cos_a = np.cos(da)
    sin_a = np.sin(da)
    M = np.array([
        [cos_a, -sin_a, dx],
        [sin_a,  cos_a, dy],
    ], dtype=np.float32)
    return M


def motion_to_homography(dx, dy, da):
    """Собирает гомографию 3×3 из dx, dy, da (без перспективных членов)."""
    cos_a = np.cos(da)
    sin_a = np.sin(da)
    H = np.array([
        [cos_a, -sin_a, dx],
        [sin_a,  cos_a, dy],
        [0,      0,     1 ],
    ], dtype=np.float32)
    return H


def motion_to_transform(dx, dy, da, model="affine"):
    """Универсальная сборка матрицы трансформации."""
    if model == "affine":
        return motion_to_affine(dx, dy, da)
    elif model == "homography":
        return motion_to_homography(dx, dy, da)
    else:
        raise ValueError(f"Неизвестная модель: {model}")


# ═══════════════════════════════════════════════════════════════════
#  МЕТРИКИ КАЧЕСТВА
# ═══════════════════════════════════════════════════════════════════

def compute_residual(pts_actual, pts_predicted):
    """
    Средняя ошибка (residual) между реальными и предсказанными точками.

    Args:
        pts_actual: (N, 2)
        pts_predicted: (N, 2)

    Returns:
        float: средняя евклидова ошибка
    """
    diff = pts_actual - pts_predicted
    err = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
    return float(np.mean(err))


def apply_transform_to_points(pts, transform, model="affine"):
    """
    Применяет трансформацию к набору точек.

    Args:
        pts: (N, 2) точки
        transform: матрица (2×3 или 3×3)
        model: "affine" или "homography"

    Returns:
        pts_transformed: (N, 2)
    """
    pts_col = pts.reshape(-1, 1, 2).astype(np.float32)

    if model == "affine":
        result = cv2.transform(pts_col, transform)
    elif model == "homography":
        result = cv2.perspectiveTransform(pts_col, transform)
    else:
        raise ValueError(f"Неизвестная модель: {model}")

    return result.reshape(-1, 2)


def mean_transform_magnitude(transform, h, w, step=15, model="affine"):
    """
    Средняя величина смещения, которое даёт трансформация.
    Меряется на сетке точек.
    """
    if transform is None:
        return 0.0

    y, x = np.mgrid[step:h:step, step:w:step]
    pts = np.stack((x, y), axis=-1).astype(np.float32).reshape(-1, 2)

    pts_transformed = apply_transform_to_points(pts, transform, model)
    diff = pts_transformed - pts
    mag = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)

    return float(np.mean(mag))
