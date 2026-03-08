"""
Основной pipeline стабилизации видео.

Шаги:
1. Читаем пары кадров
2. Вычисляем оптический поток (Farnebäck)
3. Оцениваем глобальную модель движения (affine / homography)
4. Накапливаем и сглаживаем траекторию
5. Warp каждого кадра к стабилизированной траектории
6. Записываем результат
"""

import cv2
import numpy as np

from src.optical_flow import compute_farneback, sample_flow_on_grid, mean_flow_magnitude
from src.motion_model import (
    estimate_motion,
    transform_to_motion,
    motion_to_transform,
    apply_transform_to_points,
    compute_residual,
    mean_transform_magnitude,
)
from src.trajectory import Trajectory


# ═══════════════════════════════════════════════════════════════════
#  WARP КАДРА
# ═══════════════════════════════════════════════════════════════════

def warp_frame(frame, transform, model="affine", crop_scale=1.04):
    """
    Применяет трансформацию к кадру и масштабирует для обрезки чёрных краёв.

    Args:
        frame: BGR кадр
        transform: матрица (2×3 для affine, 3×3 для homography)
        model: "affine" или "homography"
        crop_scale: масштаб увеличения (> 1 убирает чёрные края)

    Returns:
        warped: трансформированный кадр
    """
    h, w = frame.shape[:2]

    if model == "affine":
        warped = cv2.warpAffine(frame, transform, (w, h))
    elif model == "homography":
        warped = cv2.warpPerspective(frame, transform, (w, h))
    else:
        raise ValueError(f"Неизвестная модель: {model}")

    # масштабирование от центра для обрезки чёрных краёв
    if crop_scale != 1.0:
        center = (w // 2, h // 2)
        scale_matrix = cv2.getRotationMatrix2D(center, 0, crop_scale)
        warped = cv2.warpAffine(warped, scale_matrix, (w, h))

    return warped


# ═══════════════════════════════════════════════════════════════════
#  РЕЗУЛЬТАТЫ СТАБИЛИЗАЦИИ (контейнер метрик)
# ═══════════════════════════════════════════════════════════════════

class StabilizationResult:
    """Хранит все метрики и данные, собранные во время стабилизации."""

    def __init__(self):
        self.flow_magnitudes = []       # средняя величина Farnebäck
        self.transform_magnitudes = []  # средняя величина модели
        self.residuals = []             # ошибка модели
        self.trajectory = None          # объект Trajectory
        self.sample_frames = {}         # {frame_idx: (original, stabilized)}
        self.sample_flows = {}          # {frame_idx: flow_hsv}
        self.fps = 0.0
        self.total_frames = 0


# ═══════════════════════════════════════════════════════════════════
#  PIPELINE
# ═══════════════════════════════════════════════════════════════════

def stabilize_video(
    input_path,
    output_path,
    farneback_params,
    grid_step=15,
    motion_model="affine",
    smoother_name="kalman",
    smoother_kwargs=None,
    crop_scale=1.04,
    blur_kernel=(5, 5),
    save_frames_at=None,
):
    """
    Полный pipeline стабилизации видео.

    Args:
        input_path: путь к входному видео
        output_path: путь для сохранения стабилизированного видео
        farneback_params: параметры Farnebäck (dict)
        grid_step: шаг сетки для сэмплирования
        motion_model: "affine" или "homography"
        smoother_name: "kalman", "moving_avg" или "exponential"
        smoother_kwargs: доп. параметры сглаживателя
        crop_scale: масштаб для обрезки чёрных краёв
        blur_kernel: размер ядра Гаусса для предобработки
        save_frames_at: список индексов кадров для сохранения (до/после)

    Returns:
        StabilizationResult с собранными метриками
    """
    if smoother_kwargs is None:
        smoother_kwargs = {}
    if save_frames_at is None:
        save_frames_at = []

    save_frames_set = set(save_frames_at)

    # ── открываем видео ──
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError("Не удалось прочитать первый кадр")

    h, w = prev_frame.shape[:2]
    prev_gray = cv2.GaussianBlur(
        cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), blur_kernel, 0
    )

    # ── writer ──
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    writer.write(prev_frame)  # первый кадр без изменений

    # ── инициализация ──
    trajectory = Trajectory(smoother_name, **smoother_kwargs)
    result = StabilizationResult()
    result.trajectory = trajectory
    result.fps = fps
    result.total_frames = total

    # сохраняем первый кадр, если нужно
    if 0 in save_frames_set:
        result.sample_frames[0] = (prev_frame.copy(), prev_frame.copy())

    frame_idx = 0
    print(f"Стабилизация: {total} кадров, модель={motion_model}, фильтр={smoother_name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        curr_gray = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), blur_kernel, 0
        )

        # 1. оптический поток
        flow = compute_farneback(prev_gray, curr_gray, farneback_params)
        result.flow_magnitudes.append(mean_flow_magnitude(flow))

        # 2. сэмплирование + модель
        pts1, pts2 = sample_flow_on_grid(flow, grid_step)
        transform, inliers = estimate_motion(pts1, pts2, model=motion_model)

        # метрики
        if transform is not None:
            pts2_pred = apply_transform_to_points(pts1, transform, model=motion_model)
            residual = compute_residual(pts2, pts2_pred)
            mag = mean_transform_magnitude(transform, h, w, grid_step, model=motion_model)
        else:
            residual = 0.0
            mag = 0.0

        result.residuals.append(residual)
        result.transform_magnitudes.append(mag)

        # 3. разложение на dx, dy, da
        dx, dy, da = transform_to_motion(transform, model=motion_model)

        # 4. обновление траектории + поправка
        diff_x, diff_y, diff_a = trajectory.update(dx, dy, da)

        # 5. стабилизационная матрица
        dx_stab = dx + diff_x
        dy_stab = dy + diff_y
        da_stab = da + diff_a
        M_stab = motion_to_transform(dx_stab, dy_stab, da_stab, model=motion_model)

        # 6. warp
        stabilized = warp_frame(frame, M_stab, model=motion_model, crop_scale=crop_scale)
        writer.write(stabilized)

        # сохраняем примеры
        if frame_idx in save_frames_set:
            result.sample_frames[frame_idx] = (frame.copy(), stabilized.copy())
            from src.optical_flow import flow_to_hsv
            result.sample_flows[frame_idx] = flow_to_hsv(flow)

        # прогресс
        if frame_idx % 100 == 0:
            print(f"  кадр {frame_idx}/{total}")

        prev_gray = curr_gray

    cap.release()
    writer.release()

    print(f"Готово! Стабилизированное видео: {output_path}")
    return result
