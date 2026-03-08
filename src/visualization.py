"""
Модуль визуализаций.

Содержит все функции для построения графиков и сохранения
визуальных сравнений.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════
#  ГРАФИКИ ТРАЕКТОРИИ
# ═══════════════════════════════════════════════════════════════════
def draw_analysis_frame(frame, pts1, pts2, pts2_model, metrics, model="affine"):
    """
    Рисует на кадре стрелки потока, стрелки модели и текст метрик.

    Args:
        frame: BGR кадр (будет скопирован)
        pts1: (N,2) точки на предыдущем кадре
        pts2: (N,2) точки на текущем кадре (по потоку)
        pts2_model: (N,2) точки предсказанные моделью (или None)
        metrics: dict с ключами — название, значениями — числа
        model: для подписи

    Returns:
        vis: кадр с аннотациями
    """
    vis = frame.copy()

    # зелёные стрелки — Farnebäck
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        cv2.arrowedLine(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                        (0, 255, 0), 1, tipLength=0.3)

    # красные стрелки — глобальная модель
    if pts2_model is not None:
        for (x1, y1), (x2, y2) in zip(pts1, pts2_model):
            cv2.arrowedLine(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                            (0, 0, 255), 1, tipLength=0.3)

    # текст метрик
    y_offset = 30
    colors = {
        "Farneback mean": (0, 255, 0),
        "Model mean": (0, 0, 255),
        "Residual": (0, 255, 255),
        "dx": (255, 255, 0),
        "dy": (255, 255, 0),
        "da": (255, 255, 0),
    }
    for key, value in metrics.items():
        color = colors.get(key, (255, 255, 255))
        text = f"{key}: {value:.3f}"
        cv2.putText(vis, text, (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += 30

    return vis


def create_annotated_side_by_side_video(input_path, output_path,
                                         farneback_params, grid_step=15,
                                         motion_model="affine",
                                         blur_kernel=(5, 5)):
    """
    Создаёт видео: слева — оригинал со стрелками и метриками,
                    справа — HSV-карта потока.

    Это «диагностическое» видео для визуального анализа.
    """
    from src.optical_flow import (compute_farneback, sample_flow_on_grid,
                                  mean_flow_magnitude, flow_to_hsv)
    from src.motion_model import (estimate_motion, apply_transform_to_points,
                                  compute_residual, mean_transform_magnitude,
                                  transform_to_motion)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w * 2, h))

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return output_path

    prev_gray = cv2.GaussianBlur(
        cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), blur_kernel, 0
    )

    # первый кадр — без анализа
    blank_hsv = np.zeros_like(prev_frame)
    writer.write(np.hstack([prev_frame, blank_hsv]))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), blur_kernel, 0
        )

        # поток
        flow = compute_farneback(prev_gray, curr_gray, farneback_params)
        pts1, pts2 = sample_flow_on_grid(flow, grid_step)

        # модель
        transform, inliers = estimate_motion(pts1, pts2, model=motion_model)

        # метрики
        flow_mag = mean_flow_magnitude(flow)
        model_mag = mean_transform_magnitude(transform, h, w, grid_step, motion_model)
        dx, dy, da = transform_to_motion(transform, model=motion_model)

        pts2_model = None
        residual = 0.0
        if transform is not None:
            pts2_model = apply_transform_to_points(pts1, transform, model=motion_model)
            residual = compute_residual(pts2, pts2_model)

        metrics = {
            "Farneback mean": flow_mag,
            "Model mean": model_mag,
            "Residual": residual,
            "dx": dx,
            "dy": dy,
            "da": da,
        }

        # левая часть: кадр со стрелками и метриками
        vis_left = draw_analysis_frame(frame, pts1, pts2, pts2_model,
                                       metrics, model=motion_model)

        # правая часть: HSV-карта потока
        vis_right = flow_to_hsv(flow)

        combined = np.hstack([vis_left, vis_right])
        writer.write(combined)

        prev_gray = curr_gray

    cap.release()
    writer.release()
    print(f"  Annotated side-by-side видео: {output_path}")
    return output_path

def plot_trajectory(trajectory, output_dir, prefix=""):
    """
    Три графика: x(t), y(t), angle(t) — raw vs smoothed.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    components = [
        ("X (horizontal)", trajectory.raw_x, trajectory.smooth_x),
        ("Y (vertical)", trajectory.raw_y, trajectory.smooth_y),
        ("Angle (rotation)", trajectory.raw_a, trajectory.smooth_a),
    ]

    for ax, (name, raw, smooth) in zip(axes, components):
        frames = range(len(raw))
        ax.plot(frames, raw, alpha=0.6, label="Raw", color="tab:blue")
        ax.plot(frames, smooth, label="Smoothed", color="tab:orange", linewidth=2)
        ax.set_ylabel(name)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frame")
    fig.suptitle("Camera trajectory: raw vs smoothed", fontsize=14)
    fig.tight_layout()

    path = os.path.join(output_dir, f"{prefix}trajectory.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Сохранено: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════
#  ГРАФИКИ ВЕЛИЧИНЫ ПОТОКА И МОДЕЛИ
# ═══════════════════════════════════════════════════════════════════

def plot_flow_and_model(flow_magnitudes, transform_magnitudes, output_dir, prefix=""):
    """График средней величины потока (Farnebäck) и глобальной модели."""
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(flow_magnitudes, label="Farnebäck flow", alpha=0.7, color="tab:green")
    ax.plot(transform_magnitudes, label="Global model", alpha=0.7, color="tab:red")

    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean magnitude (px)")
    ax.set_title("Optical flow magnitude vs global motion model")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, f"{prefix}flow_vs_model.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Сохранено: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════
#  ОШИБКА МОДЕЛИ (RESIDUAL)
# ═══════════════════════════════════════════════════════════════════

def plot_residuals(residuals, output_dir, prefix=""):
    """График ошибки (residual) глобальной модели по кадрам."""
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(residuals, color="tab:purple", alpha=0.7)
    ax.axhline(np.mean(residuals), color="tab:red", linestyle="--",
               label=f"Mean = {np.mean(residuals):.3f}")

    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean residual (px)")
    ax.set_title("Global model residual error per frame")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, f"{prefix}residuals.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Сохранено: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════
#  СРАВНЕНИЕ ТРЁХ ФИЛЬТРОВ
# ═══════════════════════════════════════════════════════════════════

def plot_smoother_comparison(raw_values, smoothed_dict, output_dir, title="X", prefix=""):
    """
    Сравнивает несколько сглаживателей на одном графике.

    Args:
        raw_values: список исходных значений
        smoothed_dict: {"kalman": [...], "moving_avg": [...], ...}
        output_dir: папка для сохранения
        title: заголовок (например, "X", "Y", "Angle")
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    frames = range(len(raw_values))
    ax.plot(frames, raw_values, alpha=0.4, label="Raw", color="gray", linewidth=1)

    colors = {"kalman": "tab:orange", "moving_avg": "tab:blue", "exponential": "tab:green"}

    for name, values in smoothed_dict.items():
        ax.plot(frames, values, label=name, color=colors.get(name, "black"), linewidth=1.5)

    ax.set_xlabel("Frame")
    ax.set_ylabel(title)
    ax.set_title(f"Smoother comparison: {title}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, f"{prefix}smoother_cmp_{title.lower()}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Сохранено: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════
#  КАДРЫ «ДО / ПОСЛЕ»
# ═══════════════════════════════════════════════════════════════════

def save_before_after(sample_frames, output_dir, prefix=""):
    """
    Сохраняет пары кадров (original, stabilized) рядом.

    Args:
        sample_frames: dict {frame_idx: (original_bgr, stabilized_bgr)}
    """
    paths = []
    for idx in sorted(sample_frames.keys()):
        orig, stab = sample_frames[idx]
        combined = np.hstack([orig, stab])
        # подписи
        h = orig.shape[0]
        cv2.putText(combined, "Original", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "Stabilized", (orig.shape[1] + 10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        path = os.path.join(output_dir, f"{prefix}before_after_{idx:04d}.png")
        cv2.imwrite(path, combined)
        paths.append(path)
        print(f"  Сохранено: {path}")
    return paths


# ═══════════════════════════════════════════════════════════════════
#  HSV-КАРТЫ ПОТОКА
# ═══════════════════════════════════════════════════════════════════

def save_flow_hsv(sample_flows, output_dir, prefix=""):
    """Сохраняет HSV-визуализации оптического потока."""
    paths = []
    for idx in sorted(sample_flows.keys()):
        hsv_img = sample_flows[idx]
        path = os.path.join(output_dir, f"{prefix}flow_hsv_{idx:04d}.png")
        cv2.imwrite(path, hsv_img)
        paths.append(path)
        print(f"  Сохранено: {path}")
    return paths


# ═══════════════════════════════════════════════════════════════════
#  SIDE-BY-SIDE ВИДЕО
# ═══════════════════════════════════════════════════════════════════

def create_side_by_side_video(input_path, stabilized_path, output_path):
    """
    Создаёт видео с оригиналом и стабилизированным кадром рядом.

    Args:
        input_path: путь к исходному видео
        stabilized_path: путь к стабилизированному видео
        output_path: путь для сохранения side-by-side видео
    """
    cap_orig = cv2.VideoCapture(input_path)
    cap_stab = cv2.VideoCapture(stabilized_path)

    fps = cap_orig.get(cv2.CAP_PROP_FPS)
    w = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w * 2, h))

    while True:
        ret1, frame1 = cap_orig.read()
        ret2, frame2 = cap_stab.read()
        if not ret1 or not ret2:
            break

        combined = np.hstack([frame1, frame2])
        cv2.putText(combined, "Original", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "Stabilized", (w + 10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        writer.write(combined)

    cap_orig.release()
    cap_stab.release()
    writer.release()

    print(f"  Side-by-side видео: {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════
#  ERROR ANALYSIS: ХУДШИЕ КАДРЫ
# ═══════════════════════════════════════════════════════════════════

def plot_worst_frames(residuals, output_dir, top_n=10, prefix=""):
    """
    Выделяет кадры с наибольшей ошибкой модели.

    Полезно для error analysis: понять, где стабилизация ломается.
    """
    residuals = np.array(residuals)
    worst_indices = np.argsort(residuals)[-top_n:][::-1]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(residuals, alpha=0.5, color="tab:purple")

    for idx in worst_indices:
        ax.axvline(idx, color="red", alpha=0.3, linewidth=0.8)

    ax.scatter(worst_indices, residuals[worst_indices],
               color="red", zorder=5, s=30, label=f"Top-{top_n} worst")

    ax.set_xlabel("Frame")
    ax.set_ylabel("Residual (px)")
    ax.set_title(f"Frames with highest model error (top {top_n})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, f"{prefix}worst_frames.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Сохранено: {path}")
    return path, worst_indices
