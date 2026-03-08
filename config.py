"""
Единый конфиг проекта.
Все параметры собраны здесь, чтобы не хардкодить их в модулях.
"""

import os

# ───────────────────────── пути ─────────────────────────
INPUT_VIDEO = "input/abriged.mp4"
OUTPUT_DIR = "output"

# создаём выходную папку при импорте конфига
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ───────────────────────── Farnebäck ────────────────────
FARNEBACK_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)

# ───────────────────────── сетка точек ──────────────────
GRID_STEP = 15          # шаг сетки для сэмплирования потока

# ───────────────────────── модель движения ──────────────
# "affine"  → estimateAffine2D  (6 параметров)
# "homography" → findHomography   (8 параметров)
MOTION_MODEL = "affine"

# ───────────────────────── сглаживание ──────────────────
# "kalman" | "moving_avg" | "exponential"
SMOOTHER = "kalman"

# Kalman
KALMAN_PROCESS_NOISE = 1e-4
KALMAN_MEASUREMENT_NOISE = 1e-1

# Moving average
MA_WINDOW = 30

# Exponential smoothing
EXP_ALPHA = 0.05   # маленький α → сильное сглаживание

# ───────────────────────── стабилизация ─────────────────
CROP_SCALE = 1.04   # масштаб для обрезки чёрных краёв

# ───────────────────────── визуализация ─────────────────
BLUR_KERNEL = (5, 5)            # размытие перед потоком
ARROW_SCALE = 1                 # толщина стрелок
BEFORE_AFTER_FRAMES = [0, 50, 100, 200]  # кадры для сравнения до/после
