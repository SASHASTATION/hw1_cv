#!/usr/bin/env python3
"""
Точка входа: стабилизация видео + визуализации + сравнение фильтров.

Использование:
    python run.py
    python run.py --input input/shaky.mp4 --model affine --smoother kalman
    python run.py --input input/shaky.mp4 --model homography --smoother moving_avg
"""

import argparse
import os

import config
from src.stabilizer import stabilize_video
from src.smoother_comparison import compare_smoothers
from src.visualization import (
    plot_trajectory,
    plot_flow_and_model,
    plot_residuals,
    plot_worst_frames,
    plot_smoother_comparison,
    save_before_after,
    save_flow_hsv,
    create_side_by_side_video,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Video Stabilization Pipeline")

    parser.add_argument("--input", type=str, default=config.INPUT_VIDEO,
                        help="Путь к входному видео")
    parser.add_argument("--output-dir", type=str, default=config.OUTPUT_DIR,
                        help="Папка для результатов")
    parser.add_argument("--model", type=str, default=config.MOTION_MODEL,
                        choices=["affine", "homography"],
                        help="Модель движения: affine или homography")
    parser.add_argument("--smoother", type=str, default=config.SMOOTHER,
                        choices=["kalman", "moving_avg", "exponential"],
                        help="Метод сглаживания траектории")
    parser.add_argument("--crop-scale", type=float, default=config.CROP_SCALE,
                        help="Масштаб обрезки чёрных краёв")
    parser.add_argument("--grid-step", type=int, default=config.GRID_STEP,
                        help="Шаг сетки для сэмплирования потока")
    parser.add_argument("--save-frames", type=int, nargs="*",
                        default=config.BEFORE_AFTER_FRAMES,
                        help="Индексы кадров для сохранения до/после")

    return parser.parse_args()


def get_smoother_kwargs(smoother_name):
    """Возвращает параметры для выбранного сглаживателя из конфига."""
    if smoother_name == "kalman":
        return {
            "process_noise": config.KALMAN_PROCESS_NOISE,
            "measurement_noise": config.KALMAN_MEASUREMENT_NOISE,
        }
    elif smoother_name == "moving_avg":
        return {"window": config.MA_WINDOW}
    elif smoother_name == "exponential":
        return {"alpha": config.EXP_ALPHA}
    return {}


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── имя для файлов на основе параметров ──
    prefix = f"{args.model}_{args.smoother}_"
    stabilized_path = os.path.join(args.output_dir, f"{prefix}stabilized.mp4")

    # ══════════════════════════════════════════════════════════════
    #  1. СТАБИЛИЗАЦИЯ
    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("ЭТАП 1: Стабилизация видео")
    print("=" * 60)

    smoother_kwargs = get_smoother_kwargs(args.smoother)

    result = stabilize_video(
        input_path=args.input,
        output_path=stabilized_path,
        farneback_params=config.FARNEBACK_PARAMS,
        grid_step=args.grid_step,
        motion_model=args.model,
        smoother_name=args.smoother,
        smoother_kwargs=smoother_kwargs,
        crop_scale=args.crop_scale,
        blur_kernel=config.BLUR_KERNEL,
        save_frames_at=args.save_frames,
    )

    # ══════════════════════════════════════════════════════════════
    #  2. ВИЗУАЛИЗАЦИИ
    # ══════════════════════════════════════════════════════════════
    print()
    print("=" * 60)
    print("ЭТАП 2: Построение визуализаций")
    print("=" * 60)

    trajectory = result.trajectory

    # 2.1 Траектория: raw vs smoothed
    plot_trajectory(trajectory, args.output_dir, prefix=prefix)

    # 2.2 Величина потока vs модели
    plot_flow_and_model(
        result.flow_magnitudes,
        result.transform_magnitudes,
        args.output_dir, prefix=prefix,
    )

    # 2.3 Ошибка модели (residual)
    plot_residuals(result.residuals, args.output_dir, prefix=prefix)

    # 2.4 Худшие кадры
    plot_worst_frames(result.residuals, args.output_dir, top_n=10, prefix=prefix)

    # 2.5 Кадры до/после
    if result.sample_frames:
        save_before_after(result.sample_frames, args.output_dir, prefix=prefix)

    # 2.6 HSV-карты потока
    if result.sample_flows:
        save_flow_hsv(result.sample_flows, args.output_dir, prefix=prefix)

    # 2.7 Side-by-side видео
    sbs_path = os.path.join(args.output_dir, f"{prefix}side_by_side.mp4")
    create_side_by_side_video(args.input, stabilized_path, sbs_path)

    # 2.8 Annotated side-by-side (стрелки + метрики + HSV)
    from src.visualization import create_annotated_side_by_side_video
    annotated_path = os.path.join(args.output_dir, f"{prefix}annotated.mp4")
    create_annotated_side_by_side_video(
        args.input, annotated_path,
        farneback_params=config.FARNEBACK_PARAMS,
        grid_step=args.grid_step,
        motion_model=args.model,
        blur_kernel=config.BLUR_KERNEL,
    )

    # ══════════════════════════════════════════════════════════════
    #  3. СРАВНЕНИЕ СГЛАЖИВАТЕЛЕЙ
    # ══════════════════════════════════════════════════════════════
    print()
    print("=" * 60)
    print("ЭТАП 3: Сравнение сглаживателей")
    print("=" * 60)

    comparison = compare_smoothers(
        trajectory.deltas_x,
        trajectory.deltas_y,
        trajectory.deltas_a,
    )

    raw = comparison.pop("_raw")

    for component, label in [("x", "X"), ("y", "Y"), ("a", "Angle")]:
        smoothed_dict = {name: data[component] for name, data in comparison.items()}
        raw_values = raw[component]
        plot_smoother_comparison(raw_values, smoothed_dict, args.output_dir,
                                 title=label, prefix=prefix)

    # ══════════════════════════════════════════════════════════════
    #  ИТОГО
    # ══════════════════════════════════════════════════════════════
    print()
    print("=" * 60)
    print("ГОТОВО!")
    print("=" * 60)
    print(f"  Стабилизированное видео : {stabilized_path}")
    print(f"  Side-by-side видео      : {sbs_path}")
    print(f"  Annotated видео          : {annotated_path}")
    print(f"  Графики и кадры         : {args.output_dir}/")
    print()
    print("Для сравнения моделей запустите повторно с другими параметрами:")
    print(f"  python run.py --input {args.input} --model homography --smoother exponential")


if __name__ == "__main__":
    main()
