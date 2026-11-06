"""Интерфейс командной строки: генерация данных, обучение, вывод точности."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from perceptron_lab.data import generate_synthetic_data, load_iris_binary
from perceptron_lab.model import Perceptron


def positive_int(value: str) -> int:
    """Преобразует аргумент в целое число больше нуля.

    Raises:
        argparse.ArgumentTypeError: Если значение не целое или не положительное.
    """
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"ожидалось целое число, получено '{value}'") from None
    if number <= 0:
        raise argparse.ArgumentTypeError(f"значение должно быть больше 0, получено {number}")
    return number


def positive_float(value: str) -> float:
    """Преобразует аргумент в вещественное число больше нуля.

    Raises:
        argparse.ArgumentTypeError: Если значение не число или не положительное.
    """
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"ожидалось число, получено '{value}'") from None
    if number <= 0:
        raise argparse.ArgumentTypeError(f"значение должно быть больше 0, получено {number}")
    return number


def build_parser() -> argparse.ArgumentParser:
    """Создаёт парсер аргументов командной строки."""
    parser = argparse.ArgumentParser(
        prog="perceptron",
        description="Обучение перцептрона Розенблатта на синтетических данных или Iris.",
    )
    parser.add_argument(
        "--samples", type=positive_int, default=200, help="число точек (по умолчанию 200)"
    )
    parser.add_argument(
        "--lr", type=positive_float, default=0.1, help="скорость обучения (по умолчанию 0.1)"
    )
    parser.add_argument(
        "--epochs", type=positive_int, default=50, help="максимум эпох (по умолчанию 50)"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="зерно генератора для воспроизводимости"
    )
    parser.add_argument(
        "--dataset",
        choices=("synthetic", "iris"),
        default="synthetic",
        help="набор данных: synthetic (по умолчанию) или iris",
    )
    parser.add_argument("--save", type=Path, default=None, help="сохранить графики в PNG-файл")
    parser.add_argument("--no-plot", action="store_true", help="не строить графики")
    return parser


def run(args: argparse.Namespace) -> float:
    """Загружает данные, обучает модель, печатает результат и строит графики.

    Args:
        args: Разобранные аргументы командной строки.

    Returns:
        Итоговая точность модели.
    """
    if args.dataset == "iris":
        X, y, feature_names = load_iris_binary(seed=args.seed if args.seed is not None else 0)
        title = "Iris: setosa vs versicolor"
        labels = (f"{feature_names[0]} (станд.)", f"{feature_names[1]} (станд.)")
    else:
        if args.samples < 2:
            raise ValueError("для синтетических данных нужно --samples не меньше 2")
        X, y = generate_synthetic_data(samples=args.samples, seed=args.seed)
        title = "Синтетические данные 2D"
        labels = ("x1", "x2")

    model = Perceptron(n_features=2, learning_rate=args.lr, epochs=args.epochs)
    history = model.train(X, y)
    accuracy = model.score(X, y)

    print(f"Набор данных: {args.dataset}, объектов: {len(X)}")
    print(f"Параметры: lr={args.lr}, максимум эпох={args.epochs}, пройдено эпох={len(history)}")
    print(f"Веса: {[round(w, 4) for w in model.weights]}, bias: {model.bias:.4f}")
    print(f"Итоговая точность: {accuracy:.4f}")

    if not args.no_plot:
        from perceptron_lab.plotting import plot_training_results

        plot_training_results(model, X, y, title, labels, save_path=args.save)
        if args.save is not None:
            print(f"Графики сохранены в {args.save}")
    return accuracy


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа CLI.

    Args:
        argv: Аргументы командной строки (по умолчанию берутся из ``sys.argv``).

    Returns:
        Код завершения: 0 при успехе, 1 при ошибке.
    """
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (ValueError, OSError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    return 0
