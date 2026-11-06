"""Визуализация: граница решения перцептрона и график ошибок по эпохам."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from perceptron_lab.model import Perceptron

CLASS_COLORS = ("#d1495b", "#2e86ab")
EPS = 1e-12


def plot_decision_boundary(
    ax: Axes,
    model: Perceptron,
    X: Sequence[Sequence[float]],
    y: Sequence[int],
    title: str,
    feature_labels: tuple[str, str] = ("x1", "x2"),
) -> None:
    """Рисует точки двух классов, области решений и разделяющую прямую.

    Прямая задаётся уравнением ``w1·x1 + w2·x2 + bias = 0``.

    Args:
        ax: Оси matplotlib, на которых рисовать.
        model: Обученный перцептрон с двумя признаками.
        X: Точки ``[x1, x2]``.
        y: Метки классов.
        title: Заголовок графика.
        feature_labels: Подписи осей.
    """
    xs = [row[0] for row in X]
    ys = [row[1] for row in X]
    x_min, x_max = min(xs) - 0.5, max(xs) + 0.5
    y_min, y_max = min(ys) - 0.5, max(ys) + 0.5

    w1, w2 = model.weights[0], model.weights[1]
    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    regions = (w1 * grid_x + w2 * grid_y + model.bias >= 0).astype(float)
    ax.contourf(
        grid_x,
        grid_y,
        regions,
        levels=[-0.5, 0.5, 1.5],
        cmap=ListedColormap(CLASS_COLORS),
        alpha=0.12,
    )

    for label, color in enumerate(CLASS_COLORS):
        cls_x = [row[0] for row, target in zip(X, y, strict=True) if target == label]
        cls_y = [row[1] for row, target in zip(X, y, strict=True) if target == label]
        ax.scatter(cls_x, cls_y, color=color, edgecolors="white", s=40, label=f"Класс {label}")

    if abs(w2) > EPS:
        line_x = [x_min, x_max]
        line_y = [-(w1 * v + model.bias) / w2 for v in line_x]
        ax.plot(line_x, line_y, color="black", linestyle="--", label="Граница решения")
    elif abs(w1) > EPS:
        ax.axvline(-model.bias / w1, color="black", linestyle="--", label="Граница решения")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"{title}\nТочность: {model.score(X, y):.2%}")
    ax.set_xlabel(feature_labels[0])
    ax.set_ylabel(feature_labels[1])
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", alpha=0.5)


def plot_errors(ax: Axes, errors_history: Sequence[int]) -> None:
    """Рисует число ошибок на каждой эпохе обучения.

    Args:
        ax: Оси matplotlib, на которых рисовать.
        errors_history: Число ошибок по эпохам.
    """
    epochs = list(range(1, len(errors_history) + 1))
    ax.plot(epochs, list(errors_history), marker="o", color="#e07a1f")
    ax.set_title("Ошибки по эпохам")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Число ошибок")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle=":", alpha=0.5)


def plot_training_results(
    model: Perceptron,
    X: Sequence[Sequence[float]],
    y: Sequence[int],
    title: str,
    feature_labels: tuple[str, str] = ("x1", "x2"),
    save_path: Path | None = None,
) -> Figure:
    """Строит рисунок из двух графиков: граница решения и ошибки по эпохам.

    Если указан ``save_path``, рисунок сохраняется в файл, иначе открывается окно.

    Args:
        model: Обученный перцептрон с двумя признаками.
        X: Точки ``[x1, x2]``.
        y: Метки классов.
        title: Заголовок левого графика.
        feature_labels: Подписи осей левого графика.
        save_path: Путь для сохранения PNG.

    Returns:
        Построенный рисунок matplotlib.
    """
    fig, (ax_boundary, ax_errors) = plt.subplots(1, 2, figsize=(12, 5))
    plot_decision_boundary(ax_boundary, model, X, y, title, feature_labels)
    plot_errors(ax_errors, model.errors_history)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
    else:
        plt.show()
    return fig
