"""Наборы данных для перцептрона: синтетический 2D и бинарный Iris."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from sklearn.datasets import load_iris

Point = tuple[float, float]

CLASS0_CENTER: Point = (1.0, 1.0)
CLASS1_CENTER: Point = (4.0, 4.0)
SPREAD = 0.5


def generate_synthetic_data(
    samples: int = 200,
    seed: int | None = None,
    centers: tuple[Point, Point] = (CLASS0_CENTER, CLASS1_CENTER),
    spread: float = SPREAD,
) -> tuple[list[list[float]], list[int]]:

    if samples < 2:
        raise ValueError(f"samples должно быть >= 2, получено {samples}")
    if spread <= 0:
        raise ValueError(f"spread должно быть > 0, получено {spread}")

    rng = random.Random(seed)
    n_class0 = samples // 2
    n_class1 = samples - n_class0

    points: list[tuple[list[float], int]] = []
    for label, count in ((0, n_class0), (1, n_class1)):
        cx, cy = centers[label]
        for _ in range(count):
            points.append(([rng.gauss(cx, spread), rng.gauss(cy, spread)], label))

    rng.shuffle(points)
    X = [point for point, _ in points]
    y = [label for _, label in points]
    return X, y


def standardize(X: Sequence[Sequence[float]]) -> list[list[float]]:

    n_rows = len(X)
    n_cols = len(X[0])
    means = [sum(row[j] for row in X) / n_rows for j in range(n_cols)]
    stds = [
        math.sqrt(sum((row[j] - means[j]) ** 2 for row in X) / n_rows) or 1.0 for j in range(n_cols)
    ]
    return [[(row[j] - means[j]) / stds[j] for j in range(n_cols)] for row in X]


def load_iris_binary(
    seed: int | None = 0, scale: bool = True
) -> tuple[list[list[float]], list[int], list[str]]:

    iris = load_iris()
    rows: list[tuple[list[float], int]] = [
        ([float(features[0]), float(features[1])], int(target))
        for features, target in zip(iris.data, iris.target, strict=True)
        if int(target) < 2
    ]
    random.Random(seed).shuffle(rows)

    X = [features for features, _ in rows]
    y = [target for _, target in rows]
    if scale:
        X = standardize(X)
    feature_names: list[str] = [str(iris.feature_names[0]), str(iris.feature_names[1])]
    return X, y, feature_names
