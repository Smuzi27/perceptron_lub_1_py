"""Тесты генерации и загрузки данных."""

import pytest

from perceptron_lab.data import generate_synthetic_data, load_iris_binary, standardize
from perceptron_lab.model import Perceptron


@pytest.mark.parametrize("samples", [2, 101, 200])
def test_synthetic_data_shape_and_balance(samples: int) -> None:
    """Точек ровно samples, каждая двумерная, классы сбалансированы."""
    X, y = generate_synthetic_data(samples=samples, seed=0)
    assert len(X) == len(y) == samples
    assert all(len(point) == 2 for point in X)
    assert set(y) == {0, 1}
    assert abs(y.count(0) - y.count(1)) <= 1


def test_synthetic_data_is_reproducible() -> None:
    """С одинаковым seed данные совпадают, с разным — нет."""
    assert generate_synthetic_data(samples=50, seed=3) == generate_synthetic_data(50, seed=3)
    assert generate_synthetic_data(samples=50, seed=3) != generate_synthetic_data(50, seed=4)


def test_synthetic_data_rejects_too_few_samples() -> None:
    """Меньше двух точек сгенерировать нельзя."""
    with pytest.raises(ValueError):
        generate_synthetic_data(samples=1)


def test_standardize_gives_zero_mean_unit_std() -> None:
    """После стандартизации среднее 0, отклонение 1."""
    scaled = standardize([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    for j in range(2):
        column = [row[j] for row in scaled]
        mean = sum(column) / len(column)
        std = (sum((v - mean) ** 2 for v in column) / len(column)) ** 0.5
        assert mean == pytest.approx(0.0)
        assert std == pytest.approx(1.0)


def test_iris_binary_contains_two_classes() -> None:
    """В бинарном Iris 100 объектов, 2 признака, метки 0 и 1 по 50 штук."""
    X, y, names = load_iris_binary()
    assert len(X) == len(y) == 100
    assert all(len(row) == 2 for row in X)
    assert y.count(0) == y.count(1) == 50
    assert len(names) == 2


def test_perceptron_on_real_iris_data() -> None:
    """На реальных данных Iris (setosa vs versicolor) точность не ниже 95%."""
    X, y, _ = load_iris_binary(seed=0)
    model = Perceptron(n_features=2, learning_rate=0.1, epochs=100)
    model.train(X, y)
    assert model.score(X, y) >= 0.95
