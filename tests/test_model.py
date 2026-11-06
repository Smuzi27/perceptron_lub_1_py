"""Тесты класса Perceptron."""

import pytest

from perceptron_lab.data import generate_synthetic_data
from perceptron_lab.model import Perceptron


@pytest.mark.parametrize("n_features", [1, 2, 5])
def test_weights_have_expected_length(n_features: int) -> None:
    """Веса инициализируются нулями, их число равно числу признаков."""
    model = Perceptron(n_features=n_features)
    assert len(model.weights) == n_features
    assert model.weights == [0.0] * n_features
    assert model.bias == 0.0


@pytest.mark.parametrize(("z", "expected"), [(0.0, 1), (2.5, 1), (-1.0, 0), (-1e-9, 0)])
def test_activate_threshold(z: float, expected: int) -> None:
    """Пороговая функция: 1 при z >= 0, иначе 0."""
    assert Perceptron(n_features=1).activate(z) == expected


def test_predict_with_known_weights() -> None:
    """При заданных весах predict правильно разделяет точки."""
    model = Perceptron(n_features=2)
    model.weights = [1.0, 1.0]
    model.bias = -1.5
    assert model.predict([1.0, 1.0]) == 1
    assert model.predict([0.0, 1.0]) == 0
    assert model.predict([2.0, 0.0]) == 1


def test_training_separates_linearly_separable_classes() -> None:
    """После обучения на разделимых данных точность равна 1.0."""
    X, y = generate_synthetic_data(samples=200, seed=42)
    model = Perceptron(n_features=2, learning_rate=0.1, epochs=50)
    model.train(X, y)
    assert model.score(X, y) == 1.0


def test_errors_history_ends_with_zero() -> None:
    """История ошибок заполняется и на последней эпохе ошибок нет."""
    X, y = generate_synthetic_data(samples=100, seed=1)
    model = Perceptron(n_features=2, epochs=100)
    history = model.train(X, y)
    assert history == model.errors_history
    assert len(history) >= 1
    assert history[-1] == 0


def test_learning_rate_does_not_change_predictions() -> None:
    """При нулевых начальных весах lr лишь масштабирует веса, предсказания те же."""
    X, y = generate_synthetic_data(samples=100, seed=7)
    slow = Perceptron(n_features=2, learning_rate=0.01, epochs=50)
    fast = Perceptron(n_features=2, learning_rate=1.0, epochs=50)
    slow.train(X, y)
    fast.train(X, y)
    assert slow.errors_history == fast.errors_history
    assert [slow.predict(x) for x in X] == [fast.predict(x) for x in X]


def test_predict_rejects_wrong_vector_length() -> None:
    """Вектор неправильной длины вызывает понятную ошибку."""
    with pytest.raises(ValueError, match="2 признаков"):
        Perceptron(n_features=2).predict([1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    ("X", "y"),
    [([], []), ([[1.0, 2.0]], [0, 1]), ([[1.0, 2.0]], [2])],
)
def test_score_rejects_invalid_dataset(X: list[list[float]], y: list[int]) -> None:
    """Пустая выборка, разные размеры и неверные метки отклоняются."""
    with pytest.raises(ValueError):
        Perceptron(n_features=2).score(X, y)


@pytest.mark.parametrize(
    ("n_features", "learning_rate", "epochs"),
    [(0, 0.1, 10), (2, 0.0, 10), (2, -0.1, 10), (2, 0.1, 0)],
)
def test_init_rejects_invalid_params(n_features: int, learning_rate: float, epochs: int) -> None:
    """Недопустимые гиперпараметры отклоняются при создании модели."""
    with pytest.raises(ValueError):
        Perceptron(n_features=n_features, learning_rate=learning_rate, epochs=epochs)
