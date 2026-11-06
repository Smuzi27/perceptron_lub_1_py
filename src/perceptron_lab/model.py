"""Однослойный перцептрон Розенблатта для бинарной классификации.

Модель вычисляет взвешенную сумму признаков ``z = Σ w_i·x_i + bias``
и пропускает её через пороговую функцию активации, получая класс 0 или 1.
Обучение идёт по правилу Розенблатта: ``w += lr·(target − pred)·x``.
"""

from __future__ import annotations

from collections.abc import Sequence

Vector = Sequence[float]
Matrix = Sequence[Sequence[float]]


class Perceptron:
    """Линейный бинарный классификатор с пороговой активацией.

    Attributes:
        n_features: Количество входных признаков.
        learning_rate: Скорость обучения (шаг коррекции весов).
        epochs: Максимальное число полных проходов по обучающей выборке.
        weights: Веса признаков, длина равна ``n_features``.
        bias: Смещение (порог нейрона).
        errors_history: Число ошибочных предсказаний на каждой эпохе обучения.
    """

    def __init__(self, n_features: int, learning_rate: float = 0.1, epochs: int = 50) -> None:
        """Создаёт перцептрон с нулевыми весами и смещением.

        Args:
            n_features: Количество входных признаков (размерность вектора x).
            learning_rate: Скорость обучения, должна быть больше нуля.
            epochs: Максимальное число эпох, должно быть не меньше 1.

        Raises:
            ValueError: Если какой-либо параметр вне допустимого диапазона.
        """
        if n_features < 1:
            raise ValueError(f"n_features должно быть >= 1, получено {n_features}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate должна быть > 0, получено {learning_rate}")
        if epochs < 1:
            raise ValueError(f"epochs должно быть >= 1, получено {epochs}")

        self.n_features = n_features
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights: list[float] = [0.0] * n_features
        self.bias = 0.0
        self.errors_history: list[int] = []

    def activate(self, z: float) -> int:
        """Пороговая функция активации (функция Хевисайда).

        Args:
            z: Взвешенная сумма входов и смещения.

        Returns:
            1, если ``z >= 0``, иначе 0.
        """
        return 1 if z >= 0 else 0

    def net_input(self, x: Vector) -> float:
        """Вычисляет взвешенную сумму ``Σ w_i·x_i + bias`` для одного объекта.

        Args:
            x: Вектор признаков длины ``n_features``.

        Returns:
            Значение взвешенной суммы.

        Raises:
            ValueError: Если длина вектора не совпадает с ``n_features``.
        """
        if len(x) != self.n_features:
            raise ValueError(f"Ожидался вектор из {self.n_features} признаков, получено {len(x)}")
        return sum(w * xi for w, xi in zip(self.weights, x, strict=True)) + self.bias

    def predict(self, x: Vector) -> int:
        """Предсказывает класс одного объекта (прямой проход).

        Args:
            x: Вектор признаков длины ``n_features``.

        Returns:
            Предсказанный класс: 0 или 1.
        """
        return self.activate(self.net_input(x))

    def train(self, X: Matrix, y: Sequence[int]) -> list[int]:
        """Обучает перцептрон по правилу Розенблатта.

        Перед обучением веса и смещение сбрасываются в ноль, поэтому повторный
        вызов даёт тот же результат на тех же данных. Обучение останавливается
        досрочно, если за эпоху не было ни одной ошибки.

        Args:
            X: Обучающие объекты, каждый длины ``n_features``.
            y: Метки классов (0 или 1) для каждого объекта.

        Returns:
            Число ошибок на каждой эпохе (то же, что ``errors_history``).
        """
        self._check_dataset(X, y)
        self.weights = [0.0] * self.n_features
        self.bias = 0.0
        self.errors_history = []

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y, strict=True):
                update = self.learning_rate * (target - self.predict(xi))
                if update != 0.0:
                    self.weights = [w + update * v for w, v in zip(self.weights, xi, strict=True)]
                    self.bias += update
                    errors += 1
            self.errors_history.append(errors)
            if errors == 0:
                break
        return self.errors_history

    def score(self, X: Matrix, y: Sequence[int]) -> float:
        """Вычисляет долю правильных ответов (accuracy).

        Args:
            X: Объекты для проверки.
            y: Истинные метки классов.

        Returns:
            Доля верных предсказаний от 0.0 до 1.0.
        """
        self._check_dataset(X, y)
        correct = sum(1 for xi, target in zip(X, y, strict=True) if self.predict(xi) == target)
        return correct / len(y)

    def _check_dataset(self, X: Matrix, y: Sequence[int]) -> None:
        """Проверяет согласованность выборки и меток.

        Raises:
            ValueError: Если выборка пуста, размеры не совпадают
                или метки не из множества {0, 1}.
        """
        if len(X) == 0:
            raise ValueError("Выборка пуста")
        if len(X) != len(y):
            raise ValueError(f"Число объектов ({len(X)}) не совпадает с числом меток ({len(y)})")
        bad_labels = set(y) - {0, 1}
        if bad_labels:
            raise ValueError(f"Метки должны быть 0 или 1, найдены: {sorted(bad_labels)}")
