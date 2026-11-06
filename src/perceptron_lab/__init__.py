"""Перцептрон Розенблатта: модель, данные, визуализация и CLI."""

from perceptron_lab.data import generate_synthetic_data, load_iris_binary
from perceptron_lab.model import Perceptron

__all__ = ["Perceptron", "generate_synthetic_data", "load_iris_binary"]
