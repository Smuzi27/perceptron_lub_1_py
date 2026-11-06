"""Тесты интерфейса командной строки."""

from pathlib import Path

import pytest

from perceptron_lab.cli import main


def test_cli_synthetic_prints_accuracy(capsys: pytest.CaptureFixture[str]) -> None:
    """Команда из задания печатает итоговую точность 1.0."""
    code = main(["--samples", "200", "--lr", "0.1", "--epochs", "50", "--seed", "42", "--no-plot"])
    assert code == 0
    assert "Итоговая точность: 1.0000" in capsys.readouterr().out


def test_cli_iris(capsys: pytest.CaptureFixture[str]) -> None:
    """Запуск на Iris завершается успешно."""
    assert main(["--dataset", "iris", "--epochs", "100", "--no-plot"]) == 0
    assert "iris" in capsys.readouterr().out


def test_cli_saves_plot(tmp_path: Path) -> None:
    """Флаг --save сохраняет PNG с графиками."""
    target = tmp_path / "plot.png"
    assert main(["--samples", "60", "--seed", "1", "--save", str(target)]) == 0
    assert target.exists()
    assert target.stat().st_size > 0


@pytest.mark.parametrize(
    "argv", [["--samples", "-5"], ["--lr", "0"], ["--epochs", "abc"], ["--dataset", "mnist"]]
)
def test_cli_rejects_invalid_arguments(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    """Неверные аргументы дают понятное сообщение и код 2, без трейсбека."""
    with pytest.raises(SystemExit) as exc_info:
        main(argv)
    assert exc_info.value.code == 2
    assert "error" in capsys.readouterr().err


def test_cli_reports_runtime_error(capsys: pytest.CaptureFixture[str]) -> None:
    """Ошибка данных выводится сообщением и кодом 1."""
    assert main(["--samples", "1", "--no-plot"]) == 1
    assert "Ошибка" in capsys.readouterr().err
