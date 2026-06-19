import subprocess
import sys


PYREFLY_BASE_COMMAND = [
    sys.executable,
    "-m",
    "pyrefly",
    "check",
    "--project-excludes",
    "**/site-packages/**",
    "--python-version",
    "3.10",
    "--progress-bar",
    "no",
    "--summary=none",
    "--search-path",
    "src",
]


def _run_pyrefly(*args: str) -> None:
    result = subprocess.run(
        [*PYREFLY_BASE_COMMAND, *args],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_pyrefly_basic() -> None:
    _run_pyrefly("--preset", "basic", "src/ajprax")


def test_pyrefly_strict() -> None:
    _run_pyrefly(
        "--preset",
        "strict",
        "--project-excludes",
        "src/ajprax/experimental/**",
        "src/ajprax",
    )
