import builtins
import os
from typing import Optional, TextIO


def print(
    *values: object,
    sep: str = " ",
    end: str = os.linesep,
    file: Optional[TextIO] = None,
    flush: bool = False,
) -> None:
    """Only outwardly visible behavior change from builtins.print is that the content is printed atomically."""
    builtins.print(sep.join(str(value) for value in values) + end, end="", file=file, flush=flush)
