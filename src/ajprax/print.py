import builtins
import os


def print(*values, sep=" ", end=os.linesep, file=None, flush=False):
    """Only outwardly visible behavior change from builtins.print is that the content is printed atomically."""
    builtins.print(sep.join(str(value) for value in values) + end, end="", file=file, flush=flush)
