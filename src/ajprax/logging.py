from __future__ import annotations

import json
import os
import sys
import traceback
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from os import PathLike
from threading import Lock, Event
from typing import BinaryIO, Iterator, Optional, Union

from ajprax.experimental.subscriptions import Events
from ajprax.print import print
from ajprax.require import require

TRACE = 0
DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
FATAL = 50

LEVEL_NAME = {
    TRACE: "TRACE",
    DEBUG: "DEBUG",
    INFO: " INFO",
    WARN: " WARN",
    ERROR: "ERROR",
    FATAL: "FATAL",
}

LEVEL_COLOR = {
    TRACE: "\033[90m",
    DEBUG: "\033[35m",
    INFO: "\033[36m",
    WARN: "\033[33m",
    ERROR: "\033[31m",
    FATAL: "\033[1;31m",
}
RESET = "\033[0m"

LogException = Union[BaseException, bool, None]
Path = Union[str, PathLike[str]]


@dataclass
class Log:
    datetime: datetime
    level: int
    message: str
    keywords: dict[object, object]
    exception: LogException
    logger_name: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.exception, bool):
            if self.exception:
                self.exception = sys.exc_info()[1]
            else:
                self.exception = None

    def parts(self, color: bool = False) -> Iterator[str]:
        yield self.datetime.isoformat().replace("+00:00", "Z")
        if color:
            yield LEVEL_COLOR[self.level] + LEVEL_NAME[self.level] + RESET
        else:
            yield LEVEL_NAME[self.level]
        if self.logger_name is not None:
            yield f"[{self.logger_name}]"
        if self.message:
            yield self.message
        for k, v in self.keywords.items():
            yield f"{k}={repr(v)}"
        if isinstance(self.exception, BaseException):
            yield "\n" + "".join("\t" + line for line in traceback.format_exception(self.exception))

    def __str__(self) -> str:
        return " ".join(self.parts())

    @property
    def color(self) -> str:
        return " ".join(self.parts(True))

    @property
    def json(self) -> str:
        d: dict[str, object] = dict(
            time=self.datetime.isoformat().replace("+00:00", "Z"),
            level=LEVEL_NAME[self.level],
        )
        if self.logger_name:
            d["logger"] = self.logger_name
        if self.message:
            d["message"] = str(self.message)
        if self.keywords:
            d["keywords"] = {str(k): repr(v) for k, v in self.keywords.items()}
        if isinstance(self.exception, BaseException):
            d["exception"] = "".join("\t" + line for line in traceback.format_exception(self.exception))
        return json.dumps(d)


class Logger(Events):
    def __init__(self, name: Optional[str] = None) -> None:
        Events.__init__(self)
        self.name = name
        self._global_level = INFO
        self._context_level: ContextVar[Optional[int]] = ContextVar("level", default=None)
        self._kwargs: ContextVar[dict[str, object]] = ContextVar("kwargs", default={})
        self.subscribe(lambda line: print(line.color))

    @property
    def level(self) -> int:
        context_level = self._context_level.get()
        if context_level is None:
            return self._global_level
        return context_level

    @level.setter
    def level(self, level: int) -> None:
        self._global_level = level

    def _log(self, _level: int, _message: str, _exception: LogException, **kwargs: object) -> None:
        if _level >= self.level:
            self.send(Log(
                datetime.now(timezone.utc),
                _level,
                _message,
                {**self._kwargs.get(), **kwargs},
                _exception,
                self.name,
            ))

    def trace(self, _message: str = "", _exception: LogException = False, **kwargs: object) -> None:
        self._log(TRACE, _message, _exception, **kwargs)

    def debug(self, _message: str = "", _exception: LogException = False, **kwargs: object) -> None:
        self._log(DEBUG, _message, _exception, **kwargs)

    def info(self, _message: str = "", _exception: LogException = False, **kwargs: object) -> None:
        self._log(INFO, _message, _exception, **kwargs)

    def warn(self, _message: str = "", _exception: LogException = False, **kwargs: object) -> None:
        self._log(WARN, _message, _exception, **kwargs)

    def error(self, _message: str = "", _exception: LogException = False, **kwargs: object) -> None:
        self._log(ERROR, _message, _exception, **kwargs)

    def fatal(self, _message: str = "", _exception: LogException = False, **kwargs: object) -> None:
        self._log(FATAL, _message, _exception, **kwargs)

    @contextmanager
    def context_level(self, level: int) -> Iterator[None]:
        token = self._context_level.set(level)
        try:
            yield
        finally:
            self._context_level.reset(token)

    @contextmanager
    def context_kwargs(self, **kw: object) -> Iterator[None]:
        token = self._kwargs.set(kw)
        try:
            yield
        finally:
            self._kwargs.reset(token)


log = Logger()


class FlushOptimizingWriter:
    def __init__(self, file: BinaryIO) -> None:
        self.file = file
        self.write_lock = Lock()
        self.buffer_lock = Lock()
        self.buffer: list[tuple[bytes, Event]] = []

    def write(self, data: bytes) -> None:
        with self.buffer_lock:
            writer = not self.buffer
            event = Event()
            self.buffer.append((data, event))

        if writer:
            with self.write_lock:
                # swapping the buffer must happen under the write lock to guarantee that a second writer can't get the
                # new buffer and the write lock before this one, which would violate ordering expectations
                with self.buffer_lock:
                    empty: list[tuple[bytes, Event]] = []
                    to_write, self.buffer = self.buffer, empty
                for data, _ in to_write:
                    self.file.write(data)
                self.file.flush()
                for _, event in to_write:
                    event.set()
        else:
            event.wait()


class RolledLog:
    """
    Rolls log files when max_bytes are exceeded or max_duration has elapsed.

    If a single line exceeds max_bytes, it will be given a file by itself.
    Start the timer when the first log arrives after max_duration has elapsed, not when the previous window would have
    expired.
    """

    def __init__(
        self,
        directory: Path,
        *,
        max_bytes: Optional[int] = None,
        max_duration: Optional[timedelta] = None,
        file_pattern: str = "{}.log",
    ) -> None:
        require(
            max_bytes is not None or max_duration is not None,
            "must specify at least one of max_bytes and max_duration",
        )

        self.directory = directory
        self.max_bytes = max_bytes
        self.max_duration = max_duration
        self.file_pattern = file_pattern

        self.lock = Lock()
        self.writer: Optional[FlushOptimizingWriter] = None
        self.used_bytes = 0
        self.start_time: Optional[datetime] = None

    def write(self, data: bytes) -> None:
        with self.lock:
            now = datetime.now(timezone.utc)
            self.used_bytes += len(data)
            if self.writer is None:
                os.makedirs(self.directory, exist_ok=True)

            expired_by_duration = (
                self.max_duration is not None
                and self.start_time is not None
                and now - self.start_time > self.max_duration
            )
            if (
                self.writer is None
                or (self.max_bytes is not None and self.used_bytes > self.max_bytes)
                or expired_by_duration
            ):
                # don't close the old writer. it's not necessary because all data is flushed, and it opens a race where
                # one thread gets a handle to the current writer and another closes it before the write actually happens

                self.used_bytes = len(data)
                self.start_time = now
                self.writer = FlushOptimizingWriter(open(
                    os.path.join(self.directory, self.file_pattern.format(now.isoformat().replace("+00:00", "Z"))),
                    "xb",
                ))
            writer = self.writer
        assert writer is not None
        writer.write(data)

    def __call__(self, log: object) -> None:
        self.write((str(log) + "\n").encode("utf8"))
