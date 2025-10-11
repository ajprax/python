import json
import os
import sys
import traceback
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock, Event
from typing import Optional

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


@dataclass
class Log:
    datetime: datetime
    level: int
    message: str
    keywords: dict
    exception: Optional[BaseException]

    def __post_init__(self):
        if isinstance(self.exception, bool):
            if self.exception:
                self.exception = sys.exc_info()[1]
            else:
                self.exception = None

    def __str__(self):
        ts = self.datetime.isoformat().replace("+00:00", "Z")
        level = LEVEL_NAME[self.level]
        message = str(self.message)
        if self.keywords:
            message += " " if message else ""
            message += " ".join(f"{k}={repr(v)}" for k, v in self.keywords.items())
        if self.exception:
            message += "\n"
            message += "".join("\t" + line for line in traceback.format_exception(self.exception))
        return f"{ts} {level} {message}"

    @property
    def json(self):
        d = dict(
            time=self.datetime.isoformat().replace("+00:00", "Z"),
            level=LEVEL_NAME[self.level],
        )
        if self.message:
            d["message"] = str(self.message)
        if self.keywords:
            d["keywords"] = {str(k): repr(v) for k, v in self.keywords.items()}
        if self.exception:
            d["exception"] = "".join("\t" + line for line in traceback.format_exception(self.exception))
        return json.dumps(d)


class Logger(Events):
    def __init__(self):
        Events.__init__(self)
        self._global_level = INFO
        self._context_level = ContextVar("level", default=None)
        self._kwargs = ContextVar("kwargs", default={})
        self.subscribe(print)

    @property
    def level(self):
        context_level = self._context_level.get()
        if context_level is None:
            return self._global_level
        return context_level

    @level.setter
    def level(self, level):
        self._global_level = level

    def _log(self, _level, _message, _exception, **kwargs):
        if _level >= self.level:
            self.send(Log(
                datetime.now(timezone.utc),
                _level,
                _message,
                {**self._kwargs.get(), **kwargs},
                _exception,
            ))

    def trace(self, _message="", _exception=False, **kwargs):
        self._log(TRACE, _message, _exception, **kwargs)

    def debug(self, _message="", _exception=False, **kwargs):
        self._log(DEBUG, _message, _exception, **kwargs)

    def info(self, _message="", _exception=False, **kwargs):
        self._log(INFO, _message, _exception, **kwargs)

    def warn(self, _message="", _exception=False, **kwargs):
        self._log(WARN, _message, _exception, **kwargs)

    def error(self, _message="", _exception=False, **kwargs):
        self._log(ERROR, _message, _exception, **kwargs)

    def fatal(self, _message="", _exception=False, **kwargs):
        self._log(FATAL, _message, _exception, **kwargs)

    @contextmanager
    def context_level(self, level):
        token = self._context_level.set(level)
        try:
            yield
        finally:
            self._context_level.reset(token)

    @contextmanager
    def context_kwargs(self, **kw):
        token = self._kwargs.set(kw)
        try:
            yield
        finally:
            self._kwargs.reset(token)


log = Logger()


class FlushOptimizingWriter:
    def __init__(self, file):
        self.file = file
        self.write_lock = Lock()
        self.buffer_lock = Lock()
        self.buffer = []

    def write(self, data):
        with self.buffer_lock:
            writer = not self.buffer
            event = Event()
            self.buffer.append((data, event))

        if writer:
            with self.write_lock:
                # swapping the buffer must happen under the write lock to guarantee that a second writer can't get the
                # new buffer and the write lock before this one, which would violate ordering expectations
                with self.buffer_lock:
                    to_write, self.buffer = self.buffer, []
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

    def __init__(self, directory, *, max_bytes=None, max_duration=None, file_pattern="{}.log", json=False):
        require(
            max_bytes is not None or max_duration is not None,
            "must specify at least one of max_bytes and max_duration",
        )

        self.directory = directory
        self.max_bytes = max_bytes
        self.max_duration = max_duration
        self.file_pattern = file_pattern
        self.format = (lambda log: log.json) if json else str

        self.lock = Lock()
        self.writer = None
        self.used_bytes = 0
        self.start_time = None

    def write(self, bytes):
        with self.lock:
            now = datetime.now(timezone.utc)
            self.used_bytes += len(bytes)
            if self.writer is None:
                os.makedirs(self.directory, exist_ok=True)

            if (
                self.writer is None
                or (self.max_bytes is not None and self.used_bytes > self.max_bytes)
                or (self.max_duration is not None and now - self.start_time > self.max_duration)
            ):
                # don't close the old writer. it's not necessary because all data is flushed, and it opens a race where
                # one thread gets a handle to the current writer and another closes it before the write actually happens

                self.used_bytes = len(bytes)
                self.start_time = now
                self.writer = FlushOptimizingWriter(open(
                    os.path.join(self.directory, self.file_pattern.format(now.isoformat().replace("+00:00", "Z"))),
                    "xb",
                ))
        self.writer.write(bytes)

    def __call__(self, log):
        self.write((self.format(log) + "\n").encode("utf8"))
