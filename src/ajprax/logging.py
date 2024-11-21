import sys
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from ajprax.experimental.subscriptions import Events
from ajprax.print import print

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
        message = str(self.message)
        ts = self.datetime.isoformat().replace("+00:00", "Z")
        level = LEVEL_NAME[self.level]
        if self.keywords:
            message += " " if message else ""
            message += " ".join(f"{k}={repr(v)}" for k, v in self.keywords.items())
        if self.exception:
            message += "\n"
            message += "".join("\t" + line for line in traceback.format_exception(self.exception))
        return f"{ts} {level} {message}"


class Logger(Events):
    def __init__(self):
        Events.__init__(self)
        self.level = INFO
        self.subscribe(print)

    def _log(self, _level, _message, _exception=False, **kwargs):
        if _level >= self.level:
            self.send(Log(datetime.now(timezone.utc), _level, _message, kwargs, _exception))

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
    def temporary_level(self, level):
        old = self.level
        self.level = level
        try:
            yield
        finally:
            self.level = old


log = Logger()
