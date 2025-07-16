from datetime import datetime, timezone

from ajprax.collections import Iter
from ajprax.logging import LEVEL_NAME, Logger, Log, INFO


class TestLog:
    def test_level(self):
        def test(log_level, message_level, *a, **kw):
            def check(log):
                # log level is passed through to the Log object
                assert log.level == message_level
                # log is only sent if the message level meets or exceeds the log level
                assert message_level >= log_level

            logger = Logger()
            logger.level = log_level
            logger.subscribe(check)
            name = LEVEL_NAME[message_level].strip().lower()
            getattr(logger, name)(*a, **kw)

        for log_level, message_level in Iter(LEVEL_NAME).product(repeat=2):
            test(log_level, message_level)
            test(log_level, message_level, "message")
            test(log_level, message_level, key="value")
            test(log_level, message_level, "message", key="value")

    def test_format(self):
        def test(datetime, level, message, keywords, exception):
            log = str(Log(datetime, level, message, keywords, exception))
            assert ("log message" in log) == bool(message)
            assert ("key='value'" in log) == bool(keywords)
            assert ("exception message" in log) == bool(exception)
            assert log.split("\n")[0].startswith("2024-01-01T00:00:00.000001Z  INFO ")

        for (message, keywords, exception) in Iter((True, False)).product(repeat=3):
            message = "log message" if message else ""
            keywords = {"key": "value"} if keywords else {}
            exception = Exception("exception message") if exception else None
            test(
                datetime(2024, 1, 1, 0, 0, 0, 1, tzinfo=timezone.utc),
                INFO,
                message,
                keywords,
                exception,
            )
