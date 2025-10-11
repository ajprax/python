import json
from datetime import datetime, timezone
from threading import Thread, Event

from ajprax.collections import Iter
from ajprax.logging import LEVEL_NAME, Logger, Log, INFO, WARN


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

    def test_json(self):
        def test(datetime, level, message, keywords, exception):
            log = json.loads(Log(datetime, level, message, keywords, exception).json)
            assert "2024-01-01T00:00:00.000001Z" == log["time"]
            assert LEVEL_NAME[level] == log["level"]
            if message:
                assert message == log["message"]
            else:
                assert "message" not in log
            if keywords:
                assert {k: repr(v) for k, v in keywords.items()} == log["keywords"]
            else:
                assert "keywords" not in log
            if exception:
                assert str(exception) in log["exception"]
            else:
                assert "exception" not in log

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

    def test_context_level_prevents(self):
        def a():
            with log.context_level(WARN):
                bevent.set()
                log.info("a")
                aevent.wait()

        def b():
            bevent.wait()
            log.info("b")
            aevent.set()

        def test(log):
            assert log.message == "b"

        log = Logger()
        log.subscribe(test)

        aevent = Event()
        bevent = Event()
        athread = Thread(target=a)
        bthread = Thread(target=b)
        athread.start()
        bthread.start()
        athread.join()
        bthread.join()

    def test_context_level_allows(self):
        def a():
            with log.context_level(INFO):
                bevent.set()
                log.info("a")
                aevent.wait()

        def b():
            bevent.wait()
            log.info("b")
            aevent.set()

        def test(log):
            assert log.message == "a"

        log = Logger()
        log.subscribe(test)
        log.level = WARN

        aevent = Event()
        bevent = Event()
        athread = Thread(target=a)
        bthread = Thread(target=b)
        athread.start()
        bthread.start()
        athread.join()
        bthread.join()

    def test_context_kwargs(self):
        def a():
            with log.context_kwargs(a=True):
                bevent.set()
                aevent.wait()

        def b():
            bevent.wait()
            log.info(b=True)
            aevent.set()

        def test(log):
            assert "a" not in log.keywords
            assert "b" in log.keywords

        log = Logger()
        log.subscribe(test)

        aevent = Event()
        bevent = Event()
        athread = Thread(target=a)
        bthread = Thread(target=b)
        athread.start()
        bthread.start()
        athread.join()
        bthread.join()
