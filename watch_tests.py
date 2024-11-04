from os import path
from subprocess import run
from threading import Thread, Event

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

directory = path.dirname(__file__)


class Runner(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.event = Event()

    def run(self):
        while True:
            try:
                run(["source venv/bin/activate && clear && date && hatch test"], cwd=directory, shell=True)
            except KeyboardInterrupt:
                raise
            except:
                pass
            # in principle, a real change could happen while the tests are running, but it is very unlikely since the
            # tests are fast, and in reality one file change emits multiple events causing the tests to run multiple
            # times if we put clear at the top
            self.event.clear()
            # wait at the end so the tests run once on startup without needing a file change
            self.event.wait()

    def should_run(self):
        self.event.set()


class Handler(FileSystemEventHandler):
    def __init__(self):
        self.runner = Runner()
        self.runner.start()

    def handle(self, filename):
        if filename.endswith(".py"):
            self.runner.should_run()

    def on_created(self, event):
        self.handle(event.src_path)

    def on_modified(self, event):
        self.handle(event.src_path)


if __name__ == "__main__":
    obs = Observer()
    obs.schedule(Handler(), directory, recursive=True)
    obs.start()
