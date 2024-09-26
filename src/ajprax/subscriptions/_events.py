from abc import abstractmethod, ABC

from ajprax.subscriptions import Events


class EventsCombinator(ABC, Events):
    def __init__(self, events):
        Events.__init__(self)
        self.events = events
        self.events.subscribe(self.on_event)

    @abstractmethod
    def on_event(self, event):
        pass

    def close(self):
        self.events.unsubscribe(self.on_event)


class Filtered(EventsCombinator):
    def __init__(self, events, fn):
        self.fn = fn
        EventsCombinator.__init__(self, events)

    def on_event(self, event):
        if self.fn(event):
            self.send(event)


class FlatMapped(EventsCombinator):
    def __init__(self, events, fn):
        self.fn = fn
        EventsCombinator.__init__(self, events)

    def on_event(self, event):
        for event in self.fn(event):
            self.send(event)


class Mapped(EventsCombinator):
    def __init__(self, events, fn):
        self.fn = fn
        EventsCombinator.__init__(self, events)

    def on_event(self, event):
        self.send(self.fn(event))
