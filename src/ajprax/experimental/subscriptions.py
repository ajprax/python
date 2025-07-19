from ajprax.hof import identity
from ajprax.sentinel import Unset
from abc import abstractmethod, ABC
from queue import Queue

from ajprax.collections import Iter


class Notifications:
    @classmethod
    def on(cls, on):
        notifications = Notifications()
        on.subscribe(lambda *a, **kw: notifications.notify())
        return notifications

    def __init__(self):
        self._callbacks = []

    def subscribe(self, callback):
        self._callbacks.append(callback)
        return Unsubscribe(self, callback)

    def unsubscribe(self, callback):
        self._callbacks.remove(callback)

    def notify(self, *a, **kw):
        for callback in self._callbacks:
            callback(*a, **kw)


class Unsubscribe:
    def __init__(self, notifications, callback):
        self.notifications = notifications
        self.callback = callback

    def __call__(self, *args, **kwargs):
        self.notifications.unsubscribe(self.callback)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self()


class Events(Notifications):
    def send(self, event):
        self.notify(event)

    def __iter__(self):
        return IterEvents(self)

    def all(self, key=Unset):
        pass

    def any(self, key=Unset):
        pass

    def batch(self, size):
        pass

    def count(self, key=Unset):
        pass

    def distinct(self, key=Unset):
        pass

    def filter(self, fn):
        return FilteredEvents(self, fn)

    def flat_map(self, fn):
        return FlatMappedEvents(self, fn)

    def map(self, fn):
        return MappedEvents(self, fn)

    def on(self):
        return Notifications.on(self)


class Value(Notifications):
    def __iadd__(self, other):
        self.value += other
        return self

    def __iand__(self, other):
        self.value &= other
        return self

    def __idiv__(self, other):
        self.value /= other
        return self

    def __ifloordiv__(self, other):
        self.value //= other
        return self

    def __ilshift__(self, other):
        self.value <<= other
        return self

    def __imatmul__(self, other):
        self.value @= other
        return self

    def __imod__(self, other):
        self.value %= other
        return self

    def __init__(self, initial=Unset):
        Notifications.__init__(self)
        self._value = initial

    def __imul__(self, other):
        self.value *= other
        return self

    def __ior__(self, other):
        self.value |= other
        return self

    def __ipow__(self, other):
        self.value **= other
        return self

    def __irshift__(self, other):
        self.value >>= other
        return self

    def __isub__(self, other):
        self.value -= other
        return self

    def __itruediv__(self, other):
        self.value /= other
        return self

    def __ixor__(self, other):
        self.value ^= other
        return self

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self.set(value)

    def set(self, value):
        self._value = value
        self.notify(value)

    def subscribe(self, callback):
        if self._value is not Unset:
            callback(self._value)
        return Notifications.subscribe(self, callback)

    def all(self, key=Unset):
        return AllValue(self, key)

    def any(self, key=Unset):
        return AnyValue(self, key)

    def changed(self):
        return self.changes().on()

    def changes(self):
        return Changes(self)

    def has_been(self, predicate=bool):
        return HasBeen(self, predicate)

    def is_(self, predicate=bool):
        return Is(self, predicate)

    def map(self, fn):
        return MappedValue(self, fn)

    def on(self):
        return Notifications.on(self)

    def zip(self, *others):
        return ZippedValue(self, *others)


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


class AllEvents(Value):
    def __init__(self, events, key=Unset):
        self.key = identity if key is Unset else key
        Value.__init__(self)
        self.set(True)
        events.subscribe(self.on_event)

    def on_event(self, event):
        self.value &= bool(self.key(event))


class IterEvents(EventsCombinator, Iter):
    def __init__(self, events):
        def iter_from_queue():
            while True:
                yield self.queue.get()

        self.queue = Queue()
        Iter.__init__(self, iter_from_queue())
        EventsCombinator.__init__(self, events)

    def on_event(self, event):
        print("putting", event)
        self.queue.put(event)


class FilteredEvents(EventsCombinator):
    def __init__(self, events, fn):
        self.fn = fn
        EventsCombinator.__init__(self, events)

    def on_event(self, event):
        if self.fn(event):
            self.send(event)


class FlatMappedEvents(EventsCombinator):
    def __init__(self, events, fn):
        self.fn = fn
        EventsCombinator.__init__(self, events)

    def on_event(self, event):
        for event in self.fn(event):
            self.send(event)


class MappedEvents(EventsCombinator):
    def __init__(self, events, fn):
        self.fn = fn
        EventsCombinator.__init__(self, events)

    def on_event(self, event):
        self.send(self.fn(event))


class ValueCombinator(ABC, Value):
    def __init__(self, value):
        Value.__init__(self)
        self.v = value
        self.v.subscribe(self.on_set)

    @abstractmethod
    def on_set(self, value):
        pass

    def close(self):
        self.v.unsubscribe(self.on_set)


class AllValue(ValueCombinator):
    def __init__(self, value, key):
        self.key = identity if key is Unset else key
        ValueCombinator.__init__(self, value)
        self.set(True)

    def on_set(self, value):
        self.value &= bool(self.key(value))


class AnyValue(ValueCombinator):
    def __init__(self, value, key):
        self.key = identity if key is Unset else key
        ValueCombinator.__init__(self, value)
        self.set(False)

    def on_set(self, value):
        self.value |= bool(self.key(value))


class Changes(ValueCombinator):
    def on_set(self, value):
        if value != self.value:
            self.set(value)


class HasBeen(ValueCombinator):
    def __init__(self, value, predicate):
        self.predicate = predicate
        ValueCombinator.__init__(self, value)

    def on_set(self, value):
        if self.value is Unset:
            self.value = self.predicate(value)
        else:
            self.value |= self.predicate(value)


class Is(ValueCombinator):
    def __init__(self, value, predicate):
        self.predicate = predicate
        ValueCombinator.__init__(self, value)

    def on_set(self, value):
        self.set(self.predicate(value))


class MappedValue(ValueCombinator):
    def __init__(self, value, fn):
        self.fn = fn
        ValueCombinator.__init__(self, value)

    def on_set(self, value):
        self.set(self.fn(value))


class ZippedValue(Value):
    def __init__(self, *values):
        Value.__init__(self)
        raw = [Unset] * len(values)
        for i, value in enumerate(values):
            # i=i is here because if we allow seti to close over i then all copies will see the highest value of i
            # instead of their particular value
            def seti(value, i=i):
                raw[i] = value
                # TODO: avoid checking this every time
                if Unset not in raw:
                    self.set(tuple(raw))

            value.subscribe(seti)
