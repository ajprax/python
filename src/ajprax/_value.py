from abc import ABC, abstractmethod

from ajprax.sentinel import Unset
from ajprax.subscriptions import Value


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


class Always(ValueCombinator):
    def __init__(self, value, predicate):
        self.predicate = predicate
        ValueCombinator.__init__(self, value)

    def on_set(self, value):
        if self.value is Unset:
            self.value = self.predicate(value)
        else:
            self.value &= self.predicate(value)


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


class Mapped(ValueCombinator):
    def __init__(self, value, fn):
        self.fn = fn
        ValueCombinator.__init__(self, value)

    def on_set(self, value):
        self.set(self.fn(value))


class Zipped(Value):
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
