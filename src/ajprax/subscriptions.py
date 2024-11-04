from ajprax.sentinel import Unset


class Notifications:
    @classmethod
    def on(cls, on):
        notifications = Notifications()
        on.subscribe(lambda *a, **kw: notifications.notify())
        return notifications

    def __init__(self):
        self._callbacks = []

    def subscribe(self, callback):
        from ajprax._notifications import UnsubscribeOnExit
        self._callbacks.append(callback)
        return UnsubscribeOnExit(self, callback)

    def unsubscribe(self, callback):
        self._callbacks.remove(callback)

    def notify(self, *a, **kw):
        for callback in self._callbacks:
            callback(*a, **kw)


class Events(Notifications):
    def send(self, event):
        self.notify(event)

    def filter(self, fn):
        from ajprax._events import Filtered
        return Filtered(self, fn)

    def flat_map(self, fn):
        from ajprax._events import FlatMapped
        return FlatMapped(self, fn)

    def map(self, fn):
        from ajprax._events import Mapped
        return Mapped(self, fn)

    def on(self):
        return Notifications.on(self)


class Value(Notifications):
    def __iadd__(self, other):
        self.value += other

    def __iand__(self, other):
        self.value &= other

    def __idiv__(self, other):
        self.value /= other

    def __ifloordiv__(self, other):
        self.value //= other

    def __ilshift__(self, other):
        self.value <<= other

    def __imatmul__(self, other):
        self.value @= other

    def __imod__(self, other):
        self.value %= other

    def __init__(self, initial=Unset):
        Notifications.__init__(self)
        self._value = initial

    def __imul__(self, other):
        self.value *= other

    def __ior__(self, other):
        self.value |= other

    def __ipow__(self, other):
        self.value **= other

    def __irshift__(self, other):
        self.value >>= other

    def __isub__(self, other):
        self.value -= other

    def __itruediv__(self, other):
        self.value /= other

    def __ixor__(self, other):
        self.value ^= other

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

    def always(self, key=Unset):
        pass  # TODO: like all

    def is_(self, predicate=Unset):
        pass

    def has_been(self, predicate=Unset):
        pass

    def changed(self):
        return self.changes().on()

    def changes(self):
        from ajprax._value import Changes
        return Changes(self)

    def map(self, fn):
        from ajprax._value import Mapped
        return Mapped(self, fn)

    def on(self):
        return Notifications.on(self)

    def zip(self, *others):
        from ajprax._value import Zipped
        return Zipped(self, *others)
