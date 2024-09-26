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
        self._callbacks.append(callback)

    def unsubscribe(self, callback):
        self._callbacks.remove(callback)

    def notify(self, *a, **kw):
        for callback in self._callbacks:
            callback(*a, **kw)


class Events(Notifications):
    def send(self, event):
        self.notify(event)

    def filter(self, fn):
        from ajprax.subscriptions._events import Filtered
        return Filtered(self, fn)

    def flat_map(self, fn):
        from ajprax.subscriptions._events import FlatMapped
        return FlatMapped(self, fn)

    def map(self, fn):
        from ajprax.subscriptions._events import Mapped
        return Mapped(self, fn)

    def on(self):
        return Notifications.on(self)


class Value(Notifications):
    def __init__(self, initial=Unset):
        Notifications.__init__(self)
        self._value = initial

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
        Notifications.subscribe(self, callback)
        if self._value is not Unset:
            callback(self._value)

    def changed(self):
        return self.changes().on()

    def changes(self):
        from ajprax.subscriptions._value import Changes
        return Changes(self)

    def map(self, fn):
        from ajprax.subscriptions._value import Mapped
        return Mapped(self, fn)

    def on(self):
        return Notifications.on(self)

    def zip(self, *others):
        from ajprax.subscriptions._value import Zipped
        return Zipped(self, *others)
