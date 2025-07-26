from functools import partial
from inspect import signature
from threading import Lock

from ajprax.require import require
from ajprax.sentinel import Unset


def lazy(f):
    return Lazy(f)


def cache(key=Unset):
    def decorator(f):
        return Cache(f, key)

    return decorator


class Lazy:
    def __init__(self, f):
        self.f = f
        self.instance_method = False
        self.value = Unset
        self.lock = Lock()

    def __set_name__(self, owner, name):
        self.instance_method = True

    def __get__(self, instance, owner=None):
        return partial(self, instance)

    def __call__(self, *a, **kw):
        if self.instance_method:
            # this handles the case of Cls.lazy_method(instance)
            if len(a) == 2 and a[0] is None:
                a = a[1],
            require(len(a) == 1 and not kw, a=a, kw=kw)

        if self.value is Unset:
            with self.lock:
                if self.value is Unset:
                    self.value = self.f(*a, **kw)
                    del self.f
                    del self.lock
        return self.value


class DefaultKey:
    def __init__(self, parameters):
        self.parameters = parameters
        self.instance_method = False

    def __call__(self, **kw):
        # drop self for instance methods since it will always be present and may not be hashable
        return tuple(kw[param] for param in self.parameters[self.instance_method:])


class Cache(dict):
    def __init__(self, f, key=Unset):
        dict.__init__(self)
        self.f = f
        self.signature = signature(f)
        self.key = DefaultKey(tuple(self.signature.parameters)) if key is Unset else key
        self.lock = Lock()

    def __set_name__(self, owner, name):
        if isinstance(self.key, DefaultKey):
            self.key.instance_method = True

    def __get__(self, instance, owner=None):
        return partial(self, instance)

    def __call__(self, *a, **kw):
        # arguments are bound and defaults applied to ensure consistent behavior regardless of how args are specified
        args = self.signature.bind(*a, **kw)
        args.apply_defaults()
        k = self.key(**args.arguments)
        if k not in self:
            with self.lock:
                if k not in self:
                    self[k] = self.f(*a, **kw)
        return self[k]
