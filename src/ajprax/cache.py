from functools import partial
from inspect import signature
from threading import Lock, Condition

from ajprax.require import require
from ajprax.sentinel import Unset


def cache(*, key=Unset, method=False):
    """
    Major differences from functools.cache:
    - Cached function will not be called more than once for the same key, even if a second call happens before the
      first completes. Different keys can still be called concurrently.
    - Allows customizing key generation.

    Works with bare functions, instance methods, classmethods, and staticmethods.
    Instance methods and classmethods should use method=True. This will store the cache on the instance for isolation
      and so that the cache can be garbage collected with the instance. This also excludes the instance or class from
      the key function so that methods on unhashable types can still be cached.

    Handles arguments uniformly, including default values and arguments specified as any mix of positional and keyword.

    Default key function produces a tuple of arguments, so all arguments must be hashable
      (except self/cls if method=True).

    Automatically optimizes implementation when used to create a singleton.
    """

    def decorator(f):
        if len(signature(f).parameters) == method:
            require(key is Unset, "cannot use custom key function for function with no arguments")
            return Singleton(f, method)
        return Cache(f, key, method)

    return decorator


class Cell:
    """Container which can be attached to an instance or class for cache isolation"""

    def __init__(self, value=Unset):
        self.value = value
        self.lock = Lock()


class Singleton:
    def __init__(self, f, method):
        self.f = f
        self.method = method
        self.creation_lock = Lock()
        self.name = f"_{f.__name__}_singleton"

    def __get__(self, instance, owner=None):
        return self if instance is None else partial(self, instance)

    def get_or_create_cell(self, obj):
        def get(obj):
            return getattr(obj, self.name, None)

        if not get(obj):
            with self.creation_lock:
                if not get(obj):
                    setattr(obj, self.name, Cell())
        return get(obj)

    def __call__(self, *a, **kw):
        cell = self.get_or_create_cell(a[0] if self.method else self)
        if cell.value is Unset:
            with cell.lock:
                if cell.value is Unset:
                    cell.value = self.f(*a, **kw)
        return cell.value


class DefaultKey:
    def __init__(self, parameters):
        self.parameters = parameters

    def __call__(self, *a, **kw):
        return *a, *(kw[param] for param in self.parameters if param in kw)


class InProgress(Condition):
    """
    Inserted into a cache before starting to generate the value so that concurrent callers can wait for the value
    instead of redundantly calling the cached function.

    Wrapped so that we don't mistake user Condition values for our marker
    """


class Cache:
    def __init__(self, f, key, method):
        self.f = f
        self.signature = signature(f)
        self.key = DefaultKey(tuple(self.signature.parameters)) if key is Unset else key
        self.method = method
        self.name = f"_{f.__name__}_cache"
        self.creation_lock = Lock()

    def get_or_create_cell(self, obj):
        def get(obj):
            return getattr(obj, self.name, None)

        if not get(obj):
            with self.creation_lock:
                if not get(obj):
                    setattr(obj, self.name, Cell({}))
        return get(obj)

    def __get__(self, instance, owner=None):
        return self if instance is None else partial(self, instance)

    def __call__(self, *a, **kw):
        args = self.signature.bind(*a, **kw)
        args.apply_defaults()
        key = self.key(*args.args[self.method:], **args.kwargs)

        cell = self.get_or_create_cell(a[0] if self.method else self)
        with cell.lock:
            if key in cell.value:
                value = cell.value[key]
                if isinstance(value, InProgress):
                    with value:
                        value.wait()
                        value = cell.value[key]
                return value

            condition = InProgress(cell.lock)
            cell.value[key] = condition

        value = self.f(*args.args, **args.kwargs)
        cell.value[key] = value
        with cell.lock:
            condition.notify()
        return value
