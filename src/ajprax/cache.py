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
        return CacheManager(f, key, method)

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
        return partial(self, instance)

    def create_cell(self, obj):
        setattr(obj, self.name, Cell())

    def get_cell(self, obj):
        return getattr(obj, self.name, Unset)

    def get_or_create_cell(self, obj):
        if self.get_cell(obj) is Unset:
            with self.creation_lock:
                if self.get_cell(obj) is Unset:
                    self.create_cell(obj)
        return self.get_cell(obj)

    def __call__(self, *a, **kw):
        if self.method:
            # Cls.method(instance) results in an extra None inserted by __get__
            # instance.method() won't have the None
            # the first positional argument of an instance method will never actually be None
            if len(a) == 2 and a[0] is None:
                a = a[1],
            require(len(a) == 1 and not kw, a=a, kw=kw)

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


class Ongoing(Condition):
    """
    Inserted into a cache before starting to generate the value so that concurrent callers can wait for the value
    instead of redundantly calling the cached function.

    Wrapped so that we don't mistake user Condition values for our marker
    """


class CacheManager:
    def __init__(self, f, key, method):
        self.f = f
        self.signature = signature(f)
        self.key = DefaultKey(tuple(self.signature.parameters)) if key is Unset else key
        self.method = method
        self.name = f"_{f.__name__}_cache"
        self.creation_lock = Lock()

    def create_cell(self, obj):
        setattr(obj, self.name, Cell({}))

    def get_cell(self, obj):
        return getattr(obj, self.name, Unset)

    def get_or_create_cell(self, obj):
        if self.get_cell(obj) is Unset:
            with self.creation_lock:
                if self.get_cell(obj) is Unset:
                    self.create_cell(obj)
        return self.get_cell(obj)

    def __get__(self, instance, owner=None):
        return partial(self, instance)

    def __call__(self, *a, **kw):
        # Cls.method(instance, ...) results in an extra None inserted by __get__
        # instance.method(...) won't have the None
        # the first positional argument of an instance method will never actually be None
        if self.method and a[0] is None:
            a = a[1:]

        cell = self.get_or_create_cell(a[0] if self.method else self)

        args = self.signature.bind(*a, **kw)
        args.apply_defaults()
        key = self.key(*args.args[self.method:], **args.kwargs)

        with cell.lock:
            if key in cell.value:
                value = cell.value[key]
                if isinstance(value, Ongoing):
                    with value:
                        value.wait()
                        value = cell.value[key]
                return value

            condition = Ongoing(cell.lock)
            cell.value[key] = condition

        value = self.f(*args.args, **args.kwargs)
        cell.value[key] = value
        with cell.lock:
            condition.notify()
        return value
