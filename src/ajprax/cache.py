from functools import partial
from inspect import signature
from threading import Lock, Condition

from ajprax.require import require
from ajprax.sentinel import Unset


def cache(f=None, *, key=Unset, method=False):
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
    if f is not None:
        return cache(key=key, method=method)(f)

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

    def __init__(self, lock):
        Condition.__init__(self, lock)
        self.exception = None


class Cache:
    def __init__(self, f, key, method, waiting_for_in_progress=None):
        self.f = f
        self.signature = signature(f)
        self.key = DefaultKey(tuple(self.signature.parameters)) if key is Unset else key
        self.method = method
        # Event used for synchronization in tests
        self.waiting_for_in_progress = waiting_for_in_progress
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
            existing = cell.value.get(key, Unset)
            if existing is Unset:
                condition = InProgress(cell.lock)
                cell.value[key] = condition
            elif isinstance(existing, InProgress):
                condition = existing
                while condition.exception is None and cell.value.get(key) is condition:
                    if self.waiting_for_in_progress:
                        self.waiting_for_in_progress.set()
                    condition.wait()
                # raise exception in all waiting threads
                if condition.exception is not None:
                    raise condition.exception
                return cell.value[key]
            else:
                return existing

        try:
            value = self.f(*args.args, **args.kwargs)
        except Exception as e:
            with cell.lock:
                if cell.value.get(key) is condition:
                    condition.exception = e
                    condition.notify_all()
                    del cell.value[key]
            raise

        with cell.lock:
            # it's important to set this value under the lock to avoid a race condition where a concurrent caller
            # gets the InProgress from the dict before we overwrite it, but waits on it after we notify_all.
            # in short, all reads and writes to the dict should be under the lock.
            cell.value[key] = value
            condition.notify_all()
        return value
