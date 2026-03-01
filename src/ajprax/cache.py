import functools
import inspect
import threading
import weakref


class DoNotCache(Exception):
    def __init__(self, value):
        super().__init__()
        self.value = value


class _InFlight:
    __slots__ = ("event", "value", "exception", "do_not_cache")

    def __init__(self):
        self.event = threading.Event()
        self.value = None
        self.exception = None
        self.do_not_cache = False


class _ScopeState:
    __slots__ = (
        "lock",
        "values",
        "in_flight",
        "singleton_set",
        "singleton_value",
        "singleton_in_flight",
    )

    def __init__(self):
        self.lock = threading.RLock()
        self.values = {}
        self.in_flight = {}
        self.singleton_set = False
        self.singleton_value = None
        self.singleton_in_flight = None


def _signature_without_first_arg(signature):
    params = list(signature.parameters.values())
    if not params:
        return signature
    return signature.replace(parameters=params[1:])


def _signatures_match(expected, actual):
    expected_params = list(expected.parameters.values())
    actual_params = list(actual.parameters.values())
    if len(expected_params) != len(actual_params):
        return False

    for left, right in zip(expected_params, actual_params):
        if left.kind != right.kind:
            return False
        if left.kind != inspect.Parameter.POSITIONAL_ONLY and left.name != right.name:
            return False
        if (left.default is inspect._empty) != (right.default is inspect._empty):
            return False
    return True


def _make_canonical_key(signature, args, kwargs):
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()

    items = []
    for parameter in signature.parameters.values():
        value = bound.arguments[parameter.name]
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            normalized = tuple(value)
        elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
            normalized = tuple(sorted(value.items()))
        else:
            normalized = value
        items.append((parameter.name, normalized))
    return tuple(items)


class _BoundCachedCallable:
    def __init__(self, cached, owner_kind, owner):
        self._cached = cached
        self._owner_kind = owner_kind
        self._owner = owner
        functools.update_wrapper(self, cached._func)

    def __call__(self, *args, **kwargs):
        return self._cached._invoke(self._owner_kind, self._owner, args, kwargs)

    def clear(self):
        self._cached._clear(self._owner_kind, self._owner)

    def remove(self, *args, **kwargs):
        self._cached._remove(self._owner_kind, self._owner, args, kwargs)


class _CachedCallable:
    def __init__(self, func, key=None, mode="auto"):
        functools.update_wrapper(self, func)
        self._func = func
        self._key_func = key
        self._mode = mode
        self._declaring_owner = None

        self._signature = inspect.signature(func)
        self._method_signature = _signature_without_first_arg(self._signature)
        self._is_singleton_full = len(self._signature.parameters) == 0
        self._is_singleton_method = len(self._method_signature.parameters) == 0

        self._global_state = _ScopeState()
        self._instance_states = weakref.WeakKeyDictionary()
        self._class_states = weakref.WeakKeyDictionary()
        self._state_lock = threading.RLock()

        self._key_signature = None
        if self._key_func is not None:
            if not callable(self._key_func):
                raise TypeError("key must be callable")
            self._key_signature = inspect.signature(self._key_func)
            self._validate_key_signature()

    def __set_name__(self, owner, name):
        self._declaring_owner = owner
        if self._key_signature is not None and self._mode == "auto":
            if not _signatures_match(self._method_signature, self._key_signature):
                raise TypeError(
                    "key signature must match the decorated method signature "
                    "excluding self/cls"
                )

    def __get__(self, obj, owner):
        if self._mode == "class":
            return _BoundCachedCallable(self, "class", owner)
        if self._mode == "static":
            return self
        if obj is None:
            return self
        return _BoundCachedCallable(self, "instance", obj)

    def __call__(self, *args, **kwargs):
        if self._mode == "class":
            if not args:
                raise TypeError("missing class argument for cached classmethod")
            return self._invoke("class", args[0], args[1:], kwargs)

        if (
            self._mode == "auto"
            and self._declaring_owner is not None
            and args
            and isinstance(args[0], self._declaring_owner)
        ):
            return self._invoke("instance", args[0], args[1:], kwargs)

        return self._invoke("none", None, args, kwargs)

    def clear(self):
        self._clear("none", None)

    def remove(self, *args, **kwargs):
        self._remove("none", None, args, kwargs)

    def _validate_key_signature(self):
        if self._key_signature is None:
            return

        if self._mode == "class":
            valid = _signatures_match(self._method_signature, self._key_signature)
        elif self._mode == "static":
            valid = _signatures_match(self._signature, self._key_signature)
        else:
            valid = _signatures_match(self._signature, self._key_signature) or _signatures_match(
                self._method_signature, self._key_signature
            )

        if not valid:
            raise TypeError("key signature does not match decorated function signature")

    def _state_for(self, owner_kind, owner, create=True):
        if owner_kind == "none":
            return self._global_state

        mapping = self._instance_states if owner_kind == "instance" else self._class_states

        with self._state_lock:
            try:
                return mapping[owner]
            except KeyError:
                if not create:
                    return None
                state = _ScopeState()
                mapping[owner] = state
                return state
            except TypeError as exc:
                raise TypeError(
                    "cache owner must be weak-referenceable to preserve non-leaking semantics"
                ) from exc

    def _signature_for(self, owner_kind):
        if owner_kind in ("instance", "class"):
            return self._method_signature
        return self._signature

    def _is_singleton_for(self, owner_kind):
        if owner_kind in ("instance", "class"):
            return self._is_singleton_method
        return self._is_singleton_full

    def _compute_cache_key(self, owner_kind, args, kwargs):
        if self._key_func is not None:
            return self._key_func(*args, **kwargs)
        signature = self._signature_for(owner_kind)
        return _make_canonical_key(signature, args, kwargs)

    def _invoke(self, owner_kind, owner, args, kwargs):
        state = self._state_for(owner_kind, owner, create=True)
        call_signature = self._signature_for(owner_kind)
        call_signature.bind(*args, **kwargs)

        if self._is_singleton_for(owner_kind):
            return self._invoke_singleton(state, lambda: self._call_underlying(owner, args, kwargs))

        key = self._compute_cache_key(owner_kind, args, kwargs)
        return self._invoke_keyed(
            state, key, lambda: self._call_underlying(owner, args, kwargs)
        )

    def _call_underlying(self, owner, args, kwargs):
        if owner is None:
            return self._func(*args, **kwargs)
        return self._func(owner, *args, **kwargs)

    def _invoke_singleton(self, state, compute):
        while True:
            with state.lock:
                if state.singleton_set:
                    return state.singleton_value

                in_flight = state.singleton_in_flight
                if in_flight is None:
                    in_flight = _InFlight()
                    state.singleton_in_flight = in_flight
                    producer = True
                else:
                    producer = False

            if producer:
                try:
                    value = compute()
                except DoNotCache as exc:
                    with state.lock:
                        if state.singleton_in_flight is in_flight:
                            state.singleton_in_flight = None
                        in_flight.do_not_cache = True
                        in_flight.value = exc.value
                        in_flight.event.set()
                    return exc.value
                except Exception as exc:
                    with state.lock:
                        if state.singleton_in_flight is in_flight:
                            state.singleton_in_flight = None
                        in_flight.exception = exc
                        in_flight.event.set()
                    raise
                else:
                    with state.lock:
                        state.singleton_set = True
                        state.singleton_value = value
                        if state.singleton_in_flight is in_flight:
                            state.singleton_in_flight = None
                        in_flight.value = value
                        in_flight.event.set()
                    return value

            in_flight.event.wait()
            if in_flight.exception is not None:
                raise in_flight.exception
            if in_flight.do_not_cache:
                continue
            return in_flight.value

    def _invoke_keyed(self, state, key, compute):
        while True:
            with state.lock:
                if key in state.values:
                    return state.values[key]

                in_flight = state.in_flight.get(key)
                if in_flight is None:
                    in_flight = _InFlight()
                    state.in_flight[key] = in_flight
                    producer = True
                else:
                    producer = False

            if producer:
                try:
                    value = compute()
                except DoNotCache as exc:
                    with state.lock:
                        state.in_flight.pop(key, None)
                        in_flight.do_not_cache = True
                        in_flight.value = exc.value
                        in_flight.event.set()
                    return exc.value
                except Exception as exc:
                    with state.lock:
                        state.in_flight.pop(key, None)
                        in_flight.exception = exc
                        in_flight.event.set()
                    raise
                else:
                    with state.lock:
                        state.values[key] = value
                        state.in_flight.pop(key, None)
                        in_flight.value = value
                        in_flight.event.set()
                    return value

            in_flight.event.wait()
            if in_flight.exception is not None:
                raise in_flight.exception
            if in_flight.do_not_cache:
                continue
            return in_flight.value

    def _clear(self, owner_kind, owner):
        if owner_kind == "none":
            states = [self._global_state]
        else:
            state = self._state_for(owner_kind, owner, create=False)
            if state is None:
                return
            states = [state]

        for state in states:
            with state.lock:
                state.values.clear()
                state.singleton_set = False
                state.singleton_value = None

    def _remove(self, owner_kind, owner, args, kwargs):
        state = self._state_for(owner_kind, owner, create=False)
        if state is None:
            return

        signature = self._signature_for(owner_kind)
        signature.bind(*args, **kwargs)

        with state.lock:
            if self._is_singleton_for(owner_kind):
                state.singleton_set = False
                state.singleton_value = None
                return

            key = self._compute_cache_key(owner_kind, args, kwargs)
            state.values.pop(key, None)



def cache(f=None, *, key=None):
    """
    Cache the results of the decorated function.

    - Automatically optimizes to singleton implementation when decorating a function with no arguments.
    - When decorating an instance method, creates a separate cache per instance and excludes `self` from
      generated keys. The cache does not prevent the instance from being garbage collection, and the cache
      can be freed after the instance.
    - When decorating a classmethod, creates a separate cache per subclass with the same semantics as when
      decorating an instance method.
    - When `key` is specified, it must be a function with the same signature as the decorated function
      (excluding `self` when decorating an instance method) and returning the cache key. When `key` is
      None, a default key function is used which creates a tuple of all arguments (including keyword
      arguments in a consistent order).
    - Caching is threadsafe. If a second call with the same arguments happens while one is being calculated
      the second thread will wait for the results of the first instead of calling the decorated function a
      second time.
    - If the decorated function raises `DoNotCache`, the value will be returned but not included in the cache.
      If a second caller is waiting for the result, it will execute the decorated function after the first
      raises `DoNotCache`.
    - The cache can be cleared, and individual keys removed.
    """

    if f is None:
        return lambda func: cache(func, key=key)

    if isinstance(f, classmethod):
        wrapped = _CachedCallable(f.__func__, key=key, mode="class")
        return wrapped

    if isinstance(f, staticmethod):
        wrapped = _CachedCallable(f.__func__, key=key, mode="static")
        return wrapped

    return _CachedCallable(f, key=key, mode="auto")
