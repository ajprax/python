from __future__ import annotations

import functools
import inspect
import threading
import weakref
from typing import (
    Callable,
    Generic,
    Literal,
    Optional,
    ParamSpec,
    Protocol,
    TYPE_CHECKING,
    TypeVar,
    Union,
    cast,
    overload,
)


R = TypeVar("R")
R_co = TypeVar("R_co", covariant=True)
P = ParamSpec("P")
S = TypeVar("S")

if TYPE_CHECKING:
    from typing_extensions import override
else:

    def override(f: Callable[P, R]) -> Callable[P, R]:
        return f

    override.__module__ = "typing_extensions"

Args = tuple[object, ...]
Kwargs = dict[str, object]
Mode = Literal["auto", "class", "static"]
OwnerKind = Literal["none", "instance", "class"]
BoundOwnerKind = Literal["instance", "class"]
CacheKey = object


class DoNotCache(Exception):
    value: object

    def __init__(self, value: object) -> None:
        super().__init__()
        self.value = value


class _InFlight(Generic[R]):
    __slots__ = ("event", "value", "exception", "do_not_cache")

    event: threading.Event
    value: Optional[R]
    exception: Optional[BaseException]
    do_not_cache: bool

    def __init__(self) -> None:
        self.event = threading.Event()
        self.value = None
        self.exception = None
        self.do_not_cache = False


class _ScopeState(Generic[R]):
    __slots__ = (
        "lock",
        "values",
        "in_flight",
        "singleton_set",
        "singleton_value",
        "singleton_in_flight",
    )

    values: dict[CacheKey, R]
    in_flight: dict[CacheKey, _InFlight[R]]
    singleton_set: bool
    singleton_value: Optional[R]
    singleton_in_flight: Optional[_InFlight[R]]

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.values = {}
        self.in_flight = {}
        self.singleton_set = False
        self.singleton_value = None
        self.singleton_in_flight = None


def _signature_without_first_arg(signature: inspect.Signature) -> inspect.Signature:
    params = list(signature.parameters.values())
    if not params:
        return signature
    return signature.replace(parameters=params[1:])


def _signatures_match(expected: inspect.Signature, actual: inspect.Signature) -> bool:
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


def _make_canonical_key(
    signature: inspect.Signature, args: Args, kwargs: Kwargs
) -> tuple[tuple[str, object], ...]:
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()

    items: list[tuple[str, object]] = []
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


class _BoundCachedCallable(Generic[P, R]):
    def __init__(
        self, cached: _CachedCallable[..., R], owner_kind: BoundOwnerKind, owner: object
    ) -> None:
        self._cached = cached
        self._owner_kind = owner_kind
        self._owner = owner
        functools.update_wrapper(self, cached._func)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._cached._invoke(
            self._owner_kind, self._owner, cast(Args, args), cast(Kwargs, kwargs)
        )

    def clear(self) -> None:
        self._cached._clear(self._owner_kind, self._owner)

    def remove(self, *args: P.args, **kwargs: P.kwargs) -> None:
        self._cached._remove(
            self._owner_kind, self._owner, cast(Args, args), cast(Kwargs, kwargs)
        )


class _CachedCallable(Generic[P, R]):
    def __init__(
        self,
        func: Callable[P, R],
        key: Optional[Callable[..., CacheKey]] = None,
        mode: Mode = "auto",
    ) -> None:
        functools.update_wrapper(self, func)
        self._func = func
        self._key_func = key
        self._mode = mode
        self._declaring_owner: Optional[type[object]] = None

        self._signature = inspect.signature(func)
        self._method_signature = _signature_without_first_arg(self._signature)
        self._is_singleton_full = len(self._signature.parameters) == 0
        self._is_singleton_method = len(self._method_signature.parameters) == 0

        self._global_state: _ScopeState[R] = _ScopeState()
        # WeakKeyDictionary hashes its weak references using the referent's hash.
        # Index by object id instead, then verify identity on every lookup.
        self._instance_states: dict[
            int, tuple[weakref.ReferenceType[object], _ScopeState[R]]
        ] = {}
        self._class_states: dict[
            int, tuple[weakref.ReferenceType[object], _ScopeState[R]]
        ] = {}
        self._state_lock = threading.RLock()

        self._key_signature: Optional[inspect.Signature] = None
        if self._key_func is not None:
            if not callable(self._key_func):
                raise TypeError("key must be callable")
            self._key_signature = inspect.signature(self._key_func)
            self._validate_key_signature()

    def __set_name__(self, owner: type[object], name: str) -> None:
        self._declaring_owner = owner
        if self._key_signature is not None and self._mode == "auto":
            if not _signatures_match(self._method_signature, self._key_signature):
                raise TypeError(
                    "key signature must match the decorated method signature "
                    "excluding self/cls"
                )

    @overload
    def __get__(
        self, obj: None, owner: Optional[type[object]]
    ) -> Union[_CachedCallable[..., R], _BoundCachedCallable[..., R]]: ...

    @overload
    def __get__(
        self, obj: object, owner: Optional[type[object]]
    ) -> _BoundCachedCallable[..., R]: ...

    def __get__(
        self, obj: Optional[object], owner: Optional[type[object]]
    ) -> Union[_CachedCallable[..., R], _BoundCachedCallable[..., R]]:
        if self._mode == "class":
            return _BoundCachedCallable(self, "class", owner)
        if self._mode == "static":
            return self
        if obj is None:
            return self
        return _BoundCachedCallable(self, "instance", obj)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        if self._mode == "class":
            if not args:
                raise TypeError("missing class argument for cached classmethod")
            return self._invoke(
                "class", args[0], args[1:], cast(Kwargs, kwargs)
            )

        if (
            self._mode == "auto"
            and self._declaring_owner is not None
            and args
            and isinstance(args[0], self._declaring_owner)
        ):
            return self._invoke(
                "instance", args[0], args[1:], cast(Kwargs, kwargs)
            )

        return self._invoke("none", None, cast(Args, args), cast(Kwargs, kwargs))

    def clear(self) -> None:
        self._clear("none", None)

    def remove(self, *args: P.args, **kwargs: P.kwargs) -> None:
        self._remove("none", None, cast(Args, args), cast(Kwargs, kwargs))

    def _validate_key_signature(self) -> None:
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

    def _state_for(
        self, owner_kind: OwnerKind, owner: Optional[object], create: bool = True
    ) -> Optional[_ScopeState[R]]:
        if owner_kind == "none":
            return self._global_state

        mapping = (
            self._instance_states if owner_kind == "instance" else self._class_states
        )
        assert owner is not None
        owner_id = id(owner)

        with self._state_lock:
            entry = mapping.get(owner_id)
            if entry is not None:
                if entry[0]() is owner:
                    return entry[1]
                # A dead entry can only remain until its weakref callback runs. Remove
                # it here as well so a reused object id cannot inherit stale state.
                mapping.pop(owner_id, None)

            if not create:
                return None

            try:
                owner_ref = weakref.ref(
                    owner,
                    lambda ref: self._discard_state(mapping, owner_id, ref),
                )
            except TypeError as exc:
                raise TypeError(
                    "cache owner must be weak-referenceable to preserve non-leaking semantics"
                ) from exc

            state: _ScopeState[R] = _ScopeState()
            mapping[owner_id] = (owner_ref, state)
            return state

    def _discard_state(
        self,
        mapping: dict[int, tuple[weakref.ReferenceType[object], _ScopeState[R]]],
        owner_id: int,
        owner_ref: weakref.ReferenceType[object],
    ) -> None:
        with self._state_lock:
            entry = mapping.get(owner_id)
            if entry is not None and entry[0] is owner_ref:
                mapping.pop(owner_id)

    def _signature_for(self, owner_kind: OwnerKind) -> inspect.Signature:
        if owner_kind in ("instance", "class"):
            return self._method_signature
        return self._signature

    def _is_singleton_for(self, owner_kind: OwnerKind) -> bool:
        if owner_kind in ("instance", "class"):
            return self._is_singleton_method
        return self._is_singleton_full

    def _compute_cache_key(
        self, owner_kind: OwnerKind, args: Args, kwargs: Kwargs
    ) -> CacheKey:
        if self._key_func is not None:
            return self._key_func(*args, **kwargs)
        signature = self._signature_for(owner_kind)
        return _make_canonical_key(signature, args, kwargs)

    def _invoke(
        self, owner_kind: OwnerKind, owner: Optional[object], args: Args, kwargs: Kwargs
    ) -> R:
        state = self._state_for(owner_kind, owner, create=True)
        assert state is not None
        call_signature = self._signature_for(owner_kind)
        call_signature.bind(*args, **kwargs)

        if self._is_singleton_for(owner_kind):
            return self._invoke_singleton(state, lambda: self._call_underlying(owner, args, kwargs))

        key = self._compute_cache_key(owner_kind, args, kwargs)
        return self._invoke_keyed(
            state, key, lambda: self._call_underlying(owner, args, kwargs)
        )

    def _call_underlying(
        self, owner: Optional[object], args: Args, kwargs: Kwargs
    ) -> R:
        func = cast(Callable[..., R], self._func)
        if owner is None:
            return func(*args, **kwargs)
        return func(owner, *args, **kwargs)

    def _invoke_singleton(self, state: _ScopeState[R], compute: Callable[[], R]) -> R:
        while True:
            with state.lock:
                if state.singleton_set:
                    return cast(R, state.singleton_value)

                in_flight = state.singleton_in_flight
                if in_flight is None:
                    in_flight = _InFlight[R]()
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
                        in_flight.value = cast(R, exc.value)
                        in_flight.event.set()
                    return cast(R, exc.value)
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
            return cast(R, in_flight.value)

    def _invoke_keyed(
        self, state: _ScopeState[R], key: CacheKey, compute: Callable[[], R]
    ) -> R:
        while True:
            with state.lock:
                if key in state.values:
                    return state.values[key]

                in_flight = state.in_flight.get(key)
                if in_flight is None:
                    in_flight = _InFlight[R]()
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
                        in_flight.value = cast(R, exc.value)
                        in_flight.event.set()
                    return cast(R, exc.value)
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
            return cast(R, in_flight.value)

    def _clear(self, owner_kind: OwnerKind, owner: Optional[object]) -> None:
        if owner_kind == "none":
            states: list[_ScopeState[R]] = [self._global_state]
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

    def _remove(
        self, owner_kind: OwnerKind, owner: Optional[object], args: Args, kwargs: Kwargs
    ) -> None:
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


class _PropertyLike(Protocol[R_co]):
    @property
    def fget(self) -> Optional[Callable[..., R_co]]: ...


class _CachedProperty(property, Generic[R_co]):
    @override
    def __init__(
        self,
        fget: Optional[Callable[..., R_co]] = None,
        fset: Optional[Callable[..., None]] = None,
        fdel: Optional[Callable[..., None]] = None,
        doc: Optional[str] = None,
        *,
        key: Optional[Callable[..., CacheKey]] = None,
    ) -> None:
        if fget is None:
            raise TypeError("cache cannot decorate a property without a getter")

        if isinstance(fget, _CachedCallable):
            self._cached = fget
        else:
            self._cached = _CachedCallable(fget, key=key)
        super().__init__(self._cached, fset, fdel, doc)

    def __set_name__(self, owner: type[object], name: str) -> None:
        self._cached.__set_name__(owner, name)

    @override
    def __set__(self, instance: object, value: object) -> None:
        super().__set__(instance, value)
        self.clear(instance)

    @override
    def __delete__(self, instance: object) -> None:
        super().__delete__(instance)
        self.clear(instance)

    @override
    def getter(self, fget: Callable[..., S]) -> _CachedProperty[S]:
        return _CachedProperty(
            fget,
            self.fset,
            self.fdel,
            self.__doc__,
            key=self._cached._key_func,
        )

    @override
    def setter(self, fset: Callable[..., None]) -> _CachedProperty[R_co]:
        return _CachedProperty(self._cached, fset, self.fdel, self.__doc__)

    @override
    def deleter(self, fdel: Callable[..., None]) -> _CachedProperty[R_co]:
        return _CachedProperty(self._cached, self.fset, fdel, self.__doc__)

    def clear(self, instance: object) -> None:
        self._cached._clear("instance", instance)


class _CacheDecorator(Protocol):
    @overload
    def __call__(self, f: _PropertyLike[R]) -> _CachedProperty[R]: ...

    @overload
    def __call__(self, f: classmethod[object, P, R]) -> _CachedCallable[..., R]: ...

    @overload
    def __call__(self, f: staticmethod[P, R]) -> _CachedCallable[P, R]: ...

    @overload
    def __call__(self, f: Callable[P, R]) -> _CachedCallable[P, R]: ...


@overload
def cache(
    f: _PropertyLike[R],
    *,
    key: Optional[Callable[..., CacheKey]] = None,
) -> _CachedProperty[R]: ...


@overload
def cache(
    f: classmethod[object, P, R],
    *,
    key: Optional[Callable[..., CacheKey]] = None,
) -> _CachedCallable[..., R]: ...


@overload
def cache(
    f: staticmethod[P, R], *, key: Optional[Callable[..., CacheKey]] = None
) -> _CachedCallable[P, R]: ...


@overload
def cache(
    f: Callable[P, R], *, key: Optional[Callable[..., CacheKey]] = None
) -> _CachedCallable[P, R]: ...


@overload
def cache(
    f: None = None, *, key: Optional[Callable[..., CacheKey]] = None
) -> _CacheDecorator: ...


def cache(
    f: Optional[object] = None, *, key: Optional[Callable[..., CacheKey]] = None
) -> object:
    """
    Cache the results of the decorated function.

    - Automatically optimizes to singleton implementation when decorating a function with no arguments.
    - When decorating an instance method, creates a separate cache per instance and excludes `self` from
      generated keys. The cache does not prevent the instance from being garbage collection, and the cache
      can be freed after the instance.
    - When decorating a classmethod, creates a separate cache per subclass with the same semantics as when
      decorating an instance method.
    - When decorating a property, `cache` must be the outer decorator. The getter uses a separate cache per
      instance, and a successful setter or deleter call clears that instance's cached value.
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

        def decorator(func: object) -> object:
            return cache(cast(Callable[..., object], func), key=key)

        return decorator

    if isinstance(f, property):
        return _CachedProperty(f.fget, f.fset, f.fdel, f.__doc__, key=key)

    mode: Mode = "auto"
    if isinstance(f, classmethod):
        wrapped = _CachedCallable(f.__func__, key=key, mode="class")
        return wrapped

    if isinstance(f, staticmethod):
        wrapped = _CachedCallable(f.__func__, key=key, mode="static")
        return wrapped

    if not callable(f):
        raise TypeError("cache can only decorate callables")

    return _CachedCallable(f, key=key, mode=mode)
