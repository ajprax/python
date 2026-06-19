from __future__ import annotations

import itertools
import time
from collections import defaultdict, deque
from operator import itemgetter
from typing import (
    AbstractSet,
    Callable,
    Collection,
    Dict as TypingDict,
    Generic,
    Iterable,
    Iterator,
    List as TypingList,
    Optional,
    Set as TypingSet,
    TYPE_CHECKING,
    Tuple as TypingTuple,
    Type,
    TypeVar,
    Union,
    SupportsIndex,
    cast,
    overload,
)

from ajprax.hof import identity, t
from ajprax.require import require
from ajprax.sentinel import Unset

K = TypeVar("K")
K_co = TypeVar("K_co", covariant=True)
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
U = TypeVar("U")
V = TypeVar("V")
V_co = TypeVar("V_co", covariant=True)
W = TypeVar("W")

if TYPE_CHECKING:
    from typing_extensions import override
else:

    def override(f: T) -> T:
        return f

    override.__module__ = "typing_extensions"


def call(f: Callable[[], T]) -> T:
    return f()


call.__module__ = "operator"

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

dict_keys = type({}.keys())
dict_values = type({}.values())


@overload
def count() -> Iter[int]: ...


@overload
def count(start: int) -> Iter[int]: ...


@overload
def count(start: float) -> Iter[float]: ...


@overload
def count(start: complex) -> Iter[complex]: ...


@overload
def count(start: int, step: int) -> Iter[int]: ...


@overload
def count(start: int, step: float) -> Iter[float]: ...


@overload
def count(start: float, step: Union[int, float]) -> Iter[float]: ...


@overload
def count(start: Union[int, float], step: complex) -> Iter[complex]: ...


@overload
def count(start: complex, step: Union[int, float, complex]) -> Iter[complex]: ...


def count(
    start: Union[int, float, complex] = 0,
    step: Union[int, float, complex] = 1,
) -> Iter[Union[int, float, complex]]:
    return Iter(itertools.count(start, step))  # pyrefly: ignore[bad-specialization, bad-argument-type]


def repeated(item: T, n: Union[int, Type[Unset]] = Unset) -> Iter[T]:
    if n is Unset:
        return Iter(itertools.repeat(item))
    return Iter(itertools.repeat(item, cast(int, n)))


def repeatedly(f: Callable[[], T], n: Union[int, Type[Unset]] = Unset) -> Iter[T]:
    return repeated(f, n).map(call)


def timestamp(clock: Callable[[], U] = time.time) -> Callable[[T], tuple[U, T]]:
    def inner(item: T) -> tuple[U, T]:
        return clock(), item

    return inner


@overload
def wrap(it: DefaultDict[K, V]) -> DefaultDict[K, V]: ...


@overload
def wrap(it: Dict[K, V]) -> Dict[K, V]: ...


@overload
def wrap(it: DictKeys[K]) -> DictKeys[K]: ...


@overload
def wrap(it: DictValues[V]) -> DictValues[V]: ...


@overload
def wrap(it: Iter[T]) -> Iter[T]: ...


@overload
def wrap(it: List[T]) -> List[T]: ...


@overload
def wrap(it: Range) -> Range: ...


@overload
def wrap(it: Set[T]) -> Set[T]: ...


@overload
def wrap(it: Tuple[T]) -> Tuple[T]: ...


@overload
def wrap(it: defaultdict[K, V]) -> DefaultDict[K, V]: ...


@overload
def wrap(it: TypingDict[K, V]) -> Dict[K, V]: ...


@overload
def wrap(it: TypingList[T]) -> List[T]: ...


@overload
def wrap(it: TypingSet[T]) -> Set[T]: ...


@overload
def wrap(it: range) -> Range: ...


@overload
def wrap(it: TypingTuple[T, ...]) -> Tuple[T]: ...


@overload
def wrap(it: Iterable[T]) -> Iter[T]: ...


@overload
def wrap(it: object) -> object: ...


def wrap(it: object) -> object:
    wrappers: TypingDict[type[object], Callable[[object], object]] = {
        DefaultDict: cast(Callable[[object], object], identity),
        Dict: cast(Callable[[object], object], identity),
        DictKeys: cast(Callable[[object], object], identity),
        DictValues: cast(Callable[[object], object], identity),
        Iter: cast(Callable[[object], object], identity),
        List: cast(Callable[[object], object], identity),
        Range: cast(Callable[[object], object], identity),
        Set: cast(Callable[[object], object], identity),
        Tuple: cast(Callable[[object], object], identity),
        defaultdict: lambda value: DefaultDict(
            cast(defaultdict[object, object], value).default_factory
        ).update(cast(defaultdict[object, object], value)),
        dict: lambda value: Dict(cast(TypingDict[object, object], value)),
        dict_keys: lambda value: DictKeys(cast(Collection[object], value)),
        dict_values: lambda value: DictValues(cast(Collection[object], value)),
        list: lambda value: List(cast(TypingList[object], value)),
        set: lambda value: Set(cast(TypingSet[object], value)),
        range: lambda value: Range(cast(range, value)),
        tuple: lambda value: Tuple(cast(TypingTuple[object, ...], value)),
    }
    wrapper = wrappers.get(type(it))
    if wrapper is not None:
        return wrapper(it)
    return Iter(cast(Iterable[object], it))


class Dict(TypingDict[K, V]):
    @override
    def __iter__(self) -> Iter[K]:
        return Iter(dict.__iter__(self))

    def all(self, key: Union[Callable[[tuple[K, V]], object], Type[Unset]] = Unset):
        return self.items().all(key=key)

    def any(self, key: Union[Callable[[tuple[K, V]], object], Type[Unset]] = Unset):
        return self.items().any(key=key)

    def apply(self, f: Callable[[Dict[K, V]], U]):
        return f(self)

    def apply_and_wrap(self, f: Callable[[Dict[K, V]], object]):
        return wrap(f(self))

    def batch(
        self,
        size: Union[int, float],
        weight: Union[Callable[[tuple[K, V]], Union[int, float]], Type[Unset]] = Unset,
        strict: bool = False,
    ):
        return self.items().batch(size, weight=weight, strict=strict).map(Dict)

    def chain(self, *its: Iterable[object]):
        return self.items().chain(*its)

    def clear(self):  # pyrefly: ignore[bad-override, missing-override-decorator]
        dict.clear(self)
        return self

    def combinations(self, r: int, with_replacement: bool = False):
        return self.items().combinations(r, with_replacement=with_replacement).map(Dict)

    def combine_if(self, condition: object, combinator: str, *a: object, **kw: object):
        if condition:
            return getattr(self, combinator)(*a, **kw)
        return self

    @override
    def copy(self) -> Dict[K, V]:
        return Dict(dict.copy(self))

    # key is required because items are guaranteed unique
    def counts(self, key: Callable[[tuple[K, V]], U]) -> Dict[U, int]:
        return self.items().counts(key=key)

    def cycle(self):
        return self.items().cycle()

    def default_dict(self, default_factory: Callable[[], V]) -> DefaultDict[K, V]:
        return DefaultDict(default_factory).update(self)

    def dict(self) -> Dict[K, V]:
        return self

    # key is required because items are guaranteed unique
    def distinct(self, key: Callable[[tuple[K, V]], object]):
        return self.items().distinct(key=key).dict()

    def distinct_values(self, key: Union[Callable[[V], object], Type[Unset]] = Unset):
        if key is Unset:
            return self.distinct(itemgetter(1))
        return self.distinct(t(lambda k, v: key(v)))

    def do(self, f: Callable[[tuple[K, V]], object]):
        self.for_each(f)
        return self

    def drop(
        self,
        n: Union[int, float],
        weight: Union[Callable[[tuple[K, V]], Union[int, float]], Type[Unset]] = Unset,
    ):
        return self.items().drop(n, weight=weight).dict()

    def drop_while(self, predicate: Callable[[tuple[K, V]], object]):
        return self.items().drop_while(predicate).dict()

    def enumerate(self, start: int = 0):
        return self.items().enumerate(start=start)

    def filter(self, predicate: Callable[[tuple[K, V]], object] = bool):
        # predicate is required because 2-tuples are always truthy
        return self.items().filter(predicate=predicate).dict()

    def filter_keys(self, predicate: Callable[[K], object] = bool):
        return self.filter(predicate=t(lambda k, v: predicate(k))).dict()

    def filter_values(self, predicate: Callable[[V], object] = bool):
        return self.filter(predicate=t(lambda k, v: predicate(v))).dict()

    def first(
        self,
        predicate: Union[Callable[[tuple[K, V]], object], Type[Unset]] = Unset,
        default: object = Unset,
    ):
        return self.items().first(predicate=predicate, default=default)

    def flat_map(self, f: Callable[[tuple[K, V]], Iterable[tuple[U, W]]]):
        return self.items().flat_map(f).dict()

    def flat_map_keys(self, f: Callable[[K], Iterable[U]]):
        return self.items().flat_map(t(lambda k, v: ((k, v) for k in f(k)))).dict()

    def flat_map_values(self, f: Callable[[V], Iterable[U]]):
        return self.items().flat_map(t(lambda k, v: ((k, v) for v in f(v)))).dict()

    def fold(self, initial: U, f: Callable[[U, tuple[K, V]], U]):
        return self.items().fold(initial, f)

    def fold_while(
        self,
        initial: U,
        f: Callable[[U, tuple[K, V]], U],
        predicate: Callable[[U], object],
    ):
        return self.items().fold_while(initial, f, predicate)

    def for_each(self, f: Callable[[tuple[K, V]], object]):
        self.items().for_each(f)

    def group_by(
        self, key: Union[Callable[[tuple[K, V]], object], Type[Unset]] = Unset
    ):
        return self.items().group_by(key=key)

    def intersection(self, *others: TypingDict[K, V]):
        dicts = (self, *others)
        keys = set(self)
        for other in others:
            keys.intersection_update(other)
        out = Dict()
        for key in keys:
            for d in dicts:
                value = d[key]
                # TODO: should we instead skip keys that don't match?
                require(
                    out.setdefault(key, value) == value, "mismatched values", key=key
                )
        return out

    def intersperse(self, item: object):
        return self.items().intersperse(item)

    def invert(self):
        return self.items().map(lambda item: (item[1], item[0])).dict()

    def items(self) -> Iter[tuple[K, V]]:  # pyrefly: ignore[bad-override, missing-override-decorator]
        return Iter(dict.items(self))

    def keys(self) -> DictKeys[K]:  # pyrefly: ignore[bad-override, missing-override-decorator]
        return DictKeys(dict.keys(self))

    def last(
        self,
        predicate: Union[Callable[[tuple[K, V]], object], Type[Unset]] = Unset,
        default: object = Unset,
    ):
        return self.items().last(predicate=predicate, default=default)

    def list(self) -> List[tuple[K, V]]:
        return List(self.items())

    def map(self, f: Callable[[tuple[K, V]], tuple[U, W]]):
        return self.items().map(f).dict()

    def map_keys(self, f: Callable[[K], U]):
        return self.items().map(t(lambda k, v: (f(k), v))).dict()

    def map_values(self, f: Callable[[V], U]):
        return self.items().map(t(lambda k, v: (k, f(v)))).dict()

    def max(
        self,
        key: Optional[Callable[[tuple[K, V]], object]] = None,
        default: object = Unset,
    ):
        return self.items().max(key=key, default=default)

    def min(
        self,
        key: Optional[Callable[[tuple[K, V]], object]] = None,
        default: object = Unset,
    ):
        return self.items().min(key=key, default=default)

    def min_max(
        self,
        key: Optional[Callable[[tuple[K, V]], object]] = None,
        default: object = Unset,
    ):
        return self.items().min_max(key=key, default=default)

    def only(
        self,
        predicate: Union[Callable[[tuple[K, V]], object], Type[Unset]] = Unset,
        empty_default: object = Unset,
        overfull_default: object = Unset,
    ):
        return self.items().only(
            predicate=predicate,
            empty_default=empty_default,
            overfull_default=overfull_default,
        )

    def partition(self, predicate: Callable[[tuple[K, V]], object] = identity):
        return wrap(self.items().partition(predicate=predicate)).map(Dict)

    def permutations(self, r: Optional[int] = None):
        return self.items().permutations(r=r)

    def powerset(self):
        return self.items().powerset().map(Dict)

    def product(self, *its: Iterable[object], repeat: int = 1):
        return self.items().product(*its, repeat=repeat)

    def put(self, k: K, v: V):
        self[k] = v
        return self

    def repeat(self, n: Union[int, Type[Unset]] = Unset):
        return self.items().repeat(n=n)

    def set(self) -> Set[tuple[K, V]]:
        return Set(self.items())

    def size(self) -> int:
        return len(self)

    def sliding(self, size: int, step: int = 1):
        return self.items().sliding(size, step=step).map(Dict)

    def sliding_by_timestamp(
        self,
        size: float,
        step: float = 1,
        stamp: Callable[[tuple[K, V]], tuple[float, tuple[K, V]]] = timestamp(
            time.time
        ),
    ):
        return self.items().sliding_by_timestamp(size, step=step, stamp=stamp).map(Dict)

    def take(
        self,
        n: Union[int, float],
        weight: Union[Callable[[tuple[K, V]], Union[int, float]], Type[Unset]] = Unset,
    ):
        return self.items().take(n, weight=weight).dict()

    def take_while(
        self, predicate: Union[Callable[[tuple[K, V]], object], Type[Unset]] = Unset
    ):
        return self.items().take_while(predicate=predicate).dict()

    def timestamp(self, clock: Callable[[], U] = time.time):
        return self.items().timestamp(clock=clock)

    if _tqdm is not None:

        def tqdm(self, *a: object, **kw: object):
            return self.items().tqdm(*a, **kw)

    def tuple(self) -> Tuple[tuple[K, V]]:
        return self.items().tuple()

    def union(self, *others: TypingDict[K, V]):
        out = self.dict()
        for other in others:
            for key, value in other.items():
                require(
                    out.setdefault(key, value) == value, "mismatched values", key=key
                )
        return out

    def update(  # pyrefly: ignore[bad-override, missing-override-decorator]
        self,
        E: Union[TypingDict[K, V], Iterable[TypingTuple[K, V]], Type[Unset]] = Unset,
        **F: V,
    ):
        if E is Unset:
            dict.update(self, **F)  # pyrefly: ignore[no-matching-overload]
        else:
            dict.update(self, E, **F)  # pyrefly: ignore[no-matching-overload]
        return self

    def values(self) -> DictValues[V]:  # pyrefly: ignore[bad-override, missing-override-decorator]
        return DictValues(dict.values(self))

    def zip(self, *others: Iterable[object], strict: bool = False):
        return self.items().zip(*others, strict=strict)

    def zip_longest(self, *others: Iterable[object], fillvalue: object = None):
        return self.items().zip_longest(*others, fillvalue=fillvalue)


class DefaultDict(Dict[K, V], defaultdict[K, V]):
    pass


class DictKeys(Generic[K_co]):
    def __init__(self, keys: Collection[K_co]):
        self._keys = keys

    def __eq__(self, other: object):
        if isinstance(other, DictKeys):
            other = other._keys
        return self._keys == other

    def __len__(self) -> int:
        return len(self._keys)

    def __repr__(self):
        return repr(self._keys)

    def __iter__(self) -> Iter[K_co]:
        return Iter(self._keys)

    def all(self, key: Union[Callable[[K_co], object], Type[Unset]] = Unset):
        return self.iter().all(key=key)

    def any(self, key: Union[Callable[[K_co], object], Type[Unset]] = Unset):
        return self.iter().any(key=key)

    def apply(self, f: Callable[[DictKeys[K_co]], U]):
        return f(self)

    def apply_and_wrap(self, f: Callable[[DictKeys[K_co]], object]):
        return wrap(f(self))

    def batch(
        self,
        size: Union[int, float],
        weight: Union[Callable[[K_co], Union[int, float]], Type[Unset]] = Unset,
        strict: bool = False,
    ):
        return self.iter().batch(size, weight=weight, strict=strict).map(Set)

    def chain(self, *its: Iterable[object]):
        return self.iter().chain(*its)

    def combinations(self, r: int, with_replacement: bool = False):
        return self.iter().combinations(r, with_replacement=with_replacement)

    def combine_if(self, condition: object, combinator: str, *a: object, **kw: object):
        if condition:
            return getattr(self, combinator)(*a, **kw)
        return self

    @overload
    def counts(self) -> Dict[K_co, int]: ...

    @overload
    def counts(self, key: Callable[[K_co], U]) -> Dict[U, int]: ...

    def counts(self, key: Union[Callable[[K_co], object], Type[Unset]] = Unset):
        if key is Unset:
            return self.iter().counts()
        return self.iter().counts(key=key)

    def cycle(self):
        return self.iter().cycle()

    def default_dict(
        self: DictKeys[tuple[K, V]], default_factory: Callable[[], V]
    ) -> DefaultDict[K, V]:
        return DefaultDict(default_factory).update(self)

    def dict(self: DictKeys[tuple[K, V]]) -> Dict[K, V]:
        return Dict(self)

    def distinct(self, key: Union[Callable[[K_co], object], Type[Unset]] = Unset):
        return self.iter().distinct(key=key).set()

    def do(self, f: Callable[[K_co], object]):
        self.for_each(f)
        return self

    def drop(
        self,
        n: Union[int, float],
        weight: Union[Callable[[K_co], Union[int, float]], Type[Unset]] = Unset,
    ):
        return self.iter().drop(n, weight=weight).set()

    def drop_while(self, predicate: Callable[[K_co], object]):
        return self.iter().drop_while(predicate).set()

    def enumerate(self, start: int = 0):
        return self.iter().enumerate(start=start)

    def filter(self, predicate: Callable[[K_co], object] = bool):
        return self.iter().filter(predicate=predicate).set()

    def first(
        self,
        predicate: Union[Callable[[K_co], object], Type[Unset]] = Unset,
        default: object = Unset,
    ):
        return self.iter().first(predicate=predicate, default=default)

    def flat_map(self, f: Callable[[K_co], Iterable[U]]):
        return self.iter().flat_map(f).set()

    def flatten(self: DictKeys[Iterable[U]]) -> Set[U]:
        return self.iter().flatten().set()

    def fold(self, initial: U, f: Callable[[U, K_co], U]):
        return self.iter().fold(initial, f)

    def fold_while(
        self, initial: U, f: Callable[[U, K_co], U], predicate: Callable[[U], object]
    ):
        return self.iter().fold_while(initial, f, predicate)

    def for_each(self, f: Callable[[K_co], object]):
        return self.iter().for_each(f)

    def group_by(self, key: Union[Callable[[K_co], object], Type[Unset]] = Unset):
        return self.iter().group_by(key=key)

    def iter(self) -> Iter[K_co]:
        return iter(self)

    def last(
        self,
        predicate: Union[Callable[[K_co], object], Type[Unset]] = Unset,
        default: object = Unset,
    ):
        return self.iter().last(predicate=predicate, default=default)

    def list(self) -> List[K_co]:
        return List(self)

    def map(self, f: Callable[[K_co], U]):
        return self.iter().map(f).set()

    def map_to_keys(self, f: Callable[[K_co], U]):
        return self.iter().map_to_keys(f)

    def map_to_pairs(self, f: Callable[[K_co], U]):
        return self.iter().map_to_pairs(f).set()

    def map_to_values(self, f: Callable[[K_co], U]):
        return self.iter().map_to_values(f)

    def max(
        self, key: Optional[Callable[[K_co], object]] = None, default: object = Unset
    ):
        return self.iter().max(key=key, default=default)

    def min(
        self, key: Optional[Callable[[K_co], object]] = None, default: object = Unset
    ):
        return self.iter().min(key=key, default=default)

    def min_max(
        self, key: Optional[Callable[[K_co], object]] = None, default: object = Unset
    ):
        return self.iter().min_max(key=key, default=default)

    def only(
        self,
        predicate: Union[Callable[[K_co], object], Type[Unset]] = Unset,
        empty_default: object = Unset,
        overfull_default: object = Unset,
    ):
        return self.iter().only(
            predicate=predicate,
            empty_default=empty_default,
            overfull_default=overfull_default,
        )

    def partition(
        self, predicate: Union[Callable[[K_co], object], Type[Unset]] = Unset
    ):
        return wrap(self.iter().partition(predicate=predicate)).map(Set)

    def permutations(self, r: Optional[int] = None):
        return self.iter().permutations(r=r)

    def powerset(self):
        return self.iter().powerset()

    def product(self, *its: Iterable[object], repeat: int = 1):
        return self.iter().product(*its, repeat=repeat)

    def reduce(self, f: Callable[[K_co, K_co], K_co]):
        return self.iter().reduce(f)

    def repeat(self, n: Union[int, Type[Unset]] = Unset):
        return self.iter().repeat(n=n)

    def set(self) -> Set[K_co]:
        return Set(self)

    def size(self) -> int:
        return len(self)

    def sliding(self, size: int, step: int = 1):
        return self.iter().sliding(size, step=step).map(Set)

    def sliding_by_timestamp(
        self,
        size: float,
        step: float = 1,
        stamp: Callable[[K_co], tuple[float, K_co]] = timestamp(time.time),
    ):
        return self.iter().sliding_by_timestamp(size, step=step, stamp=stamp).map(Set)

    def take(
        self,
        n: Union[int, float],
        weight: Union[Callable[[K_co], Union[int, float]], Type[Unset]] = Unset,
    ):
        return self.iter().take(n, weight=weight).set()

    def take_while(
        self, predicate: Union[Callable[[K_co], object], Type[Unset]] = Unset
    ):
        return self.iter().take_while(predicate=predicate).set()

    def timestamp(self, clock: Callable[[], U] = time.time):
        return self.iter().timestamp(clock=clock).set()

    if _tqdm is not None:

        def tqdm(self, *a: object, **kw: object):
            return self.iter().tqdm(*a, **kw)

    def transpose(self):
        return self.iter().transpose()

    def tuple(self) -> Tuple[K_co]:
        return Tuple(self)

    unzip = transpose

    def zip(self, *others: Iterable[object], strict: bool = False):
        return self.iter().zip(*others, strict=strict).set()

    def zip_longest(self, *others: Iterable[object], fillvalue: object = None):
        return self.iter().zip_longest(*others, fillvalue=fillvalue).set()


class DictValues(Generic[V_co]):
    def __init__(self, values: Collection[V_co]):
        self._values = values

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self):
        return repr(self._values)

    def __iter__(self) -> Iter[V_co]:
        return Iter(self._values)

    def all(self, key: Union[Callable[[V_co], object], Type[Unset]] = Unset):
        return self.iter().all(key=key)

    def any(self, key: Union[Callable[[V_co], object], Type[Unset]] = Unset):
        return self.iter().any(key=key)

    def apply(self, f: Callable[[DictValues[V_co]], U]):
        return f(self)

    def apply_and_wrap(self, f: Callable[[DictValues[V_co]], object]):
        return wrap(f(self))

    def batch(
        self,
        size: Union[int, float],
        weight: Union[Callable[[V_co], Union[int, float]], Type[Unset]] = Unset,
        strict: bool = False,
    ):
        return self.iter().batch(size, weight=weight, strict=strict).tuple()

    def chain(self, *its: Iterable[object]):
        return self.iter().chain(*its)

    def combinations(self, r: int, with_replacement: bool = False):
        return self.iter().combinations(r, with_replacement=with_replacement)

    def combine_if(self, condition: object, combinator: str, *a: object, **kw: object):
        if condition:
            return getattr(self, combinator)(*a, **kw)
        return self

    @overload
    def counts(self) -> Dict[V_co, int]: ...

    @overload
    def counts(self, key: Callable[[V_co], U]) -> Dict[U, int]: ...

    def counts(self, key: Union[Callable[[V_co], object], Type[Unset]] = Unset):
        if key is Unset:
            return self.iter().counts()
        return self.iter().counts(key=key)

    def cycle(self):
        return self.iter().cycle()

    def default_dict(
        self: DictValues[tuple[K, V]], default_factory: Callable[[], V]
    ) -> DefaultDict[K, V]:
        return DefaultDict(default_factory).update(self)

    def dict(self: DictValues[tuple[K, V]]) -> Dict[K, V]:
        return Dict(self)

    def distinct(self, key: Union[Callable[[V_co], object], Type[Unset]] = Unset):
        return self.iter().distinct(key=key).tuple()

    def do(self, f: Callable[[V_co], object]):
        self.for_each(f)
        return self

    def drop(
        self,
        n: Union[int, float],
        weight: Union[Callable[[V_co], Union[int, float]], Type[Unset]] = Unset,
    ):
        return self.iter().drop(n, weight=weight)

    def drop_while(self, predicate: Callable[[V_co], object]):
        return self.iter().drop_while(predicate).tuple()

    def enumerate(self, start: int = 0):
        return self.iter().enumerate(start=start).tuple()

    def filter(self, predicate: Callable[[V_co], object] = bool):
        return self.iter().filter(predicate=predicate).tuple()

    def first(
        self,
        predicate: Union[Callable[[V_co], object], Type[Unset]] = Unset,
        default: object = Unset,
    ):
        return self.iter().first(predicate=predicate, default=default)

    def flat_map(self, f: Callable[[V_co], Iterable[U]]):
        return self.iter().flat_map(f).tuple()

    def flatten(self: DictValues[Iterable[U]]) -> Tuple[U]:
        return self.iter().flatten().tuple()

    def fold(self, initial: U, f: Callable[[U, V_co], U]):
        return self.iter().fold(initial, f)

    def fold_while(
        self, initial: U, f: Callable[[U, V_co], U], predicate: Callable[[U], object]
    ):
        return self.iter().fold_while(initial, f, predicate)

    def for_each(self, f: Callable[[V_co], object]):
        return self.iter().for_each(f)

    def group_by(self, key: Union[Callable[[V_co], object], Type[Unset]] = Unset):
        return self.iter().group_by(key=key)

    def intersperse(self, item: object):
        return self.iter().intersperse(item).tuple()

    def iter(self) -> Iter[V_co]:
        return Iter(self)

    def last(
        self,
        predicate: Union[Callable[[V_co], object], Type[Unset]] = Unset,
        default: object = Unset,
    ):
        return self.iter().last(predicate=predicate, default=default)

    def list(self) -> List[V_co]:
        return List(self)

    def map(self, f: Callable[[V_co], U]):
        return self.iter().map(f).tuple()

    def map_to_keys(self, f: Callable[[V_co], U]):
        return self.iter().map_to_keys(f)

    def map_to_pairs(self, f: Callable[[V_co], U]):
        return self.iter().map_to_pairs(f).tuple()

    def map_to_values(self, f: Callable[[V_co], U]):
        return self.iter().map_to_values(f)

    def max(
        self, key: Optional[Callable[[V_co], object]] = None, default: object = Unset
    ):
        return self.iter().max(key=key, default=default)

    def min(
        self, key: Optional[Callable[[V_co], object]] = None, default: object = Unset
    ):
        return self.iter().min(key=key, default=default)

    def min_max(
        self, key: Optional[Callable[[V_co], object]] = None, default: object = Unset
    ):
        return self.iter().min_max(key=key, default=default)

    def only(
        self,
        predicate: Union[Callable[[V_co], object], Type[Unset]] = Unset,
        empty_default: object = Unset,
        overfull_default: object = Unset,
    ):
        return self.iter().only(
            predicate=predicate,
            empty_default=empty_default,
            overfull_default=overfull_default,
        )

    def partition(self, predicate: Callable[[V_co], object] = identity):
        return wrap(self.iter().partition(predicate=predicate)).map(Tuple)

    def permutations(self, r: Optional[int] = None):
        return self.iter().permutations(r)

    def powerset(self):
        return self.iter().powerset()

    def product(self, *its: Iterable[object], repeat: int = 1):
        return self.iter().product(*its, repeat=repeat)

    def reduce(self, f: Callable[[V_co, V_co], V_co]):
        return self.iter().reduce(f)

    def repeat(self, n: Union[int, Type[Unset]] = Unset):
        return self.iter().repeat(n=n)

    def set(self) -> Set[V_co]:
        return Set(self)

    def size(self) -> int:
        return len(self)

    def sliding(self, size: int, step: int = 1):
        return self.iter().sliding(size, step=step).tuple()

    def sliding_by_timestamp(
        self,
        size: float,
        step: float = 1,
        stamp: Callable[[V_co], tuple[float, V_co]] = timestamp(time.time),
    ):
        return self.iter().sliding_by_timestamp(size, step=step, stamp=stamp).tuple()

    def sorted(
        self, key: Optional[Callable[[V_co], object]] = None, reverse: bool = False
    ):
        return Tuple(
            sorted(self, key=key, reverse=reverse)  # pyrefly: ignore[no-matching-overload]
        )

    def take(
        self,
        n: Union[int, float],
        weight: Union[Callable[[V_co], Union[int, float]], Type[Unset]] = Unset,
    ):
        return self.iter().take(n, weight=weight)

    def take_while(
        self, predicate: Union[Callable[[V_co], object], Type[Unset]] = Unset
    ):
        return self.iter().take_while(predicate=predicate).tuple()

    def timestamp(self, clock: Callable[[], U] = time.time):
        return self.iter().timestamp(clock=clock).tuple()

    if _tqdm is not None:

        def tqdm(self, *a: object, **kw: object):
            return self.iter().tqdm(*a, **kw)

    def transpose(self):
        return self.iter().transpose().tuple()

    def tuple(self) -> Tuple[V_co]:
        return Tuple(self)

    unzip = transpose

    def zip(self, *others: Iterable[object], strict: bool = False):
        return self.iter().zip(*others, strict=strict).tuple()

    def zip_longest(self, *others: Iterable[object], fillvalue: object = None):
        return self.iter().zip_longest(*others, fillvalue=fillvalue).tuple()


class Iter(Generic[T_co]):
    def __init__(self, it: Iterable[T_co] = ()) -> None:
        self._it = iter(it)
        self._peek: deque[T_co] = deque()

    def __add__(self, other: Iterable[U]):
        return self.chain(other)

    def __contains__(self, item: object) -> bool:
        for e in self:
            if e == item:
                return True
        return False

    def __iter__(self) -> Iterator[T_co]:
        while self._peek:
            yield self._peek.popleft()
        yield from self._it

    def __mul__(self, n: int):
        return self.repeat(n)

    def __next__(self) -> T_co:
        if self._peek:
            return self._peek.popleft()
        return next(self._it)

    def __radd__(self, other: Iterable[U]):
        return Iter(other).chain(self)

    def __rmul__(self, n: int):
        return self.repeat(n)

    def accumulate(
        self,
        f: Optional[Callable[[T_co, T_co], T_co]] = None,
        *,
        initial: Optional[T_co] = None,
    ):
        return Iter(itertools.accumulate(self, f, initial=initial))

    def all(self, key: Union[Callable[[T_co], object], Type[Unset]] = Unset):
        if key is not Unset:
            self = self.map(key)
        return all(self)

    def any(self, key: Union[Callable[[T_co], object], Type[Unset]] = Unset):
        if key is not Unset:
            self = self.map(key)
        return any(self)

    def apply(self, f: Callable[[Iter[T_co]], U]):
        return f(self)

    def apply_and_wrap(self, f: Callable[[Iter[T_co]], object]):
        return wrap(f(self))

    def batch(
        self,
        size: Union[int, float],
        weight: Union[Callable[[T_co], Union[int, float]], Type[Unset]] = Unset,
        strict: bool = False,
    ) -> Iter[Tuple[T_co]]:
        """
        :param size: Number of items (or total weight of items) in each batch.
        :param weight: Optional function to weigh items and batch by total weight.
        :param strict: If True, raise if an item's weight exceeds the batch size. Only allowed if weight is provided.
        """
        require(size > 0, size=size)
        if strict:
            require(
                weight is not Unset,
                "strict batch size has no meaning without weight function",
            )

        if weight is Unset:
            item_count = cast(int, size)
            batched = getattr(itertools, "batched", None)
            if batched is not None:
                typed_batched = cast(
                    Callable[[Iterable[T_co], int], Iterable[tuple[T_co, ...]]],
                    batched,
                )
                return Iter(typed_batched(self, item_count)).map(Tuple)
            return (
                Iter(
                    itertools.groupby(
                        self.enumerate(), key=t(lambda i, _: i // item_count)
                    )
                )
                .map(itemgetter(1))
                .map(lambda batch: map(itemgetter(1), batch))
                .map(Tuple)
            )

        weigh = cast(Callable[[T_co], Union[int, float]], weight)

        def gen():
            while self.has_next():
                batch = self.take(size, weight=weigh).tuple()
                if batch:
                    yield batch
                else:
                    require(
                        not strict,
                        "found single item heavier than batch size",
                        size=size,
                        weight=weigh,
                        item_weight=weigh(self.peek()),
                    )
                    # the next item is larger than the batch size, so it gets a whole batch to itself
                    yield Tuple((self.next(),))

        return Iter(gen())

    def chain(self, *its: Iterable[object]):
        return Iter(itertools.chain(self, *its))

    def combinations(self, r: int, with_replacement: bool = False):
        combinations = (
            itertools.combinations_with_replacement
            if with_replacement
            else itertools.combinations
        )
        return Iter(combinations(self, r))

    def combine_if(self, condition: object, combinator: str, *a: object, **kw: object):
        if condition:
            return getattr(self, combinator)(*a, **kw)
        return self

    @overload
    def counts(self) -> Dict[T_co, int]: ...

    @overload
    def counts(self, key: Callable[[T_co], U]) -> Dict[U, int]: ...

    def counts(
        self, key: Union[Callable[[T_co], object], Type[Unset]] = Unset
    ) -> Dict[object, int]:
        counts: Dict[object, int] = Dict()
        if key is Unset:
            keys = self
        else:
            keys = self.map(key)
        for k in keys:
            counts.setdefault(k, 0)
            counts[k] += 1
        return counts

    def cycle(self):
        return Iter(itertools.cycle(self))

    def default_dict(
        self: Iter[tuple[K, V]], default_factory: Callable[[], V]
    ) -> DefaultDict[K, V]:
        return DefaultDict(default_factory).update(self)

    def dict(self: Iter[tuple[K, V]]) -> Dict[K, V]:
        return Dict(self)

    def distinct(self, key: Union[Callable[[T_co], object], Type[Unset]] = Unset):
        if key is Unset:

            def gen():
                seen = set()
                for item in self:
                    if item not in seen:
                        seen.add(item)
                        yield item
        else:

            def gen():
                seen = set()
                for item in self:
                    itemk = key(item)
                    if itemk not in seen:
                        seen.add(itemk)
                        yield item

        return Iter(gen())

    def do(self, f: Callable[[T_co], object]):
        def gen():
            for item in self:
                f(item)
                yield item

        return Iter(gen())

    def drop(
        self,
        n: Union[int, float],
        weight: Union[Callable[[T_co], Union[int, float]], Type[Unset]] = Unset,
    ):
        if weight is Unset:
            item_count = cast(int, n)
            if n >= 0:
                for _ in range(item_count):
                    try:
                        self.next()
                    except StopIteration:
                        break
                return self
            else:
                window = deque(self.take(-item_count))
                while self.has_next():
                    window.popleft()
                    window.append(self.next())
                return Iter(window)
        else:
            require(n >= 0, n=n)
            weigh = cast(Callable[[T_co], Union[int, float]], weight)
            total = 0
            while self.has_next() and total < n:
                total += weigh(self.next())
            return self

    def drop_while(self, predicate: Callable[[T_co], object]):
        return Iter(itertools.dropwhile(predicate, self))

    def enumerate(self, start: int = 0):
        return Iter(enumerate(self, start))

    def filter(self, predicate: Callable[[T_co], object] = bool):
        def gen():
            for item in self:
                if predicate(item):
                    yield item

        return Iter(gen())

    @overload
    def first(
        self,
        predicate: Union[Callable[[T_co], object], Type[Unset]] = Unset,
        default: Type[Unset] = Unset,
    ) -> T_co: ...

    @overload
    def first(
        self,
        predicate: Union[Callable[[T_co], object], Type[Unset]],
        default: U,
    ) -> Union[T_co, U]: ...

    def first(
        self,
        predicate: Union[Callable[[T_co], object], Type[Unset]] = Unset,
        default: object = Unset,
    ):
        try:
            if predicate is Unset:
                return self.next()
            else:
                return self.filter(predicate=predicate).next()
        except StopIteration:
            if default is Unset:
                raise
            return default

    def flat_map(self, f: Callable[[T_co], Iterable[U]]):
        def gen():
            for item in self:
                yield from f(item)

        return Iter(gen())

    def flatten(self: Iter[Iterable[U]]) -> Iter[U]:
        return Iter(itertools.chain.from_iterable(self))

    def fold(self, initial: U, f: Callable[[U, T_co], U]):
        acc = initial
        for item in self:
            acc = f(acc, item)
        return acc

    def fold_while(
        self,
        initial: U,
        f: Callable[[U, T_co], U],
        predicate: Callable[[U], object],
    ):
        require(predicate(initial), "invalid initial value", ValueError)

        acc = initial
        while self.has_next():
            last, acc = acc, f(acc, self.peek())
            if not predicate(acc):
                return last
            self.next()
        return acc

    def for_each(self, f: Callable[[T_co], object]) -> None:
        for item in self:
            f(item)

    def group_by(self, key: Union[Callable[[T_co], object], Type[Unset]] = Unset):
        if key is Unset:
            out: Dict[object, List[T_co]] = Dict()
            for item in self:
                out.setdefault(item, List()).append(item)
        else:
            out = Dict()
            for item in self:
                out.setdefault(key(item), List()).append(item)
        return out

    def has_next(self, n: int = 1) -> bool:
        try:
            for _ in range(n - len(self._peek)):
                self._peek.append(next(self._it))
        except StopIteration:
            return False
        return len(self._peek) >= n

    def intersperse(self, item: object):
        def gen():
            if self.has_next():
                yield self.next()
                for e in self:
                    yield item
                    yield e

        return Iter(gen())

    def iter(self) -> Iter[T_co]:
        return self

    @overload
    def last(
        self,
        predicate: Union[Callable[[T_co], object], Type[Unset]] = Unset,
        default: Type[Unset] = Unset,
    ) -> T_co: ...

    @overload
    def last(
        self,
        predicate: Union[Callable[[T_co], object], Type[Unset]],
        default: U,
    ) -> Union[T_co, U]: ...

    def last(
        self,
        predicate: Union[Callable[[T_co], object], Type[Unset]] = Unset,
        default: object = Unset,
    ):
        if predicate is not Unset:
            self = self.filter(predicate=predicate)
        if self.has_next():
            for item in self:
                pass
            return item
        else:
            if default is Unset:
                raise StopIteration
            return default

    def list(self) -> List[T_co]:
        return List(self)

    def map(self, f: Callable[[T_co], U]) -> Iter[U]:
        return Iter(map(f, self))

    def map_to_keys(self, f: Callable[[T_co], U]) -> Dict[U, T_co]:
        return Dict((f(item), item) for item in self)

    def map_to_pairs(self, f: Callable[[T_co], U]) -> Iter[tuple[T_co, U]]:
        return self.map(lambda item: (item, f(item)))

    def map_to_values(self, f: Callable[[T_co], U]) -> Dict[T_co, U]:
        return Dict((item, f(item)) for item in self)

    def max(
        self,
        key: Optional[Callable[[T_co], object]] = None,
        default: object = Unset,
    ):
        if not self.has_next() and default is not Unset:
            return default
        return max(self, key=key)  # pyrefly: ignore[no-matching-overload]

    def min(
        self,
        key: Optional[Callable[[T_co], object]] = None,
        default: object = Unset,
    ):
        if not self.has_next() and default is not Unset:
            return default
        return min(self, key=key)  # pyrefly: ignore[no-matching-overload]

    def min_max(
        self,
        key: Optional[Callable[[T_co], object]] = None,
        default: object = Unset,
    ):
        if not self.has_next():
            require(
                default is not Unset,
                "min_max() arg is an empty sequence",
                _exc=ValueError,
            )
            return default

        if key is None:
            min = max = self.next()
            for item in self:
                if item < min:  # pyrefly: ignore[unsupported-operation]
                    min = item
                elif item > max:  # pyrefly: ignore[unsupported-operation]
                    max = item
            return min, max
        else:
            min = max = self.next()
            mink = maxk = key(min)
            for item in self:
                itemk = key(item)
                if itemk < mink:  # pyrefly: ignore[unsupported-operation]
                    min = item
                    mink = itemk
                elif itemk > maxk:  # pyrefly: ignore[unsupported-operation]
                    max = item
                    maxk = itemk
            return min, max

    def next(self) -> T_co:
        return next(self)

    def only(
        self,
        predicate: Union[Callable[[T_co], object], Type[Unset]] = Unset,
        empty_default: object = Unset,
        overfull_default: object = Unset,
    ):
        if predicate is not Unset:
            self = self.filter(predicate=predicate)

        if self.has_next():
            item = self.next()
            if self.has_next():
                if overfull_default is Unset:
                    raise ValueError("too many items found")
                return overfull_default
            return item
        if empty_default is Unset:
            raise ValueError("no item found")
        return empty_default

    def partition(self, predicate: Callable[[T_co], object] = identity):
        class Trues:
            def __init__(self, it: Iterable[T_co]) -> None:
                self.it = it  # pyrefly: ignore[invalid-type-var]

            def __next__(self):
                if trues:
                    return trues.popleft()
                for item in self.it:
                    if predicate(item):
                        return item
                    else:
                        falses.append(item)
                raise StopIteration

            def __iter__(self):
                return self

        class Falses:
            def __init__(self, it: Iterable[T_co]) -> None:
                self.it = it  # pyrefly: ignore[invalid-type-var]

            def __next__(self):
                if falses:
                    return falses.popleft()
                for item in self.it:
                    if predicate(item):
                        trues.append(item)
                    else:
                        return item
                raise StopIteration

            def __iter__(self):
                return self

        trues = deque()
        falses = deque()
        return Iter(Trues(self)), Iter(Falses(self))

    @overload
    def peek(self, n: int = 1, default: Type[Unset] = Unset) -> T_co: ...

    @overload
    def peek(self, n: int, default: U) -> Union[T_co, U]: ...

    def peek(self, n: int = 1, default: object = Unset):
        if self.has_next(n):
            return self._peek[n - 1]
        require(default is not Unset, "peek past end of iterator", _exc=ValueError)
        return default

    def permutations(self, r: Optional[int] = None):
        return Iter(itertools.permutations(self, r))

    def powerset(self):
        items = self.tuple()
        return Range(len(items) + 1).flat_map(items.combinations)

    def product(self, *its: Iterable[object], repeat: int = 1):
        return Iter(itertools.product(self, *its, repeat=repeat))

    def reduce(self, f: Callable[[T_co, T_co], T_co]):
        require(self.has_next(), "reduce on empty iterator", _exc=ValueError)
        return self.fold(self.next(), f)

    def repeat(self, n: Union[int, Type[Unset]] = Unset):
        items = self.tuple()
        # without this check, the returned iterator would block forever trying to return the first item
        if n is Unset and not items:
            return Iter()
        return Iter(itertools.chain.from_iterable(repeated(items, n)))

    def set(self) -> Set[T_co]:
        return Set(self)

    def size(self) -> int:
        count = 0
        for _ in self:
            count += 1
        return count

    def sliding(self, size: int, step: int = 1) -> Iter[Tuple[T_co]]:
        require(size > 0, size=size, _exc=ValueError)
        require(step > 0, step=step, _exc=ValueError)

        def gen():
            window = deque(self.take(size))
            while len(window) == size:
                yield Tuple(window)
                window.extend(self.take(step))
                for _ in range(step):
                    if window:
                        window.popleft()
                    else:
                        return

        return Iter(gen())

    def sliding_by_timestamp(
        self,
        size: float,
        step: float = 1,
        stamp: Callable[[T_co], tuple[float, T_co]] = timestamp(time.time),
    ) -> Iter[Tuple[T_co]]:
        require(size > 0, size=size)
        require(step != 0, step=step)

        def sliding(
            it: Iterable[U],
            stamp: Callable[[U], tuple[float, U]],
        ) -> Iter[Tuple[U]]:
            stamped: Iter[tuple[float, U]] = Iter(it).map(stamp)
            if not stamped.has_next():
                return Iter()

            actual_step = step
            if actual_step < 0:
                actual_step = -actual_step
                stamped = stamped.map(lambda item: (-item[0], item[1]))

            def gen():
                start = stamped.peek()[0]
                window: deque[tuple[float, U]] = deque()
                while True:
                    window.extend(
                        stamped.take_while(lambda item: item[0] < start + size)
                    )
                    while window and window[0][0] < start:
                        window.popleft()
                    yield Tuple(item[1] for item in window)
                    start += actual_step
                    if not stamped.has_next():
                        break

            return Iter(gen())

        return sliding(self, stamp)

    def take(
        self,
        n: Union[int, float],
        weight: Union[Callable[[T_co], Union[int, float]], Type[Unset]] = Unset,
    ) -> Iter[T_co]:
        if weight is Unset:
            item_count = cast(int, n)
            if n >= 0:

                def gen():
                    for _ in range(item_count):
                        try:
                            yield self.next()
                        except StopIteration:
                            pass
            else:

                def gen():
                    try:
                        windows = self.sliding(-item_count)
                        _next = next(windows)[0]
                        for window in windows:
                            yield _next
                            _next = window[0]
                    except StopIteration:
                        pass
        else:
            require(n > 0, n=n)
            weigh = cast(Callable[[T_co], Union[int, float]], weight)

            def gen():
                total = 0
                while self.has_next():
                    total += weigh(self.peek())
                    if total > n:
                        return
                    yield self.next()

        return Iter(gen())

    def take_while(
        self, predicate: Union[Callable[[T_co], object], Type[Unset]] = Unset
    ):
        # not implemented using itertools.takewhile because it discards the first non-passing element
        if predicate is Unset:

            def gen():
                for item in self:
                    if item:
                        yield item
                    else:
                        self._peek.append(item)
                        break
        else:

            def gen():
                for item in self:
                    if predicate(item):
                        yield item
                    else:
                        self._peek.append(item)
                        break

        return Iter(gen())

    def tee(self, n: int = 2):
        return tuple(map(Iter, itertools.tee(self, n)))

    def timestamp(self, clock: Callable[[], U] = time.time):
        return self.map(timestamp(clock))

    if _tqdm is not None:

        def tqdm(self, *a: object, **kw: object):
            return Iter(_tqdm(self, *a, **kw))  # pyrefly: ignore[no-matching-overload, not-callable]

    def transpose(self):
        return Iter(zip(*self, strict=True)).map(Tuple)

    def tuple(self) -> Tuple[T_co]:
        return Tuple(self)

    unzip = transpose

    def zip(self, *others: Iterable[object], strict: bool = False):
        return Iter(zip(self, *others, strict=strict))

    def zip_longest(self, *others: Iterable[object], fillvalue: object = None):
        return Iter(itertools.zip_longest(self, *others, fillvalue=fillvalue))


class List(TypingList[T]):
    @override
    def __add__(self, other: TypingList[U]) -> List[Union[T, U]]:
        return List((*self, *other))

    @overload
    def __getitem__(self, item: int) -> T:  # pyrefly: ignore[bad-override, missing-override-decorator]
        ...

    @overload
    def __getitem__(self, item: slice) -> List[T]: ...

    def __getitem__(self, item: Union[int, slice]):
        if isinstance(item, int):
            return list.__getitem__(self, item)
        return List(list.__getitem__(self, item))

    @override
    def __iter__(self) -> Iter[T]:
        return Iter(list.__iter__(self))

    @override
    def __mul__(self, other: SupportsIndex) -> List[T]:
        # TODO: avoid copying?
        return List(list.__mul__(self, other))

    @override
    def __reversed__(self) -> Iter[T]:
        return Iter(list.__reversed__(self))

    @override
    def __rmul__(self, other: SupportsIndex) -> List[T]:
        return List(list.__rmul__(self, other))

    def all(self, key: Union[Callable[[T], object], Type[Unset]] = Unset):
        return self.iter().all(key=key)

    def any(self, key: Union[Callable[[T], object], Type[Unset]] = Unset):
        return self.iter().any(key=key)

    def append(self, item: T):  # pyrefly: ignore[bad-override, missing-override-decorator]
        list.append(self, item)
        return self

    def apply(self, f: Callable[[List[T]], U]):
        return f(self)

    def apply_and_wrap(self, f: Callable[[List[T]], object]):
        return wrap(f(self))

    def batch(
        self,
        size: Union[int, float],
        weight: Union[Callable[[T], Union[int, float]], Type[Unset]] = Unset,
        strict: bool = False,
    ):
        return self.iter().batch(size, weight=weight, strict=strict).list()

    def chain(self, *its: Iterable[object]):
        return self.iter().chain(*its)

    def clear(self):  # pyrefly: ignore[bad-override, missing-override-decorator]
        list.clear(self)
        return self

    def combinations(self, r: int, with_replacement: bool = False):
        return self.iter().combinations(r, with_replacement=with_replacement)

    def combine_if(self, condition: object, combinator: str, *a: object, **kw: object):
        if condition:
            return getattr(self, combinator)(*a, **kw)
        return self

    @override
    def copy(self) -> List[T]:
        return List(list.copy(self))

    @overload
    def counts(self) -> Dict[T, int]: ...

    @overload
    def counts(self, key: Callable[[T], U]) -> Dict[U, int]: ...

    def counts(self, key: Union[Callable[[T], object], Type[Unset]] = Unset):
        if key is Unset:
            return self.iter().counts()
        return self.iter().counts(key=key)

    def cycle(self):
        return self.iter().cycle()

    def default_dict(
        self: List[tuple[K, V]], default_factory: Callable[[], V]
    ) -> DefaultDict[K, V]:
        return DefaultDict(default_factory).update(self)

    def dict(self: List[tuple[K, V]]) -> Dict[K, V]:
        return Dict(self)

    def discard(self, item: T):
        try:
            self.remove(item)
            return True
        except ValueError:
            return False

    def distinct(self, key: Union[Callable[[T], object], Type[Unset]] = Unset):
        return self.iter().distinct(key=key).list()

    def do(self, f: Callable[[T], object]):
        self.for_each(f)
        return self

    def drop(
        self,
        n: Union[int, float],
        weight: Union[Callable[[T], Union[int, float]], Type[Unset]] = Unset,
    ):
        if weight is Unset:
            return self[cast(int, n) :]
        return self.iter().drop(n, weight=weight).list()

    def drop_while(self, predicate: Callable[[T], object]):
        return self.iter().drop_while(predicate).list()

    def enumerate(self, start: int = 0):
        return self.iter().enumerate(start=start).list()

    def extend(self, iterable: Iterable[T]):  # pyrefly: ignore[bad-override, missing-override-decorator]
        list.extend(self, iterable)
        return self

    def filter(self, predicate: Callable[[T], object] = bool):
        return self.iter().filter(predicate=predicate).list()

    def first(
        self,
        predicate: Union[Callable[[T], object], Type[Unset]] = Unset,
        default: object = Unset,
    ):
        return self.iter().first(predicate=predicate, default=default)

    def flat_map(self, f: Callable[[T], Iterable[U]]):
        return self.iter().flat_map(f).list()

    def flatten(self: List[Iterable[U]]) -> List[U]:
        return self.iter().flatten().list()

    def fold(self, initial: U, f: Callable[[U, T], U]):
        return self.iter().fold(initial, f)

    def fold_while(
        self, initial: U, f: Callable[[U, T], U], predicate: Callable[[U], object]
    ):
        return self.iter().fold_while(initial, f, predicate)

    def for_each(self, f: Callable[[T], object]):
        return self.iter().for_each(f)

    def group_by(self, key: Union[Callable[[T], object], Type[Unset]] = Unset):
        return self.iter().group_by(key=key)

    def insert(self, index: int, item: T):  # pyrefly: ignore[bad-override, missing-override-decorator]
        list.insert(self, index, item)
        return self

    def intersperse(self, item: object):
        return self.iter().intersperse(item).list()

    def iter(self) -> Iter[T]:
        return iter(self)

    def last(
        self,
        predicate: Union[Callable[[T], object], Type[Unset]] = Unset,
        default: object = Unset,
    ):
        return self.iter().last(predicate=predicate, default=default)

    def list(self) -> List[T]:
        return self

    def map(self, f: Callable[[T], U]):
        return self.iter().map(f).list()

    def map_to_keys(self, f: Callable[[T], U]):
        return self.iter().map_to_keys(f)

    def map_to_pairs(self, f: Callable[[T], U]):
        return self.iter().map_to_pairs(f).list()

    def map_to_values(self, f: Callable[[T], U]):
        return self.iter().map_to_values(f)

    def max(self, key: Optional[Callable[[T], object]] = None, default: object = Unset):
        return self.iter().max(key=key, default=default)

    def min(self, key: Optional[Callable[[T], object]] = None, default: object = Unset):
        return self.iter().min(key=key, default=default)

    def min_max(
        self, key: Optional[Callable[[T], object]] = None, default: object = Unset
    ):
        return self.iter().min_max(key=key, default=default)

    def only(
        self,
        predicate: Union[Callable[[T], object], Type[Unset]] = Unset,
        empty_default: object = Unset,
        overfull_default: object = Unset,
    ):
        return self.iter().only(
            predicate=predicate,
            empty_default=empty_default,
            overfull_default=overfull_default,
        )

    def partition(self, predicate: Callable[[T], object] = identity):
        return wrap(self.iter().partition(predicate=predicate)).map(List)

    def permutations(self, r: Optional[int] = None):
        return self.iter().permutations(r)

    def powerset(self):
        return self.iter().powerset()

    def product(self, *its: Iterable[object], repeat: int = 1):
        return self.iter().product(*its, repeat=repeat)

    def reduce(self, f: Callable[[T, T], T]):
        return self.iter().reduce(f)

    def repeat(self, n: Union[int, Type[Unset]] = Unset):
        return self.iter().repeat(n=n)

    def reverse(self):  # pyrefly: ignore[bad-override, missing-override-decorator]
        list.reverse(self)
        return self

    def reversed(self):
        return reversed(self)

    def set(self) -> Set[T]:
        return Set(self)

    def size(self) -> int:
        return len(self)

    def sliding(self, size: int, step: int = 1):
        return self.iter().sliding(size, step=step).list()

    def sliding_by_timestamp(
        self,
        size: float,
        step: float = 1,
        stamp: Callable[[T], tuple[float, T]] = timestamp(time.time),
    ):
        return self.iter().sliding_by_timestamp(size, step=step, stamp=stamp).list()

    def sort(self, key: Optional[Callable[[T], object]] = None, reverse: bool = False):  # pyrefly: ignore[bad-override, missing-override-decorator]
        list.sort(self, key=key, reverse=reverse)  # pyrefly: ignore[bad-argument-type]
        return self

    def sorted(
        self, key: Optional[Callable[[T], object]] = None, reverse: bool = False
    ):
        return List(
            sorted(self, key=key, reverse=reverse)  # pyrefly: ignore[no-matching-overload]
        )

    def take(
        self,
        n: Union[int, float],
        weight: Union[Callable[[T], Union[int, float]], Type[Unset]] = Unset,
    ):
        if weight is Unset:
            return self[: cast(int, n)]
        return self.iter().take(n, weight=weight).list()

    def take_while(self, predicate: Union[Callable[[T], object], Type[Unset]] = Unset):
        return self.iter().take_while(predicate=predicate).list()

    def timestamp(self, clock: Callable[[], U] = time.time):
        return self.iter().timestamp(clock=clock).list()

    if _tqdm is not None:

        def tqdm(self, *a: object, **kw: object):
            return self.iter().tqdm(*a, **kw)

    def transpose(self):
        return self.iter().transpose().list()

    def tuple(self) -> Tuple[T]:
        return Tuple(self)

    unzip = transpose

    def zip(self, *others: Iterable[object], strict: bool = False):
        return self.iter().zip(*others, strict=strict).list()

    def zip_longest(self, *others: Iterable[object], fillvalue: object = None):
        return self.iter().zip_longest(*others, fillvalue=fillvalue).list()


class Range:
    def __init__(self, *a: Union[int, range, Range], **kw: int):
        if len(a) == 1 and isinstance(a[0], (range, Range)):
            require(not kw)
            if isinstance(a[0], range):
                self._range = a[0]
            else:
                self._range = a[0]._range
        else:
            self._range = range(*a, **kw)  # pyrefly: ignore[no-matching-overload]

    def __contains__(self, item: object):
        return item in self._range

    def __eq__(self, other: object):
        if isinstance(other, Range):
            other = other._range
        return self._range == other

    def __getitem__(self, item: Union[int, slice]):
        if isinstance(item, int):
            return self._range[item]
        return Range(self._range[item])

    def __hash__(self):
        return hash(self._range)

    def __iter__(self) -> Iter[int]:
        return Iter(self._range)

    def __len__(self):
        return len(self._range)

    def __repr__(self):
        return repr(self._range).title()

    def __reversed__(self) -> Range:
        return Range(
            self.start + (self.step * (len(self) - 1)),
            self.start - self.step,
            -self.step,
        )

    @property
    def start(self) -> int:
        return self._range.start

    @property
    def stop(self) -> int:
        return self._range.stop

    @property
    def step(self) -> int:
        return self._range.step

    def all(self, key: Union[Callable[[int], object], Type[Unset]] = Unset):
        return self.iter().all(key=key)

    def any(self, key: Union[Callable[[int], object], Type[Unset]] = Unset):
        return self.iter().any(key=key)

    def apply(self, f: Callable[[Range], U]):
        return f(self)

    def apply_and_wrap(self, f: Callable[[Range], object]):
        return wrap(f(self))

    def batch(
        self,
        size: Union[int, float],
        weight: Union[Callable[[int], Union[int, float]], Type[Unset]] = Unset,
        strict: bool = False,
    ):
        if weight is Unset:
            item_count = cast(int, size)

            def gen():
                batch_step = self.step * item_count
                start = self.start
                stop = min(self.stop, self.start + batch_step)
                if stop == self.stop:
                    yield Range(start, stop, self.step)
                else:
                    while stop < self.stop:
                        yield Range(start, stop, self.step)
                        start += batch_step
                        stop = min(self.stop, stop + batch_step)
                    yield Range(start, stop, self.step)

            return Iter(gen())

        return self.iter().batch(size, weight=weight, strict=strict)

    def chain(self, *its: Iterable[object]):
        return self.iter().chain(*its)

    def combinations(self, r: int, with_replacement: bool = False):
        return self.iter().combinations(r, with_replacement=with_replacement)

    def combine_if(self, condition: object, combinator: str, *a: object, **kw: object):
        if condition:
            return getattr(self, combinator)(*a, **kw)
        return self

    @overload
    def counts(self) -> Dict[int, int]: ...

    @overload
    def counts(self, key: Callable[[int], U]) -> Dict[U, int]: ...

    def counts(self, key: Union[Callable[[int], object], Type[Unset]] = Unset):
        if key is Unset:
            return self.iter().counts()
        return self.iter().counts(key=key)

    def cycle(self):
        return self.iter().cycle()

    def do(self, f: Callable[[int], object]):
        self.for_each(f)
        return self

    def drop(
        self,
        n: Union[int, float],
        weight: Union[Callable[[int], Union[int, float]], Type[Unset]] = Unset,
    ):
        if weight is Unset:
            return Range(self._range[cast(int, n) :])
        return self.iter().drop(n, weight=weight)

    def drop_while(self, predicate: Callable[[int], object]):
        start = self.iter().first(lambda n: not predicate(n), self.stop)
        return Range(start, self.stop, self.step)

    def enumerate(self, start: int = 0):
        return self.iter().enumerate(start=start)

    def filter(self, predicate: Callable[[int], object] = bool):
        return self.iter().filter(predicate=predicate)

    def first(
        self,
        predicate: Union[Callable[[int], object], Type[Unset]] = Unset,
        default: object = Unset,
    ):
        return self.iter().first(predicate=predicate, default=default)

    def flat_map(self, f: Callable[[int], Iterable[U]]):
        return self.iter().flat_map(f)

    def fold(self, initial: U, f: Callable[[U, int], U]):
        return self.iter().fold(initial, f)

    def fold_while(
        self, initial: U, f: Callable[[U, int], U], predicate: Callable[[U], object]
    ):
        return self.iter().fold_while(initial, f, predicate)

    def for_each(self, f: Callable[[int], object]):
        return self.iter().for_each(f)

    def group_by(self, key: Union[Callable[[int], object], Type[Unset]] = Unset):
        return self.iter().group_by(key=key)

    def index(self, value: int):
        return self._range.index(value)

    def intersperse(self, item: object):
        return self.iter().intersperse(item)

    def iter(self):
        return iter(self)

    def last(
        self,
        predicate: Union[Callable[[int], object], Type[Unset]] = Unset,
        default: object = Unset,
    ):
        return self.reversed().first(predicate=predicate, default=default)

    def list(self) -> List[int]:
        return List(self)

    def map(self, f: Callable[[int], U]):
        return self.iter().map(f)

    def map_to_keys(self, f: Callable[[int], U]):
        return self.iter().map_to_keys(f)

    def map_to_pairs(self, f: Callable[[int], U]):
        return self.iter().map_to_pairs(f)

    def map_to_values(self, f: Callable[[int], U]):
        return self.iter().map_to_values(f)

    def max(
        self, key: Optional[Callable[[int], object]] = None, default: object = Unset
    ):
        return self.iter().max(key=key, default=default)

    def min(
        self, key: Optional[Callable[[int], object]] = None, default: object = Unset
    ):
        return self.iter().min(key=key, default=default)

    def min_max(
        self, key: Optional[Callable[[int], object]] = None, default: object = Unset
    ):
        return self.iter().min_max(key=key, default=default)

    def only(
        self,
        predicate: Union[Callable[[int], object], Type[Unset]] = Unset,
        empty_default: object = Unset,
        overfull_default: object = Unset,
    ):
        return self.iter().only(
            predicate=predicate,
            empty_default=empty_default,
            overfull_default=overfull_default,
        )

    def partition(self, predicate: Union[Callable[[int], object], Type[Unset]] = Unset):
        return self.iter().partition(predicate=predicate)

    def permutations(self, r: Optional[int] = None):
        return self.iter().permutations(r=r)

    def powerset(self):
        return self.iter().powerset()

    def product(self, *its: Iterable[object], repeat: int = 1):
        return self.iter().product(*its, repeat=repeat)

    def reduce(self, f: Callable[[int, int], int]):
        return self.iter().reduce(f)

    def repeat(self, n: Union[int, Type[Unset]] = Unset):
        return self.iter().repeat(n=n)

    def reversed(self) -> Range:
        return self.__reversed__()

    def set(self) -> Set[int]:
        return Set(self)

    def size(self) -> int:
        return len(self)

    def sliding(self, size: int, step: int = 1):
        def gen():
            window = cast(Range, self.take(size))

            while window.size() == size:
                yield window
                window = Range(
                    window.start + self.step * step,
                    min(self.stop, window.stop + self.step * step),
                    self.step,
                )

        return Iter(gen())

    def sliding_by_timestamp(
        self,
        size: float,
        step: float = 1,
        stamp: Callable[[int], tuple[float, int]] = timestamp(time.time),
    ):
        return self.iter().sliding_by_timestamp(size, step=step, stamp=stamp)

    def take(
        self,
        n: Union[int, float],
        weight: Union[Callable[[int], Union[int, float]], Type[Unset]] = Unset,
    ):
        if weight is Unset:
            return Range(self._range[: cast(int, n)])
        return self.iter().take(n, weight=weight)

    def take_while(
        self, predicate: Union[Callable[[int], object], Type[Unset]] = Unset
    ):
        if predicate is Unset:
            predicate = bool
        stop = self.iter().first(lambda n: not predicate(n), self.stop)
        return Range(self.start, stop, self.step)

    def timestamp(self, clock: Callable[[], U] = time.time):
        return self.iter().timestamp(clock=clock)

    if _tqdm is not None:

        def tqdm(self, *a: object, **kw: object):
            return self.iter().tqdm(*a, **kw)

    def tuple(self) -> Tuple[int]:
        return Tuple(self)

    def zip(self, *others: Iterable[object], strict: bool = False):
        return self.iter().zip(*others, strict=strict)

    def zip_longest(self, *others: Iterable[object], fillvalue: object = None):
        return self.iter().zip_longest(*others, fillvalue=fillvalue)


class Set(TypingSet[T]):
    def __add__(self, other: Iterable[T]):
        return Set({*self, *other})

    @override
    def __and__(self, other: AbstractSet[object]) -> Set[T]:
        return Set(set.__and__(self, other))

    def __iadd__(self, other: Iterable[T]):
        self.update(other)
        return self

    @override
    def __iter__(self) -> Iter[T]:
        return Iter(set.__iter__(self))

    @override
    def __isub__(self, other: AbstractSet[object]) -> Set[T]:
        return Set(set.__isub__(self, other))

    @override
    def __or__(self, other: AbstractSet[U]) -> Set[Union[T, U]]:
        return Set(set.__or__(self, other))

    @override
    def __sub__(self, other: AbstractSet[object]) -> Set[T]:
        return Set(set.__sub__(self, other))

    @override
    def __xor__(self, other: AbstractSet[U]) -> Set[Union[T, U]]:
        return Set(set.__xor__(self, other))

    def add(self, item: T):  # pyrefly: ignore[bad-override, missing-override-decorator]
        if item in self:
            return False
        set.add(self, item)
        return True

    def all(self, key: Union[Callable[[T], object], Type[Unset]] = Unset):
        return self.iter().all(key=key)

    def any(self, key: Union[Callable[[T], object], Type[Unset]] = Unset):
        return self.iter().any(key=key)

    def apply(self, f: Callable[[Set[T]], U]):
        return f(self)

    def apply_and_wrap(self, f: Callable[[Set[T]], object]):
        return wrap(f(self))

    def batch(
        self,
        size: Union[int, float],
        weight: Union[Callable[[T], Union[int, float]], Type[Unset]] = Unset,
        strict: bool = False,
    ):
        return self.iter().batch(size, weight=weight, strict=strict).map(Set)

    def chain(self, *its: Iterable[object]):
        return self.iter().chain(*its)

    def clear(self):  # pyrefly: ignore[bad-override, missing-override-decorator]
        set.clear(self)
        return self

    def combinations(self, r: int):
        return self.iter().combinations(r).map(Set)

    def combine_if(self, condition: object, combinator: str, *a: object, **kw: object):
        if condition:
            return getattr(self, combinator)(*a, **kw)
        return self

    @override
    def copy(self) -> Set[T]:
        return Set(self)

    def counts(self, key: Callable[[T], U]) -> Dict[U, int]:
        """key is required since items are guaranteed unique"""
        return self.iter().counts(key=key)

    def cycle(self):
        return self.iter().cycle()

    def default_dict(
        self: Set[tuple[K, V]], default_factory: Callable[[], V]
    ) -> DefaultDict[K, V]:
        return DefaultDict(default_factory).update(self)

    def dict(self: Set[tuple[K, V]]) -> Dict[K, V]:
        return Dict(self)

    def discard(self, item: T):  # pyrefly: ignore[bad-override, missing-override-decorator]
        if item in self:
            self.remove(item)
            return True
        return False

    def distinct(self, key: Callable[[T], object]):
        """key is required because items are guaranteed unique"""
        return self.iter().distinct(key=key).set()

    def do(self, f: Callable[[T], object]):
        self.for_each(f)
        return self

    def drop(
        self,
        n: Union[int, float],
        weight: Union[Callable[[T], Union[int, float]], Type[Unset]] = Unset,
    ):
        return self.iter().drop(n, weight=weight).set()

    def drop_while(self, predicate: Callable[[T], object]):
        return self.iter().drop_while(predicate).set()

    def enumerate(self, start: int = 0):
        return self.iter().enumerate(start=start).set()

    def filter(self, predicate: Callable[[T], object] = bool):
        return self.iter().filter(predicate=predicate).set()

    def first(
        self,
        predicate: Union[Callable[[T], object], Type[Unset]] = Unset,
        default: object = Unset,
    ):
        return self.iter().first(predicate=predicate, default=default)

    def flat_map(self, f: Callable[[T], Iterable[U]]):
        return self.iter().flat_map(f).set()

    def flatten(self: Set[Iterable[U]]) -> Set[U]:
        return self.iter().flatten().set()

    def fold(self, initial: U, f: Callable[[U, T], U]):
        return self.iter().fold(initial, f)

    def fold_while(
        self, initial: U, f: Callable[[U, T], U], predicate: Callable[[U], object]
    ):
        return self.iter().fold_while(initial, f, predicate)

    def for_each(self, f: Callable[[T], object]):
        return self.iter().for_each(f)

    def group_by(self, key: Callable[[T], object]):
        """key is required because items are guaranteed unique"""
        return self.iter().group_by(key=key)

    def iter(self) -> Iter[T]:
        return iter(self)

    def last(
        self,
        predicate: Union[Callable[[T], object], Type[Unset]] = Unset,
        default: object = Unset,
    ):
        return self.iter().last(predicate=predicate, default=default)

    def list(self) -> List[T]:
        return List(self)

    def map(self, f: Callable[[T], U]):
        return self.iter().map(f).set()

    def map_to_keys(self, f: Callable[[T], U]):
        return self.iter().map_to_keys(f)

    def map_to_pairs(self, f: Callable[[T], U]):
        return self.iter().map_to_pairs(f).set()

    def map_to_values(self, f: Callable[[T], U]):
        return self.iter().map_to_values(f)

    def max(self, key: Optional[Callable[[T], object]] = None, default: object = Unset):
        return self.iter().max(key=key, default=default)

    def min(self, key: Optional[Callable[[T], object]] = None, default: object = Unset):
        return self.iter().min(key=key, default=default)

    def min_max(
        self, key: Optional[Callable[[T], object]] = None, default: object = Unset
    ):
        return self.iter().min_max(key=key, default=default)

    def only(
        self,
        predicate: Union[Callable[[T], object], Type[Unset]] = Unset,
        empty_default: object = Unset,
        overfull_default: object = Unset,
    ):
        return self.iter().only(
            predicate=predicate,
            empty_default=empty_default,
            overfull_default=overfull_default,
        )

    def partition(self, predicate: Union[Callable[[T], object], Type[Unset]] = Unset):
        return wrap(self.iter().partition(predicate=predicate)).map(Set)

    def permutations(self, r: Optional[int] = None):
        return self.iter().permutations(r=r)

    def powerset(self):
        return self.iter().powerset()

    def product(self, *its: Iterable[object], repeat: int = 1):
        return self.iter().product(*its, repeat=repeat)

    def reduce(self, f: Callable[[T, T], T]):
        return self.iter().reduce(f)

    def repeat(self, n: Union[int, Type[Unset]] = Unset):
        return self.iter().repeat(n=n)

    def set(self) -> Set[T]:
        return self

    def size(self) -> int:
        return len(self)

    def sliding(self, size: int, step: int = 1):
        return self.iter().sliding(size, step=step).map(Set)

    def sliding_by_timestamp(
        self,
        size: float,
        step: float = 1,
        stamp: Callable[[T], tuple[float, T]] = timestamp(time.time),
    ):
        return self.iter().sliding_by_timestamp(size, step=step, stamp=stamp).map(Set)

    def take(
        self,
        n: Union[int, float],
        weight: Union[Callable[[T], Union[int, float]], Type[Unset]] = Unset,
    ):
        return self.iter().take(n, weight=weight).set()

    def take_while(self, predicate: Union[Callable[[T], object], Type[Unset]] = Unset):
        return self.iter().take_while(predicate=predicate).set()

    def timestamp(self, clock: Callable[[], U] = time.time):
        return self.iter().timestamp(clock=clock).set()

    if _tqdm is not None:

        def tqdm(self, *a: object, **kw: object):
            return self.iter().tqdm(*a, **kw)

    def transpose(self):
        return self.iter().transpose()

    def tuple(self) -> Tuple[T]:
        return Tuple(self)

    unzip = transpose

    def update(self, *s: Iterable[T]):  # pyrefly: ignore[bad-override, missing-override-decorator]
        set.update(self, *s)
        return self

    def zip(self, *others: Iterable[object], strict: bool = False):
        return self.iter().zip(*others, strict=strict).set()

    def zip_longest(self, *others: Iterable[object], fillvalue: object = None):
        return self.iter().zip_longest(*others, fillvalue=fillvalue).set()


class Tuple(TypingTuple[T_co, ...]):
    @override
    def __add__(self, other: TypingTuple[U, ...]) -> Tuple[Union[T_co, U]]:
        return Tuple((*self, *other))

    @overload
    def __getitem__(self, item: int) -> T_co:  # pyrefly: ignore[bad-override, missing-override-decorator]
        ...

    @overload
    def __getitem__(self, item: slice) -> Tuple[T_co]: ...

    def __getitem__(self, item: Union[int, slice]):
        if isinstance(item, int):
            return tuple.__getitem__(self, item)
        return Tuple(tuple.__getitem__(self, item))

    @override
    def __iter__(self) -> Iter[T_co]:
        return Iter(tuple.__iter__(self))

    @override
    def __mul__(self, other: SupportsIndex) -> Tuple[T_co]:
        return Tuple(tuple.__mul__(self, other))

    @override
    def __reversed__(self) -> Iter[T_co]:
        return Iter(reversed(tuple(self)))

    @override
    def __rmul__(self, other: SupportsIndex) -> Tuple[T_co]:
        return Tuple(tuple.__rmul__(self, other))

    def all(self, key: Union[Callable[[T_co], object], Type[Unset]] = Unset):
        return self.iter().all(key=key)

    def any(self, key: Union[Callable[[T_co], object], Type[Unset]] = Unset):
        return self.iter().any(key=key)

    def apply(self, f: Callable[[Tuple[T_co]], U]):
        return f(self)

    def apply_and_wrap(self, f: Callable[[Tuple[T_co]], object]):
        return wrap(f(self))

    def batch(
        self,
        size: Union[int, float],
        weight: Union[Callable[[T_co], Union[int, float]], Type[Unset]] = Unset,
        strict: bool = False,
    ):
        return self.iter().batch(size, weight=weight, strict=strict).tuple()

    def chain(self, *its: Iterable[object]):
        return self.iter().chain(*its)

    def combinations(self, r: int, with_replacement: bool = False):
        return self.iter().combinations(r, with_replacement=with_replacement)

    def combine_if(self, condition: object, combinator: str, *a: object, **kw: object):
        if condition:
            return getattr(self, combinator)(*a, **kw)
        return self

    @overload
    def counts(self) -> Dict[T_co, int]: ...

    @overload
    def counts(self, key: Callable[[T_co], U]) -> Dict[U, int]: ...

    def counts(self, key: Union[Callable[[T_co], object], Type[Unset]] = Unset):
        if key is Unset:
            return self.iter().counts()
        return self.iter().counts(key=key)

    def cycle(self):
        return self.iter().cycle()

    def default_dict(
        self: Tuple[tuple[K, V]], default_factory: Callable[[], V]
    ) -> DefaultDict[K, V]:
        return DefaultDict(default_factory).update(self)

    def dict(self: Tuple[tuple[K, V]]) -> Dict[K, V]:
        return Dict(self)

    def distinct(self, key: Union[Callable[[T_co], object], Type[Unset]] = Unset):
        return self.iter().distinct(key=key).tuple()

    def do(self, f: Callable[[T_co], object]):
        self.for_each(f)
        return self

    def drop(
        self,
        n: Union[int, float],
        weight: Union[Callable[[T_co], Union[int, float]], Type[Unset]] = Unset,
    ):
        if weight is Unset:
            return self[n:]
        return self.iter().drop(n, weight=weight).tuple()

    def drop_while(self, predicate: Callable[[T_co], object]):
        return self.iter().drop_while(predicate).tuple()

    def enumerate(self, start: int = 0):
        return self.iter().enumerate(start=start).tuple()

    def filter(self, predicate: Callable[[T_co], object] = bool):
        return self.iter().filter(predicate=predicate).tuple()

    def first(
        self,
        predicate: Union[Callable[[T_co], object], Type[Unset]] = Unset,
        default: object = Unset,
    ):
        return self.iter().first(predicate=predicate, default=default)

    def flat_map(self, f: Callable[[T_co], Iterable[U]]):
        return self.iter().flat_map(f).tuple()

    def flatten(self: Tuple[Iterable[U]]) -> Tuple[U]:
        return self.iter().flatten().tuple()

    def fold(self, initial: U, f: Callable[[U, T_co], U]):
        return self.iter().fold(initial, f)

    def fold_while(
        self, initial: U, f: Callable[[U, T_co], U], predicate: Callable[[U], object]
    ):
        return self.iter().fold_while(initial, f, predicate)

    def for_each(self, f: Callable[[T_co], object]):
        return self.iter().for_each(f)

    def group_by(self, key: Union[Callable[[T_co], object], Type[Unset]] = Unset):
        return self.iter().group_by(key=key)

    def intersperse(self, item: object):
        return self.iter().intersperse(item).tuple()

    def iter(self) -> Iter[T_co]:
        return Iter(self)

    def last(
        self,
        predicate: Union[Callable[[T_co], object], Type[Unset]] = Unset,
        default: object = Unset,
    ):
        return self.iter().last(predicate=predicate, default=default)

    def list(self) -> List[T_co]:
        return List(self)

    def map(self, f: Callable[[T_co], U]):
        return self.iter().map(f).tuple()

    def map_to_keys(self, f: Callable[[T_co], U]):
        return self.iter().map_to_keys(f)

    def map_to_pairs(self, f: Callable[[T_co], U]):
        return self.iter().map_to_pairs(f).tuple()

    def map_to_values(self, f: Callable[[T_co], U]):
        return self.iter().map_to_values(f)

    def max(
        self, key: Optional[Callable[[T_co], object]] = None, default: object = Unset
    ):
        return self.iter().max(key=key, default=default)

    def min(
        self, key: Optional[Callable[[T_co], object]] = None, default: object = Unset
    ):
        return self.iter().min(key=key, default=default)

    def min_max(
        self, key: Optional[Callable[[T_co], object]] = None, default: object = Unset
    ):
        return self.iter().min_max(key=key, default=default)

    def only(
        self,
        predicate: Union[Callable[[T_co], object], Type[Unset]] = Unset,
        empty_default: object = Unset,
        overfull_default: object = Unset,
    ):
        return self.iter().only(
            predicate=predicate,
            empty_default=empty_default,
            overfull_default=overfull_default,
        )

    def partition(self, predicate: Callable[[T_co], object] = identity):
        return wrap(self.iter().partition(predicate=predicate)).map(Tuple)

    def permutations(self, r: Optional[int] = None):
        return self.iter().permutations(r)

    def powerset(self):
        return self.iter().powerset()

    def product(self, *its: Iterable[object], repeat: int = 1):
        return self.iter().product(*its, repeat=repeat)

    def reduce(self, f: Callable[[T_co, T_co], T_co]):
        return self.iter().reduce(f)

    def repeat(self, n: Union[int, Type[Unset]] = Unset):
        return self.iter().repeat(n=n)

    def reversed(self):
        return reversed(self)

    def set(self) -> Set[T_co]:
        return Set(self)

    def size(self) -> int:
        return len(self)

    def sliding(self, size: int, step: int = 1):
        return self.iter().sliding(size, step=step).tuple()

    def sliding_by_timestamp(
        self,
        size: float,
        step: float = 1,
        stamp: Callable[[T_co], tuple[float, T_co]] = timestamp(time.time),
    ):
        return self.iter().sliding_by_timestamp(size, step=step, stamp=stamp).tuple()

    def sorted(
        self, key: Optional[Callable[[T_co], object]] = None, reverse: bool = False
    ):
        return Tuple(
            sorted(self, key=key, reverse=reverse)  # pyrefly: ignore[no-matching-overload]
        )

    def take(
        self,
        n: Union[int, float],
        weight: Union[Callable[[T_co], Union[int, float]], Type[Unset]] = Unset,
    ):
        if weight is Unset:
            return self[: cast(int, n)]
        return self.iter().take(n, weight=weight).tuple()

    def take_while(
        self, predicate: Union[Callable[[T_co], object], Type[Unset]] = Unset
    ):
        return self.iter().take_while(predicate=predicate).tuple()

    def timestamp(self, clock: Callable[[], U] = time.time):
        return self.iter().timestamp(clock=clock).tuple()

    if _tqdm is not None:

        def tqdm(self, *a: object, **kw: object):
            return self.iter().tqdm(*a, **kw)

    def transpose(self):
        return self.iter().transpose().tuple()

    def tuple(self) -> Tuple[T_co]:
        return self

    unzip = transpose

    def zip(self, *others: Iterable[object], strict: bool = False):
        return self.iter().zip(*others, strict=strict).tuple()

    def zip_longest(self, *others: Iterable[object], fillvalue: object = None):
        return self.iter().zip_longest(*others, fillvalue=fillvalue).tuple()
