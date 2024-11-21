import functools
import itertools
import time
from collections import defaultdict, deque
from operator import itemgetter, add

try:
    from operator import call
except ImportError:
    def call(f):
        return f()

from ajprax.hof import identity
from ajprax.require import require

try:
    from tqdm import tqdm
except ImportError:
    tqdm = False

from ajprax.hof import t
from ajprax.sentinel import Unset


def count(start=0, step=1):
    return Iter(itertools.count(start, step))


def repeated(item, n=Unset):
    if n is Unset:
        return Iter(itertools.repeat(item))
    return Iter(itertools.repeat(item, n))


def repeatedly(f, n=Unset):
    return repeated(f, n).map(call)


def timestamp(clock=time.time):
    def inner(item):
        return clock(), item

    return inner


class Deque(deque):
    def __add__(self, other):
        return Deque(deque.__add__(self, other))

    def __getitem__(self, item):
        # TODO: support slices?
        return deque.__getitem__(self, item)

    def __iter__(self):
        return Iter(deque.__iter__(self))

    def __mul__(self, other):
        return Deque(deque.__mul__(self, other))

    def __rmul__(self, other):
        return Deque(deque.__rmul__(self, other))

    def all(self, key=Unset):
        return self.iter().all(key=key)

    def any(self, key=Unset):
        return self.iter().any(key=key)

    def append(self, item):
        deque.append(self, item)
        return self

    def appendleft(self, item):
        deque.appendleft(self, item)
        return self

    def apply(self, f):
        return f(self)

    def apply_and_iter(self, f):
        return Iter(f(self))

    def batch(self, size):
        return self.iter().batch(size)

    def chain(self, *its):
        return self.iter().chain(*its)

    def clear(self):
        deque.clear(self)
        return self

    def combinations(self, r, with_replacement=False):
        return self.iter().combinations(r, with_replacement=with_replacement)

    def combine_if(self, condition, combinator, *a, **kw):
        if condition:
            return getattr(self, combinator)(*a, **kw)
        return self

    def count(self, key=Unset):
        return self.iter().count(key=key)

    def cycle(self):
        return self.iter().cycle()

    def dict(self):
        return Dict(self)

    def discard(self, item):
        try:
            self.remove(item)
            return True
        except ValueError:
            return False

    def distinct(self, key=Unset):
        return self.iter().distinct(key=key)

    def do(self, f):
        return self.iter().do(f)

    def drop(self, n):
        # TODO: if we update to support slices, use that
        return self.iter().drop(n)

    def drop_while(self, predicate):
        return self.iter().drop_while(predicate)

    def enumerate(self, start=0):
        return self.iter().enumerate(start=start)

    def extend(self, it):
        deque.extend(self, it)
        return self

    def extendleft(self, it):
        deque.extendleft(self, it)
        return self

    def filter(self, predicate=Unset):
        return self.iter().filter(predicate=predicate)

    def first(self, predicate=Unset, default=Unset):
        return self.iter().first(predicate=predicate, default=default)

    def flat_map(self, f):
        return self.iter().flat_map(f)

    def flatten(self):
        return self.iter().flatten()

    def fold(self, initial, f):
        return self.iter().fold(initial, f)

    def fold_while(self, initial, f, predicate):
        return self.iter().fold_while(initial, f, predicate)

    def for_each(self, f):
        return self.iter().for_each(f)

    def group_by(self, key=Unset):
        return self.iter().group_by(key=key)

    def insert(self, index, item):
        deque.insert(index, item)
        return self

    def intersperse(self, item):
        return self.iter().intersperse(item)

    def iter(self):
        return Iter(self)

    def last(self, predicate=Unset, default=Unset):
        return self.iter().last(predicate=predicate, default=default)

    def list(self):
        return List(self)

    def map(self, f):
        return self.iter().map(f)

    def map_to_keys(self, f):
        return self.iter().map_to_keys(f)

    def map_to_pairs(self, f):
        return self.iter().map_to_pairs(f)

    def map_to_values(self, f):
        return self.iter().map_to_values(f)

    def max(self, key=None, default=Unset):
        return self.iter().max(key=key, default=default)

    def min(self, key=None, default=Unset):
        return self.iter().min(key=key, default=default)

    def min_max(self, key=None, default=Unset):
        return self.iter().min_max(key=key, default=default)

    def only(self, predicate=Unset):
        return self.iter().only(predicate=predicate)

    def partition(self, predicate=identity):
        return self.iter().partition(predicate=predicate)

    def permutations(self, r=None):
        return self.iter().permutations(r=r)

    def powerset(self):
        return self.iter().powerset()

    def product(self, *its, repeat=1):
        return self.iter().product(*its, repeat=repeat)

    def reduce(self, f):
        return self.iter().reduce(f)

    def repeat(self, n=Unset):
        return self.iter().repeat(n=n)

    def rotate(self, n=1):
        deque.rotate(self, n)
        return self

    def reverse(self):
        deque.reverse(self)
        return self

    def set(self):
        return Set(self)

    def size(self):
        return len(self)

    def sliding(self, size, step=1):
        return self.iter().sliding(size, step=step)

    def sliding_by_timestamp(self, size, step=1, stamp=timestamp(time.time)):
        return self.iter().sliding_by_timestamp(size, step=step, stamp=stamp)

    def sort(self, key=None, reverse=False):
        items = self.list().sort(key=key, reverse=reverse)
        return self.clear().extend(items)

    def take(self, n):
        # TODO: use slice if implemented
        return self.iter().take(n)

    def take_while(self, predicate=Unset):
        return self.iter().take_while(predicate=predicate)

    def timestamp(self, clock=time.time):
        return self.iter().timestamp(clock=clock)

    if tqdm:
        def tqdm(self, *a, **kw):
            return self.iter().tqdm(*a, **kw)

    def transpose(self):
        return self.iter().transpose()

    def tuple(self):
        return Tuple(self)

    unzip = transpose

    def zip(self, *others, strict=False):
        return self.iter().zip(*others, strict=strict)

    def zip_longest(self, *others, fillvalue=None):
        return self.iter().zip_longest(*others, fillvalue=fillvalue)


class Dict(dict):
    # TODO:
    #  give keys() and values() combinators (previously they were iterators which was awkward in interactive settings)

    def __iter__(self):
        return Iter(dict.__iter__(self))

    def all(self, key=Unset):
        return self.items().all(key=key)

    def any(self, key=Unset):
        return self.items().any(key=key)

    def apply(self, f):
        return f(self)

    def apply_and_iter(self, f):
        return Iter(f(self))

    def batch(self, size):
        return self.items().batch(size)

    def chain(self, *its):
        return self.items().chain(*its)

    def clear(self):
        dict.clear(self)
        return self

    def combinations(self, r, with_replacement=False):
        return self.items().combinations(r, with_replacement=with_replacement)

    def combine_if(self, condition, combinator, *a, **kw):
        if condition:
            return getattr(self, combinator)(*a, **kw)
        return self

    # TODO: remove and just use .values().count(). This depends on .values() returning a custom class
    # replaces count() because keys are guaranteed to be unique so count over items is degenerate
    def count_values(self, key=Unset):
        return Iter(self.values()).count(key=key)

    def cycle(self):
        return self.items().cycle()

    def dict(self):
        return Dict(self)

    # replaces distinct() because keys are guaranteed to be unique so distinct over items is degenerate
    def distinct_values(self, key=Unset):
        if key is Unset:
            return self.items().distinct()
        return self.items().distinct(key=t(lambda k, v: key(v)))

    def do(self, f):
        return self.items().do(f)

    def drop(self, n):
        return self.items().drop(n)

    def drop_while(self, predicate):
        return self.items().drop_while(predicate)

    def enumerate(self, start=0):
        return self.items().enumerate(start=start)

    def filter(self, predicate):
        return self.items().filter(predicate)

    def filter_keys(self, predicate):
        return self.items().filter(t(lambda k, v: predicate(k)))

    def filter_values(self, predicate):
        return self.items().filter(t(lambda k, v: predicate(v)))

    def first(self, predicate=Unset, default=Unset):
        return self.items().first(predicate=predicate, default=default)

    def flat_map(self, f):
        return self.items().flat_map(f)

    def flat_map_keys(self, f):
        return self.items().flat_map(t(lambda k, v: ((k, v) for k in f(k))))

    def flat_map_values(self, f):
        return self.items().flat_map(t(lambda k, v: ((k, v) for v in f(v))))

    def fold(self, initial, f):
        return self.items().fold(initial, f)

    def fold_while(self, initial, f, predicate):
        return self.items().fold_while(initial, f, predicate)

    def for_each(self, f):
        self.items().for_each(f)

    def group_by(self, key=Unset):
        return self.items().group_by(key=key)

    def intersection(self, *others):

        dicts = (self, *others)
        keys = Iter(dicts).map(lambda d: d.keys()).reduce(lambda a, b: a & b)
        out = Dict()
        for key in keys:
            for d in dicts:
                value = d[key]
                # TODO: should we instead skip keys that don't match?
                require(out.setdefault(key, value) == value, "mismatched values", key=key)
        return out

    def intersperse(self, item):
        self.items().intersperse(item)

    def invert(self):
        return self.items().map(reversed).dict()

    def items(self):
        return Iter(dict.items(self))

    def iter_keys(self):
        return Iter(self.keys())

    def iter_values(self):
        return Iter(self.values())

    def last(self, predicate=Unset, default=Unset):
        return self.items().last(predicate=predicate, default=default)

    def list(self):
        return List(self.items())

    def map(self, f):
        return self.items().map(f)

    def map_keys(self, f):
        return self.items().map(t(lambda k, v: (f(k), v)))

    def map_values(self, f):
        return self.items().map(t(lambda k, v: (k, f(v))))

    def max(self, key=None, default=Unset):
        return self.items().max(key=key, default=default)

    def min(self, key=None, default=Unset):
        return self.items().min(key=key, default=default)

    def min_max(self, key=None, default=Unset):
        return self.items().min_max(key=key, default=default)

    def only(self, predicate=Unset):
        return self.items().only(predicate=predicate)

    def partition(self, predicate):
        return self.items().partition(predicate=predicate)

    def permutations(self, r=None):
        return self.items().permutations(r=r)

    def powerset(self):
        return self.items().powerset()

    def product(self, *its, repeat=1):
        return self.items().product(*its, repeat=1)

    def put(self, k, v):
        self[k] = v
        return self

    def repeat(self, n=Unset):
        return self.items().repeat(n=n)

    def set(self):
        return Set(self.items())

    def size(self):
        return len(self)

    def sliding(self, size, step=1):
        return self.items().sliding(size, step=step)

    def sliding_by_timestamp(self, size, step=1, stamp=timestamp(time.time)):
        return self.items().sliding_by_timestamp(size, step=step, stamp=stamp)

    def take(self, n):
        return self.items().take(n)

    def take_while(self, predicate=Unset):
        return self.take_while(predicate=predicate)

    def timestamp(self, clock=time.time):
        return self.items().timestamp(clock=clock)

    if tqdm:
        def tqdm(self, *a, **kw):
            return self.items().tqdm(*a, **kw)

    def tuple(self):
        return self.items().tuple()

    def union(self, *others):
        out = self.dict()
        for other in others:
            for key, value in other.items():
                require(out.setdefault(key, value) == value, "mismatched values", key=key)
        return out

    def update(self, E=Unset, **F):
        if E is Unset:
            dict.update(self, **F)
        else:
            dict.update(self, E, **F)
        return self

    def zip(self, *others, strict=False):
        return self.items().zip(*others, strict=strict)

    def zip_longest(self, *others, fillvalue=None):
        return self.items().zip_longest(*others, fillvalue=fillvalue)


class DefaultDict(Dict, defaultdict):
    pass


class Iter:
    def __init__(self, it=()):
        self._it = iter(it)
        self._peek = Unset
        self._peek2 = Unset

    def __add__(self, other):
        return self.chain(other)

    def __contains__(self, item):
        for e in self:
            if e == item:
                return True
        return False

    def __iter__(self):
        if self._peek is not Unset:
            if self._peek2 is not Unset:
                _peek, _peek2, self._peek, self._peek2 = self._peek, self._peek2, Unset, Unset
                yield _peek
                yield _peek2
            else:
                _peek, self._peek = self._peek, Unset
                yield _peek
        yield from self._it

    def __mul__(self, n):
        return self.repeat(n)

    def __next__(self):
        if self._peek is Unset:
            return next(self._it)
        if self._peek2 is Unset:
            _peek, self._peek = self._peek, Unset
            return _peek
        _peek, self._peek, self._peek2 = self._peek, self._peek2, Unset
        return _peek

    def __radd__(self, other):
        return Iter(other).chain(self)

    def __rmul__(self, n):
        return self.repeat(n)

    def accumulate(self, f=add, *, initial=None):
        return Iter(itertools.accumulate(self, f, initial=initial))

    def all(self, key=Unset):
        if key is not Unset:
            self = self.map(key)
        return all(self)

    def any(self, key=Unset):
        if key is not Unset:
            self = self.map(key)
        return any(self)

    def apply(self, f):
        return f(self)

    def apply_and_iter(self, f):
        return Iter(f(self))

    def batch(self, size):
        """
        :param size: max number of items in each batch. the last batch may have fewer items if there are not enough to
                     fill it
        """
        require(size > 0, size=size)

        try:
            from itertools import batched
            return Iter(batched(self, size)).map(Tuple)
        except ImportError:
            return (
                self.enumerate()
                .apply_and_iter(functools.partial(itertools.groupby, key=t(lambda i, _: i // size)))
                .map(itemgetter(1))
                .map(lambda batch: map(itemgetter(1), batch))
                .map(Tuple)
            )

    def chain(self, *its):
        return Iter(itertools.chain(self, *its))

    def combinations(self, r, with_replacement=False):
        combinations = itertools.combinations_with_replacement if with_replacement else itertools.combinations
        return Iter(combinations(self, r))

    def combine_if(self, condition, combinator, *a, **kw):
        if condition:
            return getattr(self, combinator)(*a, **kw)
        return self

    def count(self, key=Unset):
        if key is not Unset:
            self = self.map(key)
        counts = Dict()
        for k in self:
            counts.setdefault(k, 0)
            counts[k] += 1
        return counts

    def cycle(self):
        return Iter(itertools.cycle(self))

    def dict(self):
        return Dict(self)

    def distinct(self, key=Unset):
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

    def do(self, f):
        def gen():
            for item in self:
                f(item)
                yield item

        return Iter(gen())

    def drop(self, n):
        if n >= 0:
            for _ in range(n):
                try:
                    self.next()
                except StopIteration:
                    break
            return self
        else:
            window = deque(self.take(-n))
            while self.has_next():
                window.popleft()
                window.append(self.next())
            return Iter(window)

    def drop_while(self, predicate):
        return Iter(itertools.dropwhile(predicate, self))

    def enumerate(self, start=0):
        return Iter(enumerate(self, start))

    def filter(self, predicate):
        def gen():
            for item in self:
                if predicate(item):
                    yield item

        return Iter(gen())

    def first(self, predicate=Unset, default=Unset):
        try:
            if predicate is Unset:
                return self.next()
            else:
                return self.filter(predicate).next()
        except StopIteration:
            if default is Unset:
                raise
            return default

    def flat_map(self, f):
        def gen():
            for item in self:
                yield from f(item)

        return Iter(gen())

    def flatten(self):
        return self.flat_map(identity)

    def fold(self, initial, f):
        acc = initial
        for item in self:
            acc = f(acc, item)
        return acc

    def fold_while(self, initial, f, predicate):
        require(predicate(initial), "invalid initial value", ValueError)

        acc = initial
        while self.has_next():
            last, acc = acc, f(acc, self.peek())
            if not predicate(acc):
                return last
            self.next()
        return acc

    def for_each(self, f):
        for item in self:
            f(item)

    def group_by(self, key=Unset):
        out = Dict()
        if key is Unset:
            for item in self:
                out.setdefault(item, List()).append(item)
        else:
            for item in self:
                out.setdefault(key(item), List()).append(item)
        return out

    def has_next(self):
        try:
            if self._peek is Unset:
                self._peek = next(self._it)
            return True
        except StopIteration:
            return False

    def has_next2(self):
        try:
            if self._peek is Unset:
                self._peek = next(self._it)
            if self._peek2 is Unset:
                self._peek2 = next(self._it)
            return True
        except StopIteration:
            return False

    def intersperse(self, item):
        def gen():
            if self.has_next():
                yield self.next()
                for e in self:
                    yield item
                    yield e

        return Iter(gen())

    def last(self, predicate=Unset, default=Unset):
        if predicate is not Unset:
            self = self.filter(predicate)
        if self.has_next():
            for item in self:
                pass
            return item
        else:
            if default is Unset:
                raise StopIteration
            return default

    def list(self):
        return List(self)

    def map(self, f):
        return Iter(map(f, self))

    def map_to_keys(self, f):
        return self.map_to_pairs(f).map(reversed).dict()

    def map_to_pairs(self, f):
        return self.map(lambda item: (item, f(item)))

    def map_to_values(self, f):
        return self.map_to_pairs(f).dict()

    def max(self, key=None, default=Unset):
        if not self.has_next() and default is not Unset:
            return default
        return max(self, key=key)

    def min(self, key=None, default=Unset):
        if not self.has_next() and default is not Unset:
            return default
        return min(self, key=key)

    def min_max(self, key=None, default=Unset):
        if not self.has_next():
            require(default is not Unset, "min_max() arg is an empty sequence", _exc=ValueError)
            return default

        if key is None:
            min = max = self.next()
            for item in self:
                if item < min:
                    min = item
                elif item > max:
                    max = item
            return min, max
        else:
            min = max = self.next()
            mink = maxk = key(min)
            for item in self:
                itemk = key(item)
                if itemk < mink:
                    min = item
                    mink = itemk
                elif itemk > maxk:
                    max = item
                    maxk = itemk
            return min, max

    def next(self):
        return next(self)

    def only(self, predicate=Unset):
        if predicate is not Unset:
            self = self.filter(predicate)

        require(self.has_next(), "no item found", _exc=ValueError)
        item = self.next()
        require(not self.has_next(), "too many items found", _exc=ValueError)
        return item

    def partition(self, predicate=identity):
        class Trues:
            def __init__(self, it):
                self.it = it

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
            def __init__(self, it):
                self.it = it

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

    def peek(self, default=Unset):
        if self.has_next():
            return self._peek
        require(default is not Unset, "peek past end of iterator", _exc=ValueError)
        return default

    def peek2(self, default=Unset):
        if self.has_next2():
            return self._peek2
        require(default is not Unset, "peek past end of iterator", _exc=ValueError)
        return default

    def permutations(self, r=None):
        return Iter(itertools.permutations(self, r))

    def powerset(self):
        items = self.tuple()
        return Range(len(items) + 1).flat_map(items.combinations)

    def product(self, *its, repeat=1):
        return Iter(itertools.product(self, *its, repeat=repeat))

    def reduce(self, f):
        require(self.has_next(), "reduce on empty iterator", _exc=ValueError)
        return self.fold(self.next(), f)

    def repeat(self, n=Unset):
        items = self.tuple()
        # without this check, the returned iterator would block forever trying to return the first item
        if n is Unset and not items:
            return Iter()
        return repeated(items, n).flatten()

    def set(self):
        return Set(self)

    def size(self):
        count = 0
        for _ in self:
            count += 1
        return count

    def sliding(self, size, step=1):
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

    def sliding_by_timestamp(self, size, step=1, stamp=timestamp(time.time)):
        require(size > 0, size=size)
        require(step != 0, step=step)

        self = self.map(stamp)
        if not self.has_next():
            return self
        if step < 0:
            # for negative steps, it's easiest to just flip the signs so that we can use the same comparison operations
            step = -step
            self = self.map(t(lambda ts, item: (-ts, item)))

        def gen():
            start = self.peek()[0]
            window = deque()
            while True:
                window.extend(self.take_while(t(lambda ts, _: ts < start + size)))
                while window and window[0][0] < start:
                    window.popleft()
                yield Tuple(window).map(itemgetter(1))
                start += step
                if not self.has_next():
                    break

        return Iter(gen())

    def take(self, n):
        if n >= 0:
            def gen():
                for _ in range(n):
                    try:
                        yield self.next()
                    except StopIteration:
                        pass
        else:
            def gen():
                try:
                    windows = self.sliding(-n)
                    _next = windows.next()[0]
                    while windows.has_next():
                        yield _next
                        _next = windows.next()[0]
                except StopIteration:
                    pass

        return Iter(gen())

    def take_while(self, predicate=Unset):
        # not implemented using itertools.takewhile because it discards the first non-passing element
        if predicate is Unset:
            def gen():
                for item in self:
                    if item:
                        yield item
                    else:
                        self._peek = item
                        break
        else:
            def gen():
                for item in self:
                    if predicate(item):
                        yield item
                    else:
                        self._peek = item
                        break

        return Iter(gen())

    def tee(self, n=2):
        return tuple(map(Iter, itertools.tee(self, n)))

    def timestamp(self, clock=time.time):
        return self.map(timestamp(clock))

    if tqdm:
        def tqdm(self, *a, **kw):
            return Iter(tqdm(self, *a, **kw))

    def transpose(self):
        return Iter(zip(*self, strict=True)).map(Tuple)

    def tuple(self):
        return Tuple(self)

    unzip = transpose

    def zip(self, *others, strict=False):
        return Iter(zip(self, *others, strict=strict))

    def zip_longest(self, *others, fillvalue=None):
        return Iter(itertools.zip_longest(self, *others, fillvalue=fillvalue))


class List(list):
    def __add__(self, other):
        # TODO: remove unnecessary concatenation restrictions
        return List(list.__add__(self, other))

    def __getitem__(self, item):
        if isinstance(item, int):
            return list.__getitem__(self, item)
        return List(list.__getitem__(self, item))

    def __iter__(self):
        return Iter(list.__iter__(self))

    def __mul__(self, other):
        # TODO: avoid copying?
        return List(list.__mul__(self, other))

    def __rmul__(self, other):
        return List(list.__rmul__(self, other))

    def all(self, key=Unset):
        return self.iter().all(key=key)

    def any(self, key=Unset):
        return self.iter().any(key=key)

    def append(self, item):
        list.append(self, item)
        return self

    def apply(self, f):
        return f(self)

    def apply_and_iter(self, f):
        return Iter(f(self))

    def batch(self, size):
        return self.iter().batch(size)

    def chain(self, *its):
        return self.iter().chain(*its)

    def clear(self):
        list.clear(self)
        return self

    def combinations(self, r, with_replacement=False):
        return self.iter().combinations(r, with_replacement=with_replacement)

    def combine_if(self, condition, combinator, *a, **kw):
        if condition:
            return getattr(self, combinator)(*a, **kw)
        return self

    def count(self, key=Unset):
        return self.iter().count(key=key)

    def cycle(self):
        return self.iter().cycle()

    def dict(self):
        return Dict(self)

    def discard(self, item):
        try:
            self.remove(item)
            return True
        except ValueError:
            return False

    def distinct(self, key=Unset):
        return self.iter().distinct(key=key)

    def do(self, f):
        return self.iter().do(f)

    def drop(self, n):
        return self[n:]

    def drop_while(self, predicate):
        return self.iter().drop_while(predicate)

    def enumerate(self, start=0):
        return self.iter().enumerate(start=start)

    def extend(self, iterable):
        list.extend(self, iterable)
        return self

    def filter(self, predicate=Unset):
        return self.iter().filter(predicate=predicate)

    def first(self, predicate=Unset, default=Unset):
        return self.iter().first(predicate=predicate, default=default)

    def flat_map(self, f):
        return self.iter().flat_map(f)

    def flatten(self):
        return self.iter().flatten()

    def fold(self, initial, f):
        return self.iter().fold(initial, f)

    def fold_while(self, initial, f, predicate):
        return self.iter().fold_while(initial, f, predicate)

    def for_each(self, f):
        return self.iter().for_each(f)

    def group_by(self, key=Unset):
        return self.iter().group_by(key=key)

    def insert(self, index, item):
        list.insert(self, index, item)
        return self

    def intersperse(self, item):
        return self.iter().intersperse(item)

    def iter(self):
        return iter(self)

    def last(self, predicate=Unset, default=Unset):
        return self.iter().last(predicate=predicate, default=default)

    def list(self):
        return List(self)

    def map(self, f):
        return self.iter().map(f)

    def map_to_keys(self, f):
        return self.iter().map_to_keys(f)

    def map_to_pairs(self, f):
        return self.iter().map_to_pairs(f)

    def map_to_values(self, f):
        return self.iter().map_to_values(f)

    def max(self, key=None, default=Unset):
        return self.iter().max(key=key, default=default)

    def min(self, key=None, default=Unset):
        return self.iter().min(key=key, default=default)

    def min_max(self, key=None, default=Unset):
        return self.iter().min_max(key=key, default=default)

    def only(self, predicate=Unset):
        return self.iter().only(predicate=predicate)

    def partition(self, predicate=identity):
        return self.iter().partition(predicate=predicate)

    def permutations(self, r=None):
        return self.iter().permutations(r)

    def powerset(self):
        return self.iter().powerset()

    def product(self, *its, repeat=1):
        return self.iter().product(*its, repeat=repeat)

    def reduce(self, f):
        return self.iter().reduce(f)

    def repeat(self, n=Unset):
        return self.iter().repeat(n=n)

    def reverse(self):
        list.reverse(self)
        return self

    def set(self):
        return Set(self)

    def size(self):
        return len(self)

    def sliding(self, size, step=1):
        return self.iter().sliding(size, step=step)

    def sliding_by_timestamp(self, size, step=1, stamp=timestamp(time.time)):
        return self.iter().sliding_by_timestamp(size, step=step, stamp=stamp)

    def sort(self, key=None, reverse=False):
        list.sort(self, key=key, reverse=reverse)
        return self

    def take(self, n):
        return self[:n]

    def take_while(self, predicate=Unset):
        return self.iter().take_while(predicate=predicate)

    def timestamp(self, clock=time.time):
        return self.iter().timestamp(clock=clock)

    if tqdm:
        def tqdm(self, *a, **kw):
            return self.iter().tqdm(*a, **kw)

    def transpose(self):
        return self.iter().transpose()

    def tuple(self):
        return Tuple(self)

    unzip = transpose

    def zip(self, *others, strict=False):
        return self.iter().zip(*others, strict=strict)

    def zip_longest(self, *others, fillvalue=None):
        return self.iter().zip_longest(*others, fillvalue=fillvalue)


class Range:
    def __init__(self, *a, **kw):
        if len(a) == 1 and isinstance(a[0], range):
            require(not kw)
            self._range = a[0]
        else:
            self._range = range(*a, **kw)

    def __contains__(self, item):
        return item in self._range

    def __eq__(self, other):
        if isinstance(other, range):
            return self._range == range
        elif isinstance(other, Range):
            return self._range == other._range
        return False

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._range[item]
        r = self._range[item]
        return Range(r.start, r.stop, r.step)

    def __hash__(self):
        return hash(self._range)

    def __iter__(self):
        return Iter(self._range)

    def __len__(self):
        return len(self._range)

    def __repr__(self):
        return repr(self._range).title()

    def __reversed__(self):
        return Range(reversed(self._range))

    @property
    def start(self):
        return self._range.start

    @property
    def stop(self):
        return self._range.stop

    @property
    def step(self):
        return self._range.step

    def all(self, key=Unset):
        return self.iter().all(key=key)

    def any(self, key=Unset):
        return self.iter().any(key=key)

    def apply(self, f):
        return f(self)

    def apply_and_iter(self, f):
        return Iter(f(self))

    def batch(self, size):
        # TODO: this can be specialized to output ranges
        return self.iter().batch(size)

    def chain(self, *its):
        return self.iter().chain(*its)

    def combinations(self, r, with_replacement=False):
        return self.iter().combinations(r, with_replacement=with_replacement)

    def combine_if(self, condition, combinator, *a, **kw):
        if condition:
            return getattr(self, combinator)(*a, **kw)
        return self

    def count(self, key=Unset):
        return self.iter().count(key=key)

    def cycle(self):
        return self.iter().cycle()

    def do(self, f):
        return self.iter().do(f)

    def drop(self, n):
        return Range(self._range[n:])

    def drop_while(self, predicate):
        return self.iter().drop_while(predicate)

    def enumerate(self, start=0):
        return self.iter().enumerate(start=start)

    def filter(self, predicate):
        return self.iter().filter(predicate)

    def first(self, predicate=Unset, default=Unset):
        return self.iter().first(predicate=predicate, default=default)

    def flat_map(self, f):
        return self.iter().flat_map(f)

    def flatten(self):
        return self.iter().flatten()

    def fold(self, initial, f):
        return self.fold(initial, f)

    def fold_while(self, initial, f, predicate):
        return self.fold_while(initial, f, predicate)

    def for_each(self, f):
        return self.iter().for_each(f)

    def group_by(self, key=Unset):
        return self.iter().group_by(key=key)

    def intersperse(self, item):
        return self.iter().intersperse(item)

    def last(self, predicate=Unset, default=Unset):
        return self.reverse().first(predicate=predicate, default=default)

    def list(self):
        return List(self)

    def map(self, f):
        return self.iter().map(f)

    def map_to_keys(self, f):
        return self.iter().map_to_keys(f)

    def map_to_pairs(self, f):
        return self.iter().map_to_pairs(f)

    def map_to_values(self, f):
        return self.iter().map_to_values(f)

    def max(self, key=None, default=Unset):
        return self.iter().max(key=key, default=default)

    def min(self, key=None, default=Unset):
        return self.iter().min(key=key, default=default)

    def min_max(self, key=None, default=Unset):
        return self.iter().min_max(key=key, default=default)

    def only(self, predicate=Unset):
        return self.iter().only(predicate=predicate)

    def partition(self, predicate=Unset):
        return self.iter().partition(predicate=predicate)

    def permutations(self, r=None):
        return self.iter().permutations(r=r)

    def powerset(self):
        return self.iter().powerset()

    def product(self, *its, repeat=1):
        return self.iter().product(*its, repeat=repeat)

    def reduce(self, f):
        return self.iter().reduce(f)

    def repeat(self, n=Unset):
        return self.iter().repeat(n=n)

    def reverse(self):
        return Range(reversed(self._range))

    def set(self):
        return Set(self)

    def size(self):
        return len(self)

    def sliding(self, size, step=1):
        # TODO: this can be specialized to output ranges
        return self.iter().sliding(size, step=step)

    def sliding_by_timestamp(self, size, step=1, stamp=timestamp(time.time)):
        return self.iter().sliding_by_timestamp(size, step=step, stamp=stamp)

    def take(self, n):
        return Range(self._range[:n])

    def take_while(self, predicate=Unset):
        return self.iter().take_while(predicate=predicate)

    def timestamp(self, clock=time.time):
        return self.iter().timestamp(clock=clock)

    if tqdm:
        def tqdm(self, *a, **kw):
            return self.iter().tqdm(*a, **kw)

    def tuple(self):
        return Tuple(self)

    def zip(self, *others, strict=False):
        return self.iter().zip(*others, strict=strict)

    def zip_longest(self, *others, fillvalue=None):
        return self.iter().zip_longest(*others, fillvalue=fillvalue)

    def index(self, value):
        return self._range.index(value)

    def iter(self):
        return iter(self)


class Set(set):
    def __iter__(self):
        return Iter(set.__iter__(self))

    def add(self, item):
        if item in self:
            return False
        set.add(self, item)
        return True

    def all(self, key=Unset):
        return self.iter().all(key=key)

    def any(self, key=Unset):
        return self.iter().any(key=key)

    def apply(self, f):
        return f(self)

    def apply_and_iter(self, f):
        return Iter(f(self))

    def batch(self, size):
        return self.iter().batch(size)

    def chain(self, *its):
        return self.iter().chain(*its)

    def clear(self):
        set.clear(self)
        return self

    def combinations(self, r, with_replacement=False):
        return self.iter().combinations(r, with_replacement=with_replacement)

    def combine_if(self, condition, combinator, *a, **kw):
        if condition:
            return getattr(self, combinator)(*a, **kw)
        return self

    def count(self, key=Unset):
        return self.iter().count(key=key)

    def cycle(self):
        return self.iter().cycle()

    def dict(self):
        return Dict(self)

    def discard(self, item):
        if item in self:
            self.remove(item)
            return True

    def distinct(self, key=Unset):
        return self.iter().distinct(key=key)

    def do(self, f):
        return self.iter().do(f)

    def drop(self, n):
        return self.iter().drop(n)

    def drop_while(self, predicate):
        return self.iter().drop_while(predicate)

    def enumerate(self, start=0):
        return self.iter().enumerate(start=start)

    def filter(self, predicate=Unset):
        return self.iter().filter(predicate=predicate)

    def first(self, predicate=Unset, default=Unset):
        return self.iter().first(predicate=predicate, default=default)

    def flat_map(self, f):
        return self.iter().flat_map(f)

    def flatten(self):
        return self.iter().flatten()

    def fold(self, initial, f):
        return self.iter().fold(initial, f)

    def fold_while(self, initial, f, predicate):
        return self.iter().fold_while(initial, f, predicate)

    def for_each(self, f):
        return self.iter().for_each(f)

    def group_by(self, key=Unset):
        return self.iter().group_by(key=key)

    def intersperse(self, item):
        return self.iter().intersperse(item)

    def iter(self):
        return iter(self)

    def last(self, predicate=Unset, default=Unset):
        return self.iter().last(predicate=predicate, default=default)

    def list(self):
        return List(self)

    def map(self, f):
        return self.iter().map(f)

    def map_to_keys(self, f):
        return self.iter().map_to_keys(f)

    def map_to_pairs(self, f):
        return self.iter().map_to_pairs(f)

    def map_to_values(self, f):
        return self.iter().map_to_values(f)

    def max(self, key=None, default=Unset):
        return self.iter().max(key=key, default=default)

    def min(self, key=None, default=Unset):
        return self.iter().min(key=key, default=default)

    def min_max(self, key=None, default=Unset):
        return self.iter().min_max(key=key, default=default)

    def only(self, predicate=Unset):
        return self.iter().only(predicate=predicate)

    def partition(self, predicate=Unset):
        return self.iter().partition(predicate=predicate)

    def permutations(self, r=None):
        return self.iter().permutations(r=r)

    def powerset(self):
        return self.iter().powerset()

    def product(self, *its, repeat=1):
        return self.iter().product(*its, repeat=repeat)

    def reduce(self, f):
        return self.iter().reduce(f)

    def repeat(self, n=Unset):
        return self.iter().repeat(n=n)

    def set(self):
        return Set(self)

    def size(self):
        return len(self)

    def sliding(self, size, step=1):
        return self.iter().sliding(size, step=step)

    def sliding_by_timestamp(self, size, step=1, stamp=timestamp(time.time)):
        return self.iter().sliding_by_timestamp(size, step=step, stamp=stamp)

    def take(self, n):
        return self.iter().take(n)

    def take_while(self, predicate=Unset):
        return self.iter().take_while(predicate=predicate)

    def timestamp(self, clock=time.time):
        return self.iter().timestamp(clock=clock)

    if tqdm:
        def tqdm(self, *a, **kw):
            return self.iter().tqdm(*a, **kw)

    def transpose(self):
        return self.iter().transpose()

    def tuple(self):
        return Tuple(self)

    unzip = transpose

    def update(self, *s):
        set.update(self, *s)
        return s

    def zip(self, *others, strict=False):
        return self.iter().zip(*others, strict=strict)

    def zip_longest(self, *others, fillvalue=None):
        return self.iter().zip_longest(*others, fillvalue=fillvalue)


class Tuple(tuple):
    def __add__(self, other):
        return Tuple(tuple.__add__(self, other))

    def __mul__(self, other):
        return Tuple(tuple.__mul__(self, other))

    def __rmul__(self, other):
        return Tuple(tuple.__rmul__(self, other))

    def __getitem__(self, item):
        if isinstance(item, int):
            return tuple.__getitem__(self, item)
        return Tuple(tuple.__getitem__(self, item))

    def __iter__(self):
        return Iter(tuple.__iter__(self))

    def all(self, key=Unset):
        return self.iter().all(key=key)

    def any(self, key=Unset):
        return self.iter().any(key=key)

    def apply(self, f):
        return f(self)

    def apply_and_iter(self, f):
        return Iter(f(self))

    def batch(self, size):
        return self.iter().batch(size)

    def chain(self, *its):
        return self.iter().chain(*its)

    def combinations(self, r, with_replacement=False):
        return self.iter().combinations(r, with_replacement=with_replacement)

    def combine_if(self, condition, combinator, *a, **kw):
        if condition:
            return getattr(self, combinator)(*a, **kw)
        return self

    def count(self, key=Unset):
        return self.iter().count(key=key)

    def cycle(self):
        return self.iter().cycle()

    def dict(self):
        return Dict(self)

    def distinct(self, key=Unset):
        return self.iter().distinct(key=key)

    def do(self, f):
        return self.iter().do(f)

    def drop(self, n):
        return self[n:]

    def drop_while(self, predicate):
        return self.iter().drop_while(predicate)

    def enumerate(self, start=0):
        return self.iter().enumerate(start=start)

    def filter(self, predicate=Unset):
        return self.iter().filter(predicate=predicate)

    def first(self, predicate=Unset, default=Unset):
        return self.iter().first(predicate=predicate, default=default)

    def flat_map(self, f):
        return self.iter().flat_map(f)

    def flatten(self):
        return self.iter().flatten()

    def fold(self, initial, f):
        return self.iter().fold(initial, f)

    def fold_while(self, initial, f, predicate):
        return self.iter().fold_while(initial, f, predicate)

    def for_each(self, f):
        return self.iter().for_each(f)

    def group_by(self, key=Unset):
        return self.iter().group_by(key=key)

    def intersperse(self, item):
        return self.iter().intersperse(item)

    def iter(self):
        return Iter(self)

    def last(self, predicate=Unset, default=Unset):
        return self.iter().last(predicate=predicate, default=default)

    def list(self):
        return List(self)

    def map(self, f):
        return self.iter().map(f)

    def map_to_keys(self, f):
        return self.iter().map_to_keys(f)

    def map_to_pairs(self, f):
        return self.iter().map_to_pairs(f)

    def map_to_values(self, f):
        return self.iter().map_to_values(f)

    def max(self, key=None, default=Unset):
        return self.iter().max(key=key, default=default)

    def min(self, key=None, default=Unset):
        return self.iter().min(key=key, default=default)

    def min_max(self, key=None, default=Unset):
        return self.iter().min_max(key=key, default=default)

    def only(self, predicate=Unset):
        return self.iter().only(predicate=predicate)

    def partition(self, predicate=identity):
        return self.iter().partition(predicate=predicate)

    def permutations(self, r=None):
        return self.iter().permutations(r)

    def powerset(self):
        return self.iter().powerset()

    def product(self, *its, repeat=1):
        return self.iter().product(*its, repeat=repeat)

    def reduce(self, f):
        return self.iter().reduce(f)

    def repeat(self, n=Unset):
        return self.iter().repeat(n=n)

    def reverse(self):
        return Tuple(reversed(self))

    def set(self):
        return Set(self)

    def size(self):
        return len(self)

    def sliding(self, size, step=1):
        return self.iter().sliding(size, step=step)

    def sliding_by_timestamp(self, size, step=1, stamp=timestamp(time.time)):
        return self.iter().sliding_by_timestamp(size, step=step, stamp=stamp)

    def sorted(self, key=None, reverse=False):
        return Tuple(sorted(self, key=key, reverse=reverse))

    def take(self, n):
        return self[:n]

    def take_while(self, predicate=Unset):
        return self.iter().take_while(predicate=predicate)

    def timestamp(self, clock=time.time):
        return self.iter().timestamp(clock=clock)

    if tqdm:
        def tqdm(self, *a, **kw):
            return self.iter().tqdm(*a, **kw)

    def transpose(self):
        return self.iter().transpose()

    def tuple(self):
        return Tuple(self)

    unzip = transpose

    def zip(self, *others, strict=False):
        return self.iter().zip(*others, strict=strict)

    def zip_longest(self, *others, fillvalue=None):
        return self.iter().zip_longest(*others, fillvalue=fillvalue)
