from collections import defaultdict
from operator import add, mul, itemgetter
from unittest.mock import patch

import pytest

from ajprax.collections import DefaultDict, Dict, Iter, List, Range, Set, Tuple, count, repeated, repeatedly, timestamp, \
    wrap, DictKeys, DictValues
from ajprax.hof import identity, t
from ajprax.require import RequirementException
from tests import by_iter_eq, by_eq, double, is_even, is_odd, less_than, should_raise, tower, greater_than, iter_eq, \
    key_less_than, key_is_odd, double_key, double_value, by_member_eq

skip = pytest.mark.skip


def test_coverage():
    """
    Asserts that all functions, and class methods in collections have a corresponding test, and that there are no tests
    for functions, classes, and methods that do not exist.
    """

    def functions_and_classes(module):
        from inspect import isfunction, isclass

        functions = set()
        classes = dict()

        for name in dir(module):
            value = getattr(module, name)
            if isfunction(value) and value.__module__ == module.__name__:
                functions.add(name)
            elif isclass(value) and value.__module__ == module.__name__:
                classes[name] = methods(value)

        return functions, classes

    def methods(cls):
        from inspect import isfunction

        methods = set()
        for name in dir(cls):
            value = getattr(cls, name)
            if isfunction(value) or isinstance(value, property):
                methods.add(name)
        return methods

    import ajprax, tests

    functions, classes = functions_and_classes(ajprax.collections)
    test_functions, test_classes = functions_and_classes(tests.collections)
    tested_functions = {f[5:] for f in test_functions} - {"coverage"}
    tested_classes = {cls[4:]: {method[5:] for method in methods} for cls, methods in test_classes.items()}

    assert not functions.symmetric_difference(tested_functions)

    for cls, methods in classes.items():
        if cls in ("DefaultDict", "DictKeys", "DictValues"):
            continue  # these are effectively tested by Dict, Set, and Tuple
        tested_methods = tested_classes[cls]
        assert not methods.symmetric_difference(tested_methods), cls


def test_count():
    def test(expected, *a, **kw):
        actual = count(*a, **kw).take(3)
        assert iter_eq(actual, expected)

    test([0, 1, 2])
    test([1, 2, 3], 1)
    test([1, 2, 3], start=1)
    test([0, 2, 4], step=2)
    test([1, 3, 5], 1, 2)

    assert count().drop(1000).next() == 1000


def test_repeated():
    def test(expected, *a, **kw):
        actual = repeated(*a, **kw).take(3)
        assert iter_eq(actual, expected)

    test([0, 0, 0], 0)
    test([1, 1, 1], 1)
    test([0, 0, 0], 0, 5)
    test([0, 0, 0], 0, n=5)
    test([0, 0, 0], 0, n=4)
    test([0, 0, 0], 0, n=3)
    test([0, 0], 0, n=2)
    test([0], 0, n=1)
    test([], 0, n=0)


def test_repeatedly():
    def test(expected, *a, **kw):
        actual = repeatedly(*a, **kw).take(3)
        assert iter_eq(actual, expected)

    test([0, 0, 0], lambda: 0)
    test([1, 1, 1], lambda: 1)
    test([0, 0, 0], lambda: 0, n=5)
    test([0, 0, 0], lambda: 0, n=4)
    test([0, 0, 0], lambda: 0, n=3)
    test([0, 0], lambda: 0, n=2)
    test([0], lambda: 0, n=1)
    test([], lambda: 0, n=0)

    def inc():
        nonlocal i
        i += 1
        return i

    i = 0
    test([1, 2, 3], inc)


def test_timestamp():
    def clock():
        nonlocal now
        now += 1
        return now

    now = 0
    stamp = timestamp(clock)
    assert stamp(0) == (1, 0)
    assert stamp(10) == (2, 10)
    assert stamp(None) == (3, None)


def test_wrap():
    def test(expected, it):
        actual = wrap(it)
        assert actual == expected
        assert type(actual) == type(expected)
        if isinstance(it, (DefaultDict, Dict, DictKeys, DictValues, Iter, List, Range, Set, Tuple)):
            assert actual is it

    test(DefaultDict(int), defaultdict(int))
    test(DefaultDict(int), DefaultDict(int))
    dd = defaultdict(int)
    dd.update(a=1, b=2)
    test(DefaultDict(int).update(a=1, b=2), dd)
    test(DefaultDict(int).update(a=1, b=2), DefaultDict(int).update(a=1, b=2))
    test(Dict(), dict())
    test(Dict(), Dict())
    test(Dict(a=1, b=2), dict(a=1, b=2))
    test(Dict(a=1, b=2), Dict(a=1, b=2))
    test(DictKeys(dict().keys()), dict().keys())
    test(DictKeys(dict().keys()), Dict().keys())
    test(DictKeys(dict(a=1, b=2).keys()), dict(a=1, b=2).keys())
    test(DictKeys(dict(a=1, b=2).keys()), Dict(a=1, b=2).keys())
    test(List(), [])
    test(List(), List())
    test(List([1, 2, 3]), [1, 2, 3])
    test(List([1, 2, 3]), List([1, 2, 3]))
    test(Range(5), range(5))
    test(Range(5), Range(5))
    test(Set(), set())
    test(Set(), Set())
    test(Set({1, 2, 3}), {1, 2, 3})
    test(Set({1, 2, 3}), Set({1, 2, 3}))
    test(Tuple(), tuple())
    test(Tuple(), Tuple())
    test(Tuple((1, 2, 3)), (1, 2, 3))
    test(Tuple((1, 2, 3)), Tuple((1, 2, 3)))


class TestDict:
    def test___iter__(self):
        it = iter(Dict(key="value"))
        assert isinstance(it, Iter)
        assert next(it) == "key"

    def test_all(self):
        def test(expected, d, *a, **kw):
            actual = Dict(d).all(*a, **kw)
            assert actual == expected

        test(True, {})
        test(True, {}, lambda _: False)
        test(False, {"a": 1}, lambda _: False)
        test(True, {"a": 1}, t(lambda _, v: is_odd(v)))
        test(False, {"a": 1, "b": 2}, t(lambda _, v: is_odd(v)))

    def test_any(self):
        def test(expected, d, *a, **kw):
            actual = Dict(d).any(*a, **kw)
            assert actual == expected

        test(False, {})
        test(False, {}, lambda _: False)
        test(False, {"a": 1}, lambda _: False)
        test(True, {"a": 1}, t(lambda _, v: is_odd(v)))
        test(True, {"a": 1, "b": 2}, t(lambda _, v: is_odd(v)))

    def test_apply(self):
        test = by_eq(Dict, "apply")

        test(False, [], bool)
        test(True, [("a", 1)], bool)
        test(True, [], all)
        test(True, [("a", 1)], all)
        test(False, [], lambda d: "a" in d)
        test(True, [("a", 1)], lambda d: "a" in d)

    def test_apply_and_wrap(self):
        test = by_iter_eq(Dict, "apply_and_wrap")

        test([], [], lambda d: d.keys())
        test(["a"], [("a", 1)], lambda d: d.keys())
        test([1], [("a", 1)], lambda d: d.values())

    def test_batch(self):
        test = by_iter_eq(Dict, "batch")

        test([{"a": 1}, {"b": 2}], [("a", 1), ("b", 2)], 1)
        for i in range(2, 5):
            test([], [], i)
            test([{"a": 1}], [("a", 1)], i)
            test([{"a": 1, "b": 2}], [("a", 1), ("b", 2)], i)

    def test_chain(self):
        test = by_iter_eq(Dict, "chain")

        test([], [])
        test([], [], [])
        test([], [], [], [])
        test([("a", 1)], [("a", 1)], [])
        test([("a", 1), ("b", 2)], [("a", 1), ("b", 2)], [])
        test([("a", 1), ("b", 2), 3], [("a", 1), ("b", 2)], [3])

    def test_clear(self):
        test = by_eq(Dict, "clear")

        test({}, [])
        test({}, [("a", 1)])
        test({}, [("a", 1), ("b", 2)])

    def test_combinations(self):
        test = by_iter_eq(Dict, "combinations")

        a, b, c = enumerate("abc")

        test([{}], [], 0)
        test([{}], [a], 0)
        test([{}], [a, b], 0)
        test([{}], [b, a], 0)
        test([{}], [a, b, c], 0)
        test([], [], 1)
        test([dict([a])], [a], 1)
        test([dict([a]), dict([b])], [a, b], 1)
        test([dict([b]), dict([a])], [b, a], 1)
        test([dict([a]), dict([b]), dict([c])], [a, b, c], 1)
        test([], [], 2)
        test([], [a], 2)
        test([dict([a, b])], [a, b], 2)
        test([dict([b, a])], [b, a], 2)
        test([dict([a, b]), dict([a, c]), dict([b, c])], [a, b, c], 2)
        test([{}], [], 0, with_replacement=True)
        test([{}], [a], 0, with_replacement=True)
        test([{}], [a, b], 0, with_replacement=True)
        test([{}], [b, a], 0, with_replacement=True)
        test([{}], [a, b, c], 0, with_replacement=True)
        test([], [], 1, with_replacement=True)
        test([dict([a])], [a], 1, with_replacement=True)
        test([dict([a]), dict([b])], [a, b], 1, with_replacement=True)
        test([dict([b]), dict([a])], [b, a], 1, with_replacement=True)
        test([dict([a]), dict([b]), dict([c])], [a, b, c], 1, with_replacement=True)
        test([], [], 2, with_replacement=True)
        test([dict([a, a])], [a], 2, with_replacement=True)
        test([dict([a, a]), dict([a, b]), dict([b, b])], [a, b], 2, with_replacement=True)
        test([dict([b, b]), dict([b, a]), dict([a, a])], [b, a], 2, with_replacement=True)
        test([dict([a, a]), dict([a, b]), dict([a, c]), dict([b, b]), dict([b, c]), dict([c, c])], [a, b, c], 2,
             with_replacement=True)

    def test_combine_if(self):
        def test(expected, items, *a, **kw):
            actual = Dict(items).combine_if(*a, **kw).dict()
            assert actual == expected

        a, b, c = enumerate("abc")

        test({}, [], False, "map_keys", double)
        test({}, [], True, "map_keys", double)
        test({0: "a"}, [a], False, "map_keys", double)
        test({0: "a"}, [a], True, "map_keys", double)
        test({1: "b"}, [b], False, "map_keys", double)
        test({2: "b"}, [b], True, "map_keys", double)
        test({0: "a", 1: "b"}, [a, b], False, "map_keys", double)
        test({0: "a", 2: "b"}, [a, b], True, "map_keys", double)
        test({0: "a", 1: "b", 2: "c"}, [a, b, c], False, "map_keys", double)
        test({0: "a", 2: "b", 4: "c"}, [a, b, c], True, "map_keys", double)

    def test_copy(self):
        def test(items):
            original = Dict(items)
            copy = original.copy()
            assert copy == original
            assert copy is not original
            assert isinstance(copy, Dict)

        a, b, c = enumerate("abc")
        test([])
        test([a])
        test([b])
        test([a, b])
        test([b, a])
        test([a, b, c])

    def test_count(self):
        test = by_eq(Dict, "count")

        a, b, c = enumerate("abc")

        test({}, [], identity)
        test({a: 1}, [a], identity)
        test({a: 1, b: 1}, [a, b], identity)
        test({a: 1, (1, "a"): 1}, [a, (1, "a")], identity)
        test({a: 1, (1, "a"): 1, (2, "b"): 1}, [a, (1, "a"), (2, "b")], identity)

        test({}, [], itemgetter(0))
        test({0: 1}, [a], itemgetter(0))
        test({0: 1, 1: 1}, [a, b], itemgetter(0))
        test({0: 1, 1: 1}, [a, (1, "a")], itemgetter(0))
        test({0: 1, 1: 1, 2: 1}, [a, (1, "a"), (2, "b")], itemgetter(0))

        test({}, [], itemgetter(1))
        test({"a": 1}, [a], itemgetter(1))
        test({"a": 1, "b": 1}, [a, b], itemgetter(1))
        test({"a": 2}, [a, (1, "a")], itemgetter(1))
        test({"a": 2, "b": 1}, [a, (1, "a"), (2, "b")], itemgetter(1))

    def test_cycle(self):
        def test(expected, items):
            actual = Dict(items).cycle().take(5)
            assert iter_eq(actual, expected)

        a, b, c = enumerate("abc")

        test([], [])
        test([a, a, a, a, a], [a])
        test([b, b, b, b, b], [b])
        test([a, b, a, b, a], [a, b])
        test([b, a, b, a, b], [b, a])
        test([a, b, c, a, b], [a, b, c])

    def test_default_dict(self):
        def test(items, *a, **kw):
            result = Dict(items).default_dict(*a, **kw)
            assert isinstance(result, DefaultDict)
            assert dict(result) == dict(items)

        a, b, c = enumerate("abc")

        test([], int)
        test([a], int)
        test([a, b], int)
        test([], list)
        test([("a", [1, 2])], list)
        test([("a", [1, 2]), ("b", [3, 4])], list)
        test([], str)
        test([("a", "hello")], str)
        test([("a", "hello"), ("b", "world")], str)

        result = Dict({"a": 1}).default_dict(int)
        assert result["a"] == 1
        assert result["b"] == 0

        result = Dict({}).default_dict(list)
        result["new_list"].append(1)
        assert result["new_list"] == [1]

    def test_dict(self):
        def test(items):
            d1 = Dict(items)
            d2 = d1.dict()
            assert d1 == d2
            assert d1 is d2

        a, b, c = enumerate("abc")

        test([])
        test([a])
        test([b])
        test([a, b])
        test([b, a])
        test([a, b, c])

    def test_distinct(self):
        test = by_eq(Dict, "distinct")

        a, b, c = enumerate("abc")

        test(dict(), [], itemgetter(1))
        test(dict([a]), [a], itemgetter(1))
        test(dict([b]), [b], itemgetter(1))
        test(dict([a, b]), [a, b], itemgetter(1))
        test(dict([b, a]), [b, a], itemgetter(1))
        test(dict([a, b, c]), [a, b, c], itemgetter(1))
        test(dict([a]), [a, a], itemgetter(1))
        test(dict([a, b]), [a, a, b], itemgetter(1))
        test(dict([a, b]), [a, b, a], itemgetter(1))

        def odd_letter_value(k, v):
            return ord(v) % 2

        test(dict(), [], key=t(odd_letter_value))
        test(dict([a]), [a], key=t(odd_letter_value))
        test(dict([b]), [b], key=t(odd_letter_value))
        test(dict([a, b]), [a, b], key=t(odd_letter_value))
        test(dict([b, a]), [b, a], key=t(odd_letter_value))
        test(dict([a, b]), [a, b, c], key=t(odd_letter_value))
        test(dict([a]), [a, a], key=t(odd_letter_value))
        test(dict([a, b]), [a, a, b], key=t(odd_letter_value))
        test(dict([a, b]), [a, b, a], key=t(odd_letter_value))

    def test_distinct_values(self):
        test = by_iter_eq(Dict, "distinct_values")

        a, b, c = enumerate("abc")

        test(dict(), [])
        test(dict([a]), [a])
        test(dict([b]), [b])
        test(dict([a, b]), [a, b])
        test(dict([b, a]), [b, a])
        test(dict([a, b, c]), [a, b, c])
        test(dict([a]), [a, a])
        test(dict([a, b]), [a, a, b])
        test(dict([a, b]), [a, b, a])
        test(dict(), [], key=lambda c: ord(c) % 2)
        test(dict([a]), [a], key=lambda c: ord(c) % 2)
        test(dict([b]), [b], key=lambda c: ord(c) % 2)
        test(dict([a, b]), [a, b], key=lambda c: ord(c) % 2)
        test(dict([b, a]), [b, a], key=lambda c: ord(c) % 2)
        test(dict([a, b]), [a, b, c], key=lambda c: ord(c) % 2)
        test(dict([a]), [a, a], key=lambda c: ord(c) % 2)
        test(dict([a, b]), [a, a, b], key=lambda c: ord(c) % 2)
        test(dict([a, b]), [a, b, a], key=lambda c: ord(c) % 2)

    def test_do(self):
        def test(items):
            actual = []

            def f(item):
                actual.append(item)

            assert iter_eq(Iter(items).do(f), items)
            assert iter_eq(actual, items)

        a, b, c = enumerate("abc")

        test([])
        test([a])
        test([b])
        test([a, b])
        test([b, a])
        test([a, b, c])

    def test_drop(self):
        test = by_eq(Dict, "drop")

        a, b, c = enumerate("abc")

        test(dict([]), [], 0)
        test(dict([]), [], 1)
        test(dict([]), [], 4)
        test(dict([a]), [a], 0)
        test(dict([]), [a], 1)
        test(dict([]), [a], 4)
        test(dict([b]), [b], 0)
        test(dict([]), [b], 1)
        test(dict([]), [b], 4)
        test(dict([a, b]), [a, b], 0)
        test(dict([b]), [a, b], 1)
        test(dict([]), [a, b], 4)
        test(dict([b, a]), [b, a], 0)
        test(dict([a]), [b, a], 1)
        test(dict([]), [b, a], 4)
        test(dict([a, b, c]), [a, b, c], 0)
        test(dict([b, c]), [a, b, c], 1)
        test(dict([]), [a, b, c], 4)

    def test_drop_while(self):
        test = by_eq(Dict, "drop_while")

        a, b, c = enumerate("abc")

        test(dict([]), [], t(key_less_than(1)))
        test(dict([]), [a], t(key_less_than(1)))
        test(dict([b]), [b], t(key_less_than(1)))
        test(dict([b]), [a, b], t(key_less_than(1)))
        test(dict([b, a]), [b, a], t(key_less_than(1)))
        test(dict([b, c]), [a, b, c], t(key_less_than(1)))
        test(dict([]), [], t(key_less_than(2)))
        test(dict([]), [a], t(key_less_than(2)))
        test(dict([]), [b], t(key_less_than(2)))
        test(dict([]), [a, b], t(key_less_than(2)))
        test(dict([]), [b, a], t(key_less_than(2)))
        test(dict([c]), [a, b, c], t(key_less_than(2)))

    def test_enumerate(self):
        test = by_iter_eq(Dict, "enumerate")

        a, b, c = enumerate("abc")

        test([], [])
        test([(0, a)], [a])
        test([(0, b)], [b])
        test([(0, a), (1, b)], [a, b])
        test([(0, b), (1, a)], [b, a])
        test([(0, a), (1, b), (2, c)], [a, b, c])
        test([], [], start=2)
        test([(2, a)], [a], start=2)
        test([(2, b)], [b], start=2)
        test([(2, a), (3, b)], [a, b], start=2)
        test([(2, b), (3, a)], [b, a], start=2)
        test([(2, a), (3, b), (4, c)], [a, b, c], start=2)

    def test_filter(self):
        test = by_eq(Dict, "filter")

        a, b, c = enumerate("abc")

        test(dict([]), [], t(key_less_than(1)))
        test(dict([a]), [a], t(key_less_than(1)))
        test(dict([]), [b], t(key_less_than(1)))
        test(dict([a]), [a, b], t(key_less_than(1)))
        test(dict([a]), [b, a], t(key_less_than(1)))
        test(dict([a]), [a, b, c], t(key_less_than(1)))
        test(dict([]), [], t(key_less_than(2)))
        test(dict([a]), [a], t(key_less_than(2)))
        test(dict([b]), [b], t(key_less_than(2)))
        test(dict([a, b]), [a, b], t(key_less_than(2)))
        test(dict([b, a]), [b, a], t(key_less_than(2)))
        test(dict([a, b]), [a, b, c], t(key_less_than(2)))

    def test_filter_keys(self):
        test = by_eq(Dict, "filter_keys")

        a, b, c = enumerate("abc")

        test(dict([]), [], less_than(1))
        test(dict([a]), [a], less_than(1))
        test(dict([]), [b], less_than(1))
        test(dict([a]), [a, b], less_than(1))
        test(dict([a]), [b, a], less_than(1))
        test(dict([a]), [a, b, c], less_than(1))
        test(dict([]), [], less_than(2))
        test(dict([a]), [a], less_than(2))
        test(dict([b]), [b], less_than(2))
        test(dict([a, b]), [a, b], less_than(2))
        test(dict([b, a]), [b, a], less_than(2))
        test(dict([a, b]), [a, b, c], less_than(2))

    def test_filter_values(self):
        test = by_eq(Dict, "filter_values")

        a, b, c = enumerate("abc")

        test(dict([]), [], less_than("b"))
        test(dict([a]), [a], less_than("b"))
        test(dict([]), [b], less_than("b"))
        test(dict([a]), [a, b], less_than("b"))
        test(dict([a]), [b, a], less_than("b"))
        test(dict([a]), [a, b, c], less_than("b"))
        test(dict([]), [], less_than("c"))
        test(dict([a]), [a], less_than("c"))
        test(dict([b]), [b], less_than("c"))
        test(dict([a, b]), [a, b], less_than("c"))
        test(dict([b, a]), [b, a], less_than("c"))
        test(dict([a, b]), [a, b, c], less_than("c"))

    def test_first(self):
        test = by_eq(Dict, "first")

        a, b, c = enumerate("abc")

        with should_raise(StopIteration):
            test(None, [])
        test(a, [a])
        test(b, [b])
        test(a, [a, b])
        test(b, [b, a])
        test(a, [a, b, c])
        with should_raise(StopIteration):
            test(None, [], predicate=t(key_less_than(1)))
        test(a, [a], predicate=t(key_less_than(1)))
        with should_raise(StopIteration):
            test(None, [b], predicate=t(key_less_than(1)))
        test(a, [a, b], predicate=t(key_less_than(1)))
        test(a, [b, a], predicate=t(key_less_than(1)))
        test(a, [a, b, c], predicate=t(key_less_than(1)))
        test(None, [], default=None)
        test(a, [a], default=None)
        test(b, [b], default=None)
        test(a, [a, b], default=None)
        test(b, [b, a], default=None)
        test(a, [a, b, c], default=None)
        test(None, [], predicate=t(key_less_than(1)), default=None)
        test(a, [a], predicate=t(key_less_than(1)), default=None)
        test(None, [b], predicate=t(key_less_than(1)), default=None)
        test(a, [a, b], predicate=t(key_less_than(1)), default=None)
        test(a, [b, a], predicate=t(key_less_than(1)), default=None)
        test(a, [a, b, c], predicate=t(key_less_than(1)), default=None)

    def test_flat_map(self):
        test = by_eq(Dict, "flat_map")

        a, b, c = enumerate("abc")

        def key_range(key, value):
            return [(k, value) for k in range(key)]

        test(dict([]), [], t(key_range))
        test(dict([]), [a], t(key_range))
        test(dict([(0, "b")]), [b], t(key_range))
        test(dict([(0, "b")]), [a, b], t(key_range))
        test(dict([(0, "b")]), [b, a], t(key_range))
        test(dict([(0, "c"), (1, "c")]), [a, b, c], t(key_range))

    def test_flat_map_keys(self):
        test = by_eq(Dict, "flat_map_keys")

        a, b, c = enumerate("abc")

        test(dict([]), [], range)
        test(dict([]), [a], range)
        test(dict([(0, "b")]), [b], range)
        test(dict([(0, "b")]), [a, b], range)
        test(dict([(0, "b")]), [b, a], range)
        test(dict([(0, "c"), (1, "c")]), [a, b, c], range)

    def test_flat_map_values(self):
        test = by_eq(Dict, "flat_map_values")

        a, b, c = enumerate("abc")

        def letter_tower(c):
            n = ord(c) - ord("a")
            return c * n

        test(dict([]), [], letter_tower)
        test(dict([]), [a], letter_tower)
        test(dict([b]), [b], letter_tower)
        test(dict([b]), [a, b], letter_tower)
        test(dict([b]), [b, a], letter_tower)
        test(dict([b, c]), [a, b, c], letter_tower)

    def test_fold(self):
        test = by_eq(Dict, "fold")

        a, b, c = enumerate("abc")

        def add_keys(acc, item):
            return acc + item[0]

        def mul_keys(acc, item):
            return acc * item[0]

        test(0, [], 0, add_keys)
        test(0, [a], 0, add_keys)
        test(1, [b], 0, add_keys)
        test(1, [a, b], 0, add_keys)
        test(1, [b, a], 0, add_keys)
        test(3, [a, b, c], 0, add_keys)
        test(1, [], 1, add_keys)
        test(1, [a], 1, add_keys)
        test(2, [b], 1, add_keys)
        test(2, [a, b], 1, add_keys)
        test(2, [b, a], 1, add_keys)
        test(4, [a, b, c], 1, add_keys)
        test(1, [], 1, mul_keys)
        test(0, [a], 1, mul_keys)
        test(1, [b], 1, mul_keys)
        test(0, [a, b], 1, mul_keys)
        test(0, [b, a], 1, mul_keys)
        test(0, [a, b, c], 1, mul_keys)

    def test_fold_while(self):
        test = by_eq(Dict, "fold_while")

        def add_values(acc, item):
            k, v = item
            return acc + v

        def mul_values(acc, item):
            k, v = item
            return acc * v

        test(0, [], 0, add_values, less_than(3))
        test(1, [], 1, add_values, less_than(3))
        test(1, [("a", 1), ("b", 2), ("c", 3), ("d", 4)], 0, add_values, less_than(3))
        test(3, [("a", 1), ("b", 2), ("c", 3), ("d", 4)], 0, add_values, less_than(4))
        test(2, [("a", 1), ("b", 2), ("c", 3), ("d", 4)], 1, add_values, less_than(3))
        test(2, [("a", 1), ("b", 2), ("c", 3), ("d", 4)], 1, add_values, less_than(4))
        test(6, [("a", 1), ("b", 2), ("c", 3), ("d", 4)], 0, add_values, less_than(10))
        test(7, [("a", 1), ("b", 2), ("c", 3), ("d", 4)], 1, add_values, less_than(10))
        test(24, [("a", 1), ("b", 2), ("c", 3), ("d", 4)], 1, mul_values, less_than(30))
        test(0, [("a", 1), ("b", 2), ("c", 3), ("d", 4)], 0, mul_values, less_than(30))

    def test_for_each(self):
        def test(items):
            actual = []
            Dict(items).for_each(actual.append)
            assert iter_eq(actual, items)

        a, b, c = enumerate("abc")

        test([])
        test([a])
        test([b])
        test([a, b])
        test([b, a])
        test([a, b, c])

    def test_group_by(self):
        test = by_eq(Dict, "group_by")

        def key_len(item):
            k, v = item
            return len(k)

        a = ("a", 0)
        ab = ("ab", 1)
        abc = ("abc", 2)
        bcd = ("bcd", 3)

        test({}, [])
        test({a: [a], ab: [ab], abc: [abc], bcd: [bcd]}, [a, ab, abc, bcd])
        test({}, [], key_len)
        test({1: [a], 2: [ab], 3: [abc, bcd]}, [a, ab, abc, bcd], key_len)

    def test_intersection(self):
        test = by_eq(Dict, "intersection")

        a = {"a": 1}
        b = {"b": 2}
        c = {"c": 3}
        ab = {"a": 1, "b": 2}
        abc = {"a": 1, "b": 2, "c": 3}

        for l, r in Iter((a, b, c)).combinations(2):
            test({}, l, r)
            test({}, r, l)
        test({}, a, b, c)
        test(a, a, a)
        test(a, a, ab)
        test(a, ab, a)
        test(a, a, abc)
        test(a, abc, a)
        test(b, b, ab)
        test(b, ab, b)
        test(ab, ab, abc)
        test(ab, abc, ab)

    def test_intersperse(self):
        test = by_iter_eq(Dict, "intersperse")

        a, b, c = enumerate("abc")
        sep = ("sep", "sep")

        test([], [], sep)
        test([a], [a], sep)
        test([b], [b], sep)
        test([a, sep, b], [a, b], sep)
        test([b, sep, a], [b, a], sep)
        test([a, sep, b, sep, c], [a, b, c], sep)
        test([b, sep, a, sep, c], [b, a, c], sep)
        test([c, sep, b, sep, a], [c, b, a], sep)
        test([a, (0, 1), b], [a, b], (0, 1))
        test([a, None, b, None, c], [a, b, c], None)

    def test_invert(self):
        test = by_eq(Dict, "invert")

        a, b, c = enumerate("abc")

        test({}, [])
        test({"a": 0}, [a])
        test({"b": 1}, [b])
        test({"a": 0, "b": 1}, [a, b])
        test({"a": 0, "b": 1}, [b, a])
        test({"a": 0, "b": 1, "c": 2}, [a, b, c])

    def test_items(self):
        test = by_iter_eq(Dict, "items")

        a, b, c = enumerate("abc")

        test([], [])
        test([a], [a])
        test([b], [b])
        test([a, b], [a, b])
        test([b, a], [b, a])
        test([a, b, c], [a, b, c])

    def test_keys(self):
        def test(expected_keys, items):
            d = Dict(items)
            keys = d.keys()
            assert isinstance(keys, DictKeys)
            assert keys == expected_keys

        a, b, c = enumerate("abc")

        test(set(), [])
        test({"a"}, [("a", 1)])
        test({"b"}, [("b", 2)])
        test({0}, [a])
        test({1}, [b])
        test({0, 1}, [a, b])
        test({1, 0}, [b, a])
        test({0, 1, 2}, [a, b, c])

    def test_last(self):
        test = by_eq(Dict, "last")

        a, b, c = enumerate("abc")

        with should_raise(StopIteration):
            test(None, [])
        test(a, [a])
        test(b, [b])
        test(b, [a, b])
        test(a, [b, a])
        test(c, [a, b, c])
        with should_raise(StopIteration):
            test(None, [], predicate=t(key_is_odd))
        with should_raise(StopIteration):
            test(None, [a], predicate=t(key_is_odd))
        test(b, [b], predicate=t(key_is_odd))
        test(b, [a, b], predicate=t(key_is_odd))
        test(b, [b, a], predicate=t(key_is_odd))
        test(b, [a, b, c], predicate=t(key_is_odd))
        test(None, [], default=None)
        test(a, [a], default=None)
        test(b, [b], default=None)
        test(b, [a, b], default=None)
        test(a, [b, a], default=None)
        test(c, [a, b, c], default=None)
        test(None, [], predicate=t(key_is_odd), default=None)
        test(None, [a], predicate=t(key_is_odd), default=None)
        test(b, [b], predicate=t(key_is_odd), default=None)
        test(b, [a, b], predicate=t(key_is_odd), default=None)
        test(b, [b, a], predicate=t(key_is_odd), default=None)
        test(b, [a, b, c], predicate=t(key_is_odd), default=None)

    def test_list(self):
        def test(items):
            actual = Dict(items).list()
            assert isinstance(actual, List)
            assert actual == list(items)

        a, b, c = enumerate("abc")

        test([])
        test([a])
        test([b])
        test([a, b])
        test([b, a])
        test([a, b, c])

    def test_map(self):
        test = by_eq(Dict, "map")

        a, b, c = enumerate("abc")

        test(dict([]), [], t(double_key))
        test(dict([a]), [a], t(double_key))
        test(dict([(2, "b")]), [b], t(double_key))
        test(dict([a, (2, "b")]), [a, b], t(double_key))
        test(dict([(2, "b"), a]), [b, a], t(double_key))
        test(dict([a, (2, "b"), (4, "c")]), [a, b, c], t(double_key))
        test(dict([]), [], t(double_value))
        test(dict([(0, "aa")]), [a], t(double_value))
        test(dict([(1, "bb")]), [b], t(double_value))
        test(dict([(0, "aa"), (1, "bb")]), [a, b], t(double_value))
        test(dict([(1, "bb"), (0, "aa")]), [b, a], t(double_value))
        test(dict([(0, "aa"), (1, "bb"), (2, "cc")]), [a, b, c], t(double_value))

    def test_map_keys(self):
        test = by_iter_eq(Dict, "map_keys")

        a, b, c = enumerate("abc")

        test([], [], double)
        test(dict([a]), [a], double)
        test(dict([(2, "b")]), [b], double)
        test(dict([a, (2, "b")]), [a, b], double)
        test(dict([(2, "b"), a]), [b, a], double)
        test(dict([a, (2, "b"), (4, "c")]), [a, b, c], double)

    def test_map_values(self):
        test = by_iter_eq(Dict, "map_values")

        a, b, c = enumerate("abc")

        test({}, [], double)
        test({0: "aa"}, [a], double)
        test({1: "bb"}, [b], double)
        test({0: "aa", 1: "bb"}, [a, b], double)
        test({1: "bb", 0: "aa"}, [b, a], double)
        test({0: "aa", 1: "bb", 2: "cc"}, [a, b, c], double)

    def test_max(self):
        test = by_eq(Dict, "max")

        a, b, c = enumerate("abc")

        with should_raise(ValueError):
            test(None, [])
        test(a, [a])
        test(b, [b])
        test(b, [a, b])
        test(b, [b, a])
        test(c, [a, b, c])
        with should_raise(ValueError):
            test([], [], key=t(lambda k, v: -k))
        test(a, [a], key=t(lambda k, v: -k))
        test(b, [b], key=t(lambda k, v: -k))
        test(a, [a, b], key=t(lambda k, v: -k))
        test(a, [b, a], key=t(lambda k, v: -k))
        test(a, [a, b, c], key=t(lambda k, v: -k))
        test(None, [], default=None)
        test(a, [a], default=None)
        test(b, [b], default=None)
        test(b, [a, b], default=None)
        test(b, [b, a], default=None)
        test(c, [a, b, c], default=None)
        test(None, [], key=t(lambda k, v: -k), default=None)
        test(a, [a], key=t(lambda k, v: -k), default=None)
        test(b, [b], key=t(lambda k, v: -k), default=None)
        test(a, [a, b], key=t(lambda k, v: -k), default=None)
        test(a, [b, a], key=t(lambda k, v: -k), default=None)
        test(a, [a, b, c], key=t(lambda k, v: -k), default=None)

    def test_min(self):
        test = by_eq(Dict, "min")

        a, b, c = enumerate("abc")

        with should_raise(ValueError):
            test(None, [])
        test(a, [a])
        test(b, [b])
        test(a, [a, b])
        test(a, [b, a])
        test(a, [a, b, c])
        with should_raise(ValueError):
            test([], [], key=t(lambda k, v: -k))
        test(a, [a], key=t(lambda k, v: -k))
        test(b, [b], key=t(lambda k, v: -k))
        test(b, [a, b], key=t(lambda k, v: -k))
        test(b, [b, a], key=t(lambda k, v: -k))
        test(c, [a, b, c], key=t(lambda k, v: -k))
        test(None, [], default=None)
        test(a, [a], default=None)
        test(b, [b], default=None)
        test(a, [a, b], default=None)
        test(a, [b, a], default=None)
        test(a, [a, b, c], default=None)
        test(None, [], key=t(lambda k, v: -k), default=None)
        test(a, [a], key=t(lambda k, v: -k), default=None)
        test(b, [b], key=t(lambda k, v: -k), default=None)
        test(b, [a, b], key=t(lambda k, v: -k), default=None)
        test(b, [b, a], key=t(lambda k, v: -k), default=None)
        test(c, [a, b, c], key=t(lambda k, v: -k), default=None)

    def test_min_max(self):
        test = by_eq(Dict, "min_max")

        a, b, c = enumerate("abc")

        with should_raise(ValueError):
            test(None, [])
        test((a, a), [a])
        test((b, b), [b])
        test((a, b), [a, b])
        test((a, b), [b, a])
        test((a, c), [a, b, c])
        with should_raise(ValueError):
            test([], [], key=t(lambda k, v: -k))
        test((a, a), [a], key=t(lambda k, v: -k))
        test((b, b), [b], key=t(lambda k, v: -k))
        test((b, a), [a, b], key=t(lambda k, v: -k))
        test((b, a), [b, a], key=t(lambda k, v: -k))
        test((c, a), [a, b, c], key=t(lambda k, v: -k))
        test(None, [], default=None)
        test((a, a), [a], default=None)
        test((b, b), [b], default=None)
        test((a, b), [a, b], default=None)
        test((a, b), [b, a], default=None)
        test((a, c), [a, b, c], default=None)
        test(None, [], key=t(lambda k, v: -k), default=None)
        test((a, a), [a], key=t(lambda k, v: -k), default=None)
        test((b, b), [b], key=t(lambda k, v: -k), default=None)
        test((b, a), [a, b], key=t(lambda k, v: -k), default=None)
        test((b, a), [b, a], key=t(lambda k, v: -k), default=None)
        test((c, a), [a, b, c], key=t(lambda k, v: -k), default=None)

    def test_only(self):
        test = by_eq(Dict, "only")

        a, b, c = enumerate("abc")

        with should_raise(ValueError, "no item found"):
            test(None, [])
        test(a, [a])
        test(b, [b])
        with should_raise(ValueError, "too many items found"):
            test(None, [a, b])
        with should_raise(ValueError, "too many items found"):
            test(None, [b, a])
        with should_raise(ValueError, "too many items found"):
            test(None, [a, b, c])
        with should_raise(ValueError, "no item found"):
            test(None, [], predicate=t(key_is_odd))
        with should_raise(ValueError, "no item found"):
            test(None, [a], predicate=t(key_is_odd))
        test(b, [b], predicate=t(key_is_odd))
        test(b, [a, b], predicate=t(key_is_odd))
        test(b, [b, a], predicate=t(key_is_odd))
        test(b, [a, b, c], predicate=t(key_is_odd))

    def test_partition(self):
        def test(expected, items, *a, **kw):
            etrue, efalse = expected
            atrue, afalse = Dict(items).partition(*a, **kw)
            assert iter_eq(atrue, etrue)
            assert iter_eq(afalse, efalse)

        test(({}, {}), [], itemgetter(0))
        test(({}, {}), [], itemgetter(1))
        test(({True: 0}, {False: 0}), {True: 0, False: 0}, itemgetter(0))
        test(({}, {True: 0, False: 0}), {True: 0, False: 0}, itemgetter(1))
        test(({1: "a"}, {2: "b"}), {1: "a", 2: "b"}, t(key_is_odd))

    def test_permutations(self):
        test = by_iter_eq(Dict, "permutations")

        a = ("a", 1)
        b = ("b", 2)

        for r in range(1, 3):
            test([], [], r)

        test([(a,)], [a], 1)
        test([], [a], 2)
        test([(a, b), (b, a)], [a, b])
        test([(a,), (b,)], [a, b], 1)
        test([(a, b), (b, a)], [a, b], 2)
        test([(b,), (a,)], [b, a], 1)
        test([(b, a), (a, b)], [b, a], 2)

    def test_powerset(self):
        test = by_iter_eq(Dict, "powerset")

        a, b, c = enumerate("abc")

        test([{}], [])
        test([{}, dict([a])], [a])
        test([{}, dict([a]), dict([b]), dict([a, b])], [a, b])
        test([{}, dict([a]), dict([b]), dict([c]), dict([a, b]), dict([a, c]), dict([b, c]), dict([a, b, c])],
             [a, b, c])

    def test_product(self):
        test = by_iter_eq(Dict, "product")

        a, b, c = enumerate("abc")
        x, y = (("x", 3), ("y", 4))

        test([], [], [])
        test([], [], [x, y])
        test([], [a], [])
        test([], [a, b], [])
        test([(a, x), (a, y)], [a], [x, y])
        test([(a, x)], [a], [x])
        test([(a, x), (a, y), (b, x), (b, y)], [a, b], [x, y])
        test([(a, x), (b, x)], [a, b], [x])
        test([(a, a)], [a], repeat=2)
        test([(a, a), (a, b), (b, a), (b, b)], [a, b], repeat=2)
        test([(a, a, a), (a, a, b), (a, b, a), (a, b, b), (b, a, a), (b, a, b), (b, b, a), (b, b, b)], [a, b], repeat=3)
        test(
            [(a, x, 1), (a, x, 2), (a, y, 1), (a, y, 2), (b, x, 1), (b, x, 2), (b, y, 1), (b, y, 2)],
            [a, b],
            [x, y],
            [1, 2],
        )

    def test_put(self):
        test = by_eq(Dict, "put")

        test({"a": 1}, [], "a", 1)
        test({"a": 1}, [("a", 1)], "a", 1)
        test({"a": 1}, [("a", 0)], "a", 1)
        test({"a": 1, "b": 2}, [("b", 2)], "a", 1)
        test({"a": 1, "b": 2}, [("a", 1), ("b", 2)], "a", 1)
        test({"a": 1, "b": 2}, [("a", 0), ("b", 2)], "a", 1)

    def test_repeat(self):
        test = by_iter_eq(Dict, "repeat")

        a, b, c = enumerate("abc")

        test([], [], 0)
        test([], [], 1)
        test([], [], 2)
        test([a], [a], 1)
        test([a, a], [a], 2)
        test([a, a, a], [a], 3)
        test([a, b, a, b], [a, b], 2)
        test([a, b, a, b, a, b], [a, b], 3)

    def test_set(self):
        test = by_eq(Dict, "set")

        a, b, c = enumerate("abc")

        test(set(), [])
        test({a}, [a])
        test({b}, [b])
        test({a, b}, [a, b])
        test({a, b}, [b, a])
        test({a, b, c}, [a, b, c])

    def test_size(self):
        test = by_eq(Dict, "size")

        a, b, c = enumerate("abc")

        test(0, [])
        test(1, [a])
        test(2, [a, b])
        test(3, [a, b, c])

    def test_sliding(self):
        test = by_iter_eq(Dict, "sliding")

        a, b, c, d = enumerate("abcd")

        test([], [], 1)
        test([], [], 2)
        test([], [a], 2)
        test([dict([a])], [a], 1)
        test([dict([a, b])], [a, b], 2)
        test([dict([a, b, c])], [a, b, c], 3)
        test([dict([a, b]), dict([b, c])], [a, b, c], 2, 1)
        test([dict([a, b]), dict([b, c]), dict([c, d])], [a, b, c, d], 2, 1)
        test([dict([a, b, c]), dict([b, c, d])], [a, b, c, d], 3, 1)
        test([dict([a, b]), dict([c, d])], [a, b, c, d], 2, 2)
        test([dict([a, b, c])], [a, b, c, d], 3, 2)
        test([dict([a, b, c])], [a, b, c], 3, 1)
        test([dict([a, b])], [a, b, c], 2, 3)

    def test_sliding_by_timestamp(self):
        def clock(timestamps):
            iterator = iter(timestamps)

            def get_next():
                return next(iterator)

            return get_next

        test = by_iter_eq(Dict, "sliding_by_timestamp")

        a, b, c, d = enumerate("abcd")

        test([], [], 1, 1, timestamp(clock([])))
        test([dict([a])], [a], 1, 1, timestamp(clock([0])))
        test([dict([a, b]), dict([b, c]), dict([c, d])], [a, b, c, d], 2, 1, timestamp(clock([0, 1, 2, 3])))
        test([dict([a, b]), dict([c, d])], [a, b, c, d], 2, 2, timestamp(clock([0, 1, 2, 3])))
        test([dict([a]), dict([b])], [a, b], 1, 1, timestamp(clock([0, 1])))
        test([dict([a, b]), dict([c])], [a, b, c], 1, 1, timestamp(clock([0, 0, 1])))

    def test_take(self):
        test = by_eq(Dict, "take")

        a, b, c = enumerate("abc")

        test(dict(), [], 0)
        test(dict(), [], 1)
        test(dict(), [a], 0)
        test(dict([a]), [a], 1)
        test(dict([a]), [a], 2)
        test(dict([a]), [a, b], 1)
        test(dict([a, b]), [a, b], 2)
        test(dict([a, b]), [a, b, c], 2)

    def test_take_while(self):
        test = by_eq(Dict, "take_while")

        a, b, c = enumerate("abc")

        test(dict(), [], lambda x: True)
        test(dict(), [], lambda x: False)
        test(dict(), [a], lambda x: False)
        test(dict([a]), [a], lambda x: True)
        test(dict([a]), [a, b], t(lambda k, v: k == 0))
        test(dict([a, b]), [a, b, c], t(lambda k, v: k < 2))
        test(dict(), [a, b, c], lambda x: False)

    def test_timestamp(self):
        test = by_iter_eq(Dict, "timestamp")

        def clock():
            nonlocal now
            now += 1
            return now

        a, b, c = enumerate("abc")

        now = 0
        test([], [], clock)

        now = 0
        test([(1, a)], [a], clock)

        now = 0
        test([(1, a), (2, b)], [a, b], clock)

        now = 0
        test([(1, a), (2, b), (3, c)], [a, b, c], clock)

    def test_tqdm(self):
        # tqdm shouldn't change the input, and shouldn't raise any exceptions
        def test(items):
            actual = Dict(items).tqdm()
            assert iter_eq(actual, items)

        test([])
        test([("a", 1)])
        test([("a", 1), ("b", 2)])
        test([("a", 1), ("b", 2), ("c", 3)])

    def test_tuple(self):
        test = by_eq(Dict, "tuple")

        a, b, c = enumerate("abc")

        test((), [])
        test((a,), [a])
        test((b,), [b])
        test((a, b), [a, b])
        test((b, a), [b, a])
        test((a, b, c), [a, b, c])

    def test_union(self):
        test = by_eq(Dict, "union")

        test({}, [], {})
        test({}, [], {}, {})
        test({"a": 1}, [("a", 1)], {})
        test({"a": 1}, [], {"a": 1})
        test({"a": 1, "b": 2}, [("a", 1)], {"b": 2})
        test({"a": 1, "b": 2}, [("a", 1), ("b", 2)], {})
        test({"a": 1, "b": 2}, [], {"a": 1, "b": 2})
        test({"a": 1, "b": 2, "c": 3}, [("a", 1), ("b", 2)], {"c": 3})

    def test_update(self):
        test = by_eq(Dict, "update")

        test({"a": 1}, [], {"a": 1})
        test({"a": 1, "b": 2}, [], {"a": 1, "b": 2})
        test({"a": 1}, [("a", 1)], {"a": 1})
        test({"a": 1, "b": 2}, [("a", 1)], {"b": 2})
        test({"a": 1, "b": 2}, [("a", 1)], [("b", 2)])
        test({"a": 1, "b": 2, "c": 3}, [("a", 1), ("b", 2)], {"c": 3})

    def test_values(self):
        def test(expected_values, items):
            d = Dict(items)
            values = d.values()
            assert isinstance(values, DictValues)
            assert iter_eq(values, expected_values)

        a, b, c = enumerate("abc")
        test([], [])
        test([1], [("a", 1)])
        test([2], [("b", 2)])
        test(["a"], [a])
        test(["b"], [b])
        test(["a", "b"], [a, b])
        test(["b", "a"], [b, a])

    def test_zip(self):
        test = by_iter_eq(Dict, "zip")

        a, b, c = enumerate("abc")
        x, y = (("x", 3), ("y", 4))

        test([], [], [])
        test([], [], [x, y])
        test([], [a], [])
        test([(a, x), (b, y)], [a, b], [x, y])
        test([(a, x)], [a, b], [x])
        test([(a, x)], [a], [x, y])
        test([(a, x, 1), (b, y, 2)], [a, b], [x, y], [1, 2])

    def test_zip_longest(self):
        test = by_iter_eq(Dict, "zip_longest")

        a, b, c = enumerate("abc")
        x, y = (("x", 3), ("y", 4))

        test([], [], [])
        test([(None, x), (None, y)], [], [x, y])
        test([(a, None)], [a], [])
        test([(a, x), (b, y)], [a, b], [x, y])
        test([(a, x), (b, None)], [a, b], [x])
        test([(a, x), (None, y)], [a], [x, y])
        test([(a, x, 1), (b, y, None), (None, None, 3)], [a, b], [x, y], [1, None, 3])


class TestIter:
    def test___add__(self):
        test = by_iter_eq(Iter, "__add__")

        assert iter_eq(Iter([1, 2]) + [3, 4], [1, 2, 3, 4])
        test([], [], [])
        test([1], [], [1])
        test([1], [1], [])
        test([1, 2], [1], [2])
        test([2, 1], [2], [1])

    def test___contains__(self):
        test = by_eq(Iter, "__contains__")

        assert 1 in Iter([1])
        test(False, [], 0)
        test(False, [], None)
        test(True, [0], 0)
        test(False, [0], None)
        test(True, [0], False)

    def test___init__(self):
        def test(items):
            it = Iter(items)
            assert iter_eq(items, it)

        it = Iter()
        assert not it.has_next()

        test([])
        test([1, 2, 3])
        test(range(3))
        test("abc")
        test((1, 2, 3))
        test({1, 2, 3})

    def test___iter__(self):
        test = by_iter_eq(Iter, "__iter__")

        test(range(0), range(0))
        test(range(1), range(1))
        test(range(2), range(2))

        it = Iter(range(1))
        assert it.peek() == 0
        assert iter_eq(iter(it), range(1))
        it = Iter(range(2))
        assert it.peek() == 0
        assert iter_eq(iter(it), range(2))
        it = Iter(range(2))
        assert it.peek() == 0
        assert it.peek(2) == 1
        assert iter_eq(iter(it), range(2))
        it = Iter(range(3))
        assert it.peek() == 0
        assert it.peek(2) == 1
        assert iter_eq(iter(it), range(3))

        items = []
        for i in Iter(range(3)):
            items.append(i)
        assert iter_eq(items, range(3))

    def test___mul__(self):
        test = by_iter_eq(Iter, "__mul__")

        assert iter_eq(Iter([1, 2]) * 2, [1, 2, 1, 2])
        test([], [], 0)
        test([], [], 1)
        test([], [], 2)
        test([], [1], 0)
        test([1], [1], 1)
        test([1, 1], [1], 2)
        test([], [1, 2], 0)
        test([1, 2], [1, 2], 1)
        test([1, 2, 1, 2], [1, 2], 2)

    def test___next__(self):
        it = Iter()
        with should_raise(StopIteration):
            next(it)

        it = Iter(range(1))
        assert next(it) == 0
        with should_raise(StopIteration):
            next(it)

        it = Iter(range(1))
        assert it.peek() == 0
        assert next(it) == 0
        with should_raise(StopIteration):
            next(it)

        it = Iter(range(2))
        assert it.peek() == 0
        assert it.peek(2) == 1
        assert next(it) == 0
        assert it.peek() == 1
        assert next(it) == 1
        with should_raise(StopIteration):
            next(it)

    def test___radd__(self):
        test = by_iter_eq(Iter, "__radd__")

        assert iter_eq([1, 2] + Iter([3, 4]), [1, 2, 3, 4])
        test([], [], [])
        test([1], [], [1])
        test([1], [1], [])
        test([2, 1], [1], [2])
        test([1, 2], [2], [1])

    def test___rmul__(self):
        test = by_iter_eq(Iter, "__rmul__")

        assert iter_eq(2 * Iter([1, 2]), [1, 2, 1, 2])
        test([], [], 0)
        test([], [], 1)
        test([], [], 2)
        test([], [1], 0)
        test([1], [1], 1)
        test([1, 1], [1], 2)
        test([], [1, 2], 0)
        test([1, 2], [1, 2], 1)
        test([1, 2, 1, 2], [1, 2], 2)

    def test_accumulate(self):
        test = by_iter_eq(Iter, "accumulate")

        test([1, 3, 6, 10], [1, 2, 3, 4])
        test([], [])
        test([5], [5])
        test([1, 3], [1, 2])
        test([1, 2, 6, 24], [1, 2, 3, 4], mul)
        test([2, 6, 30], [2, 3, 5], mul)
        test([10, 11, 13, 16], [1, 2, 3], add, initial=10)
        test([5], [], add, initial=5)
        test([0, 1, 3, 6], [1, 2, 3], add, initial=0)

        def concat(a, b):
            return str(a) + str(b)

        test(["a", "ab", "abc"], ["a", "b", "c"], concat)

    def test_all(self):
        test = by_eq(Iter, "all")

        test(True, [])
        test(False, [0])
        test(False, [0, 1])
        test(True, [0], is_even)
        test(True, [0], key=is_even)
        test(False, [0, 1], is_even)
        test(False, [0, 1], is_odd)

    def test_any(self):
        test = by_eq(Iter, "any")

        test(False, [])
        test(False, [0])
        test(True, [0, 1])
        test(True, [0], is_even)
        test(True, [0], key=is_even)
        test(True, [0, 1], is_even)
        test(True, [0, 1], is_odd)

    def test_apply(self):
        test = by_eq(Iter, "apply")

        test(True, [], bool)
        test(True, [1], bool)
        test(True, [], all)
        test(True, [1], all)
        test(True, [1, 2], all)
        test(True, [], lambda it: all(is_odd(i) for i in it))
        test(True, [1], lambda it: all(is_odd(i) for i in it))
        test(False, [1, 2], lambda it: all(is_odd(i) for i in it))

    def test_apply_and_wrap(self):
        test = by_iter_eq(Iter, "apply_and_wrap")

        test([], [], lambda it: it.flat_map(tower))
        test([1], [1], lambda it: it.flat_map(tower))
        test([1, 2, 2], [1, 2], lambda it: it.flat_map(tower))

    def test_batch(self):
        test = by_iter_eq(Iter, "batch")

        with should_raise(RequirementException):
            Iter().batch(0)
        test([], [], 1)
        for i in range(1, 5):
            test([(1,)], [1], i)
        test([(1,), (2,)], [1, 2], 1)
        for i in range(2, 5):
            test([(1, 2)], [1, 2], i)
        test([(1, 2), (3,)], [1, 2, 3], 2)

    def test_chain(self):
        test = by_iter_eq(Iter, "chain")

        test([], [], [])
        test([1], [], [1])
        test([1], [1], [])
        test([1, 2], [1], [2])
        test([2, 1], [2], [1])

    def test_combinations(self):
        test = by_iter_eq(Iter, "combinations")

        a, b, c = range(3)

        test([()], [], 0)
        test([()], [a], 0)
        test([()], [a, b], 0)
        test([()], [b, a], 0)
        test([()], [a, b, c], 0)
        test([], [], 1)
        test([(a,)], [a], 1)
        test([(a,), (b,)], [a, b], 1)
        test([(b,), (a,)], [b, a], 1)
        test([(a,), (b,), (c,)], [a, b, c], 1)
        test([], [], 2)
        test([], [a], 2)
        test([(a, b)], [a, b], 2)
        test([(b, a)], [b, a], 2)
        test([(a, b), (a, c), (b, c)], [a, b, c], 2)
        test([()], [], 0, with_replacement=True)
        test([()], [a], 0, with_replacement=True)
        test([()], [a, b], 0, with_replacement=True)
        test([()], [b, a], 0, with_replacement=True)
        test([()], [a, b, c], 0, with_replacement=True)
        test([], [], 1, with_replacement=True)
        test([(a,)], [a], 1, with_replacement=True)
        test([(a,), (b,)], [a, b], 1, with_replacement=True)
        test([(b,), (a,)], [b, a], 1, with_replacement=True)
        test([(a,), (b,), (c,)], [a, b, c], 1, with_replacement=True)
        test([], [], 2, with_replacement=True)
        test([(a, a)], [a], 2, with_replacement=True)
        test([(a, a), (a, b), (b, b)], [a, b], 2, with_replacement=True)
        test([(b, b), (b, a), (a, a)], [b, a], 2, with_replacement=True)
        test([(a, a), (a, b), (a, c), (b, b), (b, c), (c, c)], [a, b, c], 2, with_replacement=True)

    def test_combine_if(self):
        test = by_iter_eq(Iter, "combine_if")

        test([], [], True, "map", double)
        test([], [], False, "map", double)
        test([2], [1], True, "map", double)
        test([1], [1], False, "map", double)
        test([2, 4], [1, 2], True, "map", double)
        test([1, 2], [1, 2], False, "map", double)

    def test_count(self):
        test = by_eq(Iter, "count")

        for items in Iter(["a", "b", "b"]).permutations(3):
            test({"a": 1, "b": 2}, items)
        for items in Iter(["ab", "b", "b"]).permutations(3):
            test({1: 2, 2: 1}, items, len)

    def test_cycle(self):
        def test(expected, items):
            actual = Iter(items).cycle().take(5).tuple()
            assert iter_eq(actual, expected), (actual, expected)

        test([], [])
        test([1, 1, 1, 1, 1], [1])
        test([1, 2, 1, 2, 1], [1, 2])
        test([2, 1, 2, 1, 2], [2, 1])
        test([1, 2, 3, 1, 2], [1, 2, 3])

    def test_default_dict(self):
        def test(items, *a, **kw):
            result = Iter(items).default_dict(*a, **kw)
            assert isinstance(result, DefaultDict)
            assert dict(result) == dict(items)

        a, b, c = enumerate("abc")

        test([], int)
        test([a], int)
        test([a, b], int)
        test([], list)
        test([("a", [1, 2])], list)
        test([("a", [1, 2]), ("b", [3, 4])], list)
        test([], str)
        test([("a", "hello")], str)
        test([("a", "hello"), ("b", "world")], str)

        result = Dict({"a": 1}).default_dict(int)
        assert result["a"] == 1
        assert result["b"] == 0

        result = Dict({}).default_dict(list)
        result["new_list"].append(1)
        assert result["new_list"] == [1]

    def test_dict(self):
        test = by_eq(Iter, "dict")

        test({}, [])
        for items in Iter([("a", 1), ("b", 2)]).permutations(2):
            test({"a": 1, "b": 2}, items)

    def test_distinct(self):
        test = by_iter_eq(Iter, "distinct")

        test([], [])
        test([1], [1])
        test([1, 2], [1, 1, 2])
        test([1, 2], [1, 2, 1])
        test([1, 2], [1, 2, 3], is_odd)
        test([1, 2], [1, 2, 3], is_even)

    def test_do(self):
        def test(items):
            actual = []

            def f(item):
                actual.append(item)

            assert iter_eq(Iter(items).do(f), items)
            assert iter_eq(actual, items)

        for i in range(5):
            test(range(i))

    def test_drop(self):
        test = by_iter_eq(Iter, "drop")

        test([], [], 0)
        test([1], [1], 0)
        test([1, 2], [1, 2], 0)
        test([1, 2, 3], [1, 2, 3], 0)
        test([], [], 1)
        test([], [1], 1)
        test([2], [1, 2], 1)
        test([2, 3], [1, 2, 3], 1)
        test([3], [1, 2, 3], 2)
        test([], [], -1)
        test([1], [1], -1)
        test([2], [1, 2], -1)
        test([3], [1, 2, 3], -1)
        test([2, 3], [1, 2, 3], -2)

    def test_drop_while(self):
        test = by_iter_eq(Iter, "drop_while")

        test([], [], less_than(3))
        test([3, 4], [1, 2, 3, 4], less_than(3))
        test([], [1, 2, 3], less_than(5))
        test([1, 2, 3], [1, 2, 3], less_than(0))

    def test_enumerate(self):
        test = by_iter_eq(Iter, "enumerate")

        test([], [])
        test([(0, 1)], [1])
        test([(0, 1), (1, 2)], [1, 2])
        test([(1, 1)], [1], start=1)
        test([(2, 1), (3, 2)], [1, 2], start=2)

    def test_filter(self):
        test = by_iter_eq(Iter, "filter")

        test([], [], is_even)
        test([], [1], is_even)
        test([2], [1, 2], is_even)
        test([2], [1, 2, 3], is_even)

    def test_first(self):
        test = by_eq(Iter, "first")

        test(1, [1, 2, 3])
        test(2, [1, 2, 3], predicate=greater_than(1))
        test(None, [], default=None)
        test(None, [1, 2, 3], predicate=greater_than(5), default=None)
        with should_raise(StopIteration):
            Iter().first()
        with should_raise(StopIteration):
            Iter([1, 2, 3]).first(predicate=greater_than(5))

    def test_flat_map(self):
        test = by_iter_eq(Iter, "flat_map")

        test([1], [[1]], identity)
        test([], [], tower)
        test([1], [1], tower)
        test([1, 2, 2], [1, 2], tower)
        test([1, 2, 2, 3, 3, 3], [1, 2, 3], tower)

    def test_flatten(self):
        test = by_iter_eq(Iter, "flatten")

        test([], [])
        test([1], [[1]])
        test([1, 2], [[1, 2]])
        test([1], [[1], []])
        test([1, 2, 3], [[1, 2], [3]])
        test([1, 2, 3], [[1], [2], [3]])

    def test_fold(self):
        test = by_eq(Iter, "fold")

        test(0, [], 0, add)
        test(1, [], 1, add)
        test(10, [1, 2, 3, 4], 0, add)
        test(24, [1, 2, 3, 4], 1, mul)

    def test_fold_while(self):
        test = by_eq(Iter, "fold_while")

        test(0, [], 0, add, less_than(3))
        test(1, [], 1, add, less_than(3))
        test(1, [1, 2, 3, 4], 0, add, less_than(3))
        test(3, [1, 2, 3, 4], 0, add, less_than(4))
        test(2, [1, 2, 3, 4], 1, add, less_than(3))
        test(2, [1, 2, 3, 4], 1, add, less_than(4))
        test(6, [1, 2, 3, 4], 0, add, less_than(10))
        test(7, [1, 2, 3, 4], 1, add, less_than(10))
        test(24, [1, 2, 3, 4], 1, mul, less_than(30))
        test(0, [1, 2, 3, 4], 0, mul, less_than(30))

    def test_for_each(self):
        def test(items):
            actual = []
            Iter(items).for_each(actual.append)
            assert iter_eq(actual, items)

        for i in range(5):
            test(range(i))

    def test_group_by(self):
        test = by_eq(Iter, "group_by")

        test({}, [])
        test({}, [], len)
        test({"a": ["a"]}, ["a"])
        test({"a": ["a"], "b": ["b"]}, ["a", "b"])
        test({"a": ["a", "a"]}, ["a", "a"])
        test({"a": ["a", "a"], "b": ["b"]}, ["a", "a", "b"])
        test({1: ["a", "a", "b"]}, ["a", "a", "b"], len)
        test({1: ["a", "b"], 2: ["ab"]}, ["ab", "a", "b"], len)

    def test_has_next(self):
        test = by_eq(Iter, "has_next")

        test(False, [])
        test(True, [1])

        it = Iter([1])
        assert it.peek() == 1
        assert it.has_next()

        test(False, [], 2)
        test(False, [1], 2)
        test(True, [1, 2], 2)

        it = Iter([1])
        assert it.peek() == 1
        assert it.has_next()
        assert not it.has_next(2)

        it = Iter([1, 2])
        assert it.peek() == 1
        assert it.peek(2) == 2
        assert it.has_next(2)
        assert it.next() == 1
        assert it.has_next()
        assert not it.has_next(2)

    def test_intersperse(self):
        test = by_iter_eq(Iter, "intersperse")

        test([], [], 0)
        test([1], [1], 0)
        test([1, 0, 2], [1, 2], 0)
        test([1, 0, 2, 0, 3], [1, 2, 3], 0)

    def test_iter(self):
        def test(items):
            it = Iter(items)
            it2 = it.iter()
            assert it is it2

        test([])
        test([1])
        test([2])
        test([1, 2, 3])

    def test_last(self):
        test = by_eq(Iter, "last")

        with should_raise(StopIteration):
            Iter().last()
        with should_raise(StopIteration):
            Iter().last(predicate=is_odd)
        with should_raise(StopIteration):
            Iter([0, 2]).last(predicate=is_odd)
        test(None, [], default=None)
        test(None, [], predicate=is_odd, default=None)
        test(2, [1, 2])
        test(1, [1, 2], predicate=is_odd)
        test(2, [1, 2], default=None)
        test(1, [1, 2], predicate=is_odd, default=None)

    def test_list(self):
        def test(items):
            actual = Iter(items).list()
            assert isinstance(actual, List)
            assert iter_eq(actual, items)

        for i in range(5):
            test(range(i))

    def test_map(self):
        test = by_iter_eq(Iter, "map")

        test([], [], double)
        test([2], [1], double)
        test([2, 2], [1, 1], double)
        test([2, 4], [1, 2], double)
        test([4, 2], [2, 1], double)

    def test_map_to_keys(self):
        test = by_eq(Iter, "map_to_keys")

        test({}, [], double)
        test({2: 1}, [1], double)
        test({2: 1}, [1, 1], double)
        test({2: 1, 4: 2}, [1, 2], double)
        test({2: 1, 4: 2}, [2, 1], double)

    def test_map_to_pairs(self):
        test = by_iter_eq(Iter, "map_to_pairs")

        test([], [], double)
        test([(1, 2)], [1], double)
        test([(1, 2), (1, 2)], [1, 1], double)
        test([(1, 2), (2, 4)], [1, 2], double)
        test([(2, 4), (1, 2)], [2, 1], double)

    def test_map_to_values(self):
        test = by_eq(Iter, "map_to_values")

        test({}, [], double)
        test({1: 2}, [1], double)
        test({1: 2}, [1, 1], double)
        test({1: 2, 2: 4}, [1, 2], double)
        test({1: 2, 2: 4}, [2, 1], double)

    def test_max(self):
        test = by_eq(Iter, "max")

        with should_raise(ValueError):
            Iter().max()
        with should_raise(ValueError):
            Iter().max(key=len)
        test(None, [], default=None)
        test(1, [1])
        test(1, [1], default=None)
        test("a", ["a"])
        test("b", ["aa", "b"])
        test("aa", ["aa", "b"], len)

    def test_min(self):
        test = by_eq(Iter, "min")

        with should_raise(ValueError):
            Iter().min()
        with should_raise(ValueError):
            Iter().min(key=len)
        test(None, [], default=None)
        test(1, [1])
        test(1, [1], default=None)
        test("a", ["a"])
        test("aa", ["aa", "b"])
        test("b", ["aa", "b"], len)

    def test_min_max(self):
        test = by_eq(Iter, "min_max")

        with should_raise(ValueError):
            Iter().min_max()
        with should_raise(ValueError):
            Iter().min_max(key=len)
        test(None, [], default=None)
        test((1, 1), [1])
        test((1, 1), [1], default=None)
        test(("a", "a"), ["a"])
        test(("aa", "b"), ["aa", "b"])
        test(("b", "aa"), ["aa", "b"], len)
        test(("aa", "c"), ["aa", "bbb", "c"])
        test(("c", "bbb"), ["aa", "bbb", "c"], len)

    def test_next(self):
        it = Iter()
        with should_raise(StopIteration):
            it.next()

        it = Iter(range(1))
        assert it.next() == 0
        with should_raise(StopIteration):
            it.next()

        it = Iter(range(1))
        assert it.peek() == 0
        assert it.next() == 0
        with should_raise(StopIteration):
            it.next()

        it = Iter(range(2))
        assert it.peek() == 0
        assert it.peek(2) == 1
        assert it.next() == 0
        assert it.peek() == 1
        assert it.next() == 1
        with should_raise(StopIteration):
            it.next()

    def test_only(self):
        test = by_eq(Iter, "only")

        with should_raise(ValueError, "no item found"):
            Iter().only()
        with should_raise(ValueError, "no item found"):
            Iter().only(predicate=is_odd)
        with should_raise(ValueError, "no item found"):
            Iter([0]).only(predicate=is_odd)
        with should_raise(ValueError, "too many items found"):
            Iter([1, 1]).only()
        with should_raise(ValueError, "too many items found"):
            Iter([1, 1]).only(predicate=is_odd)
        test(1, [1])
        test(1, [1], predicate=is_odd)
        test(1, [1, 2], predicate=is_odd)
        test(1, [2, 1], predicate=is_odd)

    def test_partition(self):
        def test(expected, items, *a, **kw):
            etrue, efalse = expected
            atrue, afalse = Iter(items).partition(*a, **kw)
            assert iter_eq(atrue, etrue)
            assert iter_eq(afalse, efalse)

        test(([], []), [])
        test(([], []), [], predicate=is_odd)
        test(([True], []), [True])
        test(([], [False]), [False])
        test(([True], [False]), [True, False])
        test(([True], [False]), [False, True])
        test(([1], [2]), [1, 2], is_odd)
        test(([1], [2]), [2, 1], is_odd)

    def test_peek(self):
        with should_raise(ValueError, "peek past end of iterator"):
            Iter().peek()
        it = Iter([1])
        assert it.peek() == 1
        assert it.next() == 1
        with should_raise(ValueError, "peek past end of iterator"):
            it.peek()
        it = Iter([1, 2])
        assert it.peek() == 1
        assert it.next() == 1
        assert it.peek() == 2
        assert it.next() == 2
        with should_raise(ValueError, "peek past end of iterator"):
            it.peek()

        with should_raise(ValueError, "peek past end of iterator"):
            Iter().peek(2)
        it = Iter([1])
        with should_raise(ValueError, "peek past end of iterator"):
            it.peek(2)
        assert it.next() == 1
        with should_raise(ValueError, "peek past end of iterator"):
            it.peek(2)
        it = Iter([1, 2])
        assert it.peek(2) == 2
        assert it.next() == 1
        with should_raise(ValueError, "peek past end of iterator"):
            it.peek(2)
        assert it.next() == 2
        with should_raise(ValueError, "peek past end of iterator"):
            it.peek(2)
        it = Iter([1, 2, 3])
        assert it.peek(2) == 2
        assert it.next() == 1
        assert it.peek(2) == 3
        assert it.next() == 2
        with should_raise(ValueError, "peek past end of iterator"):
            it.peek(2)
        assert it.next() == 3
        with should_raise(ValueError, "peek past end of iterator"):
            it.peek(2)

    def test_permutations(self):
        test = by_iter_eq(Iter, "permutations")

        for r in range(1, 3):
            test([], [], r)
        test([(1,)], [1], 1)
        test([], [1], 2)
        test([(1, 2), (2, 1)], [1, 2])
        test([(1,), (2,)], [1, 2], 1)
        test([(1, 2), (2, 1)], [1, 2], 2)
        test([(2,), (1,)], [2, 1], 1)
        test([(2, 1), (1, 2)], [2, 1], 2)

    def test_powerset(self):
        test = by_iter_eq(Iter, "powerset")

        test([()], [])
        test([(), (1,)], [1])
        test([(), (1,), (2,), (1, 2)], [1, 2])
        test([(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)], [1, 2, 3])

    def test_product(self):
        test = by_iter_eq(Iter, "product")

        test([], [])
        test([], [], [1])
        test([], [], [1, 2])
        test([], [], [1, 2, 3])
        test([], [1], [])
        test([], [1, 2], [])
        test([], [1, 2, 3], [])
        test([(1,)], [1])
        test([(1,), (2,)], [1, 2])
        test([(1,), (2,), (3,)], [1, 2, 3])
        test([(1, 3), (1, 4), (2, 3), (2, 4)], [1, 2], [3, 4])
        # the docs for itertools.product only describe the behavior of different iterables or one iterable with
        # repeat > 1, but it does allow multiple iterables and repeat > 1, and it repeats all iterables. Effectively it
        # takes the product of the iterables and products the result `repeat` times, flattening the result.
        test(
            [
                (1, 3, 1, 3), (1, 3, 1, 4), (1, 3, 2, 3), (1, 3, 2, 4), (1, 4, 1, 3), (1, 4, 1, 4), (1, 4, 2, 3),
                (1, 4, 2, 4), (2, 3, 1, 3), (2, 3, 1, 4), (2, 3, 2, 3), (2, 3, 2, 4), (2, 4, 1, 3), (2, 4, 1, 4),
                (2, 4, 2, 3), (2, 4, 2, 4),
            ],
            [1, 2],
            [3, 4],
            repeat=2,
        )
        test([], [], [1], repeat=2)
        test([], [], [1, 2], repeat=2)
        test([], [], [1, 2, 3], repeat=2)
        test([], [1], [], repeat=2)
        test([], [1, 2], [], repeat=2)
        test([], [1, 2, 3], [], repeat=2)
        test([], [], repeat=2)
        test([(1, 1)], [1], repeat=2)
        test([(1, 1), (1, 2), (2, 1), (2, 2)], [1, 2], repeat=2)
        test([(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)], [1, 2, 3], repeat=2)

    def test_reduce(self):
        test = by_eq(Iter, "reduce")

        with should_raise(ValueError, "reduce on empty iterator"):
            Iter().reduce(add)
        test(1, [1], add)
        test(3, [1, 2], add)
        test(3, [2, 1], add)
        test(6, [1, 2, 3], add)
        test(1, [1], mul)
        test(2, [1, 2], mul)
        test(2, [2, 1], mul)
        test(6, [1, 2, 3], mul)

    def test_repeat(self):
        def test(expected, items, *a, **kw):
            actual = Iter(items).repeat(*a, **kw).take(5)
            assert iter_eq(actual, expected)

        test([], [])
        test([1, 1, 1, 1, 1], [1])
        test([1, 2, 1, 2, 1], [1, 2])
        test([1, 2, 3, 1, 2], [1, 2, 3])
        test([], [], 0)
        test([], [1], 0)
        test([], [1, 2], 0)
        test([], [1, 2, 3], 0)
        test([1], [1], 1)
        test([1, 2], [1, 2], 1)
        test([1, 2, 3], [1, 2, 3], 1)
        test([1, 1], [1], 2)
        test([1, 2, 1, 2], [1, 2], 2)
        test([1, 2, 3, 1, 2], [1, 2, 3], 2)
        test([1, 1, 1], [1], 3)
        test([1, 2, 1, 2, 1], [1, 2], 3)
        test([1, 2, 3, 1, 2], [1, 2, 3], 3)

    def test_set(self):
        it = Iter([1, 2, 3]).set()
        assert isinstance(it, Set)
        assert isinstance(it, set)
        assert it == {1, 2, 3}

    def test_size(self):
        test = by_eq(Iter, "size")

        for i in range(4):
            for items in Iter([1, 2, 3]).permutations(i):
                test(i, items)

    def test_sliding(self):
        test = by_iter_eq(Iter, "sliding")

        with should_raise(ValueError, "size=0"):
            Iter([]).sliding(0, 1)
        with should_raise(ValueError, "step=0"):
            Iter([]).sliding(1, 0)
        with should_raise(ValueError, "size=-1"):
            Iter([]).sliding(-1, 1)
        with should_raise(ValueError, "step=-1"):
            Iter([]).sliding(1, -1)

        for size in range(1, 4):
            for step in range(1, 4):
                test([], [], size, step)

        test([(1,)], [1], 1)
        test([], [1], 2)
        test([(1,), (2,)], [1, 2], 1)
        test([(1, 2)], [1, 2], 2)
        test([(1,), (2,), (3,)], [1, 2, 3], 1)
        test([(1, 2), (2, 3)], [1, 2, 3], 2)
        test([(1,)], [1], 1, 2)
        test([(1,)], [1, 2], 1, 2)
        test([(1,), (3,)], [1, 2, 3], 1, 2)
        test([], [1], 2, 2)
        test([(1, 2)], [1, 2], 2, 2)
        test([(1, 2)], [1, 2, 3], 2, 2)
        test([(1, 2), (3, 4)], [1, 2, 3, 4], 2, 2)

    def test_sliding_by_timestamp(self):
        def clock(timestamps):
            iterator = iter(timestamps)

            def get_next():
                return next(iterator)

            return get_next

        def test(expected_windows, items, size, step, stamp):
            result = Iter(items).sliding_by_timestamp(size, step, stamp)
            actual_windows = [list(window) for window in result]
            assert actual_windows == expected_windows

        test([], [], 1, 1, timestamp(clock([])))
        test([[1]], [1], 1, 1, timestamp(clock([0])))
        test([[1, 2], [2, 3], [3, 4]], [1, 2, 3, 4], 2, 1, timestamp(clock([0, 1, 2, 3])))
        test([[1, 2], [3, 4]], [1, 2, 3, 4], 2, 2, timestamp(clock([0, 1, 2, 3])))
        test([[1, 2], [3]], [1, 2, 3], 1, 1, timestamp(clock([0, 0, 1])))

    def test_take(self):
        test = by_iter_eq(Iter, "take")

        for i in range(5):
            items = range(5)
            expected = items[:i]
            test(expected, items, i)

    def test_take_while(self):
        test = by_iter_eq(Iter, "take_while")

        test([], [])
        test([], [], is_odd)
        test([], [], is_even)
        test([], [0, 1])
        test([1], [1, 0])
        test([1], [1], is_odd)
        test([], [1], is_even)
        test([1], [1, 2], is_odd)
        test([], [1, 2], is_even)

    def test_tee(self):
        it1, it2 = Iter([1, 2, 3]).tee()
        assert list(it1) == [1, 2, 3]
        assert list(it2) == [1, 2, 3]

        it1, it2 = Iter([1, 2, 3]).tee(2)
        assert list(it1) == [1, 2, 3]
        assert list(it2) == [1, 2, 3]

        it1, it2, it3 = Iter([1, 2, 3]).tee(3)
        assert list(it1) == [1, 2, 3]
        assert list(it2) == [1, 2, 3]
        assert list(it3) == [1, 2, 3]

        (it1,) = Iter([1, 2, 3]).tee(1)
        assert list(it1) == [1, 2, 3]

        it1, it2 = Iter([]).tee()
        assert list(it1) == []
        assert list(it2) == []

        it1, it2 = Iter([1, 2, 3]).tee()
        assert next(it1) == 1
        assert list(it2) == [1, 2, 3]  # it2 should still have all items
        assert list(it1) == [2, 3]  # it1 should have remaining items

    def test_timestamp(self):
        test = by_iter_eq(Iter, "timestamp")

        def clock(times):
            i = 0

            def inner():
                nonlocal i
                try:
                    return times[i]
                finally:
                    i += 1

            return inner

        test([], [])
        test([], [], clock([]))
        test([(0, 1)], [1], clock([0]))
        test([(0, 1), (0, 2)], [1, 2], clock([0, 0]))
        test([(0, 1), (0, 2), (1, 3)], [1, 2, 3], clock([0, 0, 1]))

    def test_tqdm(self):
        # tqdm shouldn't change the input, and shouldn't raise any exceptions
        def test(items):
            actual = Iter(items).tqdm()
            assert iter_eq(actual, items)

        for i in range(5):
            test(range(i))

    def test_transpose(self):
        def test(expected, items):
            actual = Iter(items).transpose().list()
            assert iter_eq(actual, expected), (actual, expected)
            assert iter_eq(actual.transpose(), items), (actual.transpose(), items)

        test([], [])
        test([(1, 3), (2, 4)], [(1, 2), (3, 4)])
        test([(1, 3, 5), (2, 4, 6)], [(1, 2), (3, 4), (5, 6)])

    def test_tuple(self):
        it = Iter([1, 2, 3]).tuple()
        assert isinstance(it, Tuple)
        assert isinstance(it, tuple)
        assert iter_eq(it, [1, 2, 3])

    def test_unzip(self):
        pass  # direct alias of transpose, does not need further testing

    def test_zip(self):
        test = by_iter_eq(Iter, "zip")

        test([], [], [])
        test([], [], [1])
        test([], [], [1, 2])
        test([(1, 2)], [1], [2, 3])
        test([(1, 3), (2, 4)], [1, 2], [3, 4])
        test([(1, 4), (2, 5), (3, 6)], [1, 2, 3], [4, 5, 6])
        test([(1, 4, 7), (2, 5, 8), (3, 6, 9)], [1, 2, 3], [4, 5, 6], [7, 8, 9])

    def test_zip_longest(self):
        test = by_iter_eq(Iter, "zip_longest")

        test([], [])
        test([], [], [])
        test([(1,)], [1])
        test([(1, None)], [1], [])
        test([(1, 2)], [1], [2])
        test([(1, 3), (2, None)], [1, 2], [3])
        test([(1, 3), (2, 4)], [1, 2], [3, 4])


class TestList:
    def test___add__(self):
        def test(expected, left, right):
            actual = List(left) + right
            assert iter_eq(actual, expected)
            assert isinstance(actual, List)

        test([1, 2, 3, 4], [1, 2], [3, 4])
        test([], [], [])
        test([1, 2], [1, 2], [])
        test([1, 2], [], [1, 2])
        test([1, 2, 3, 4], [1, 2], List([3, 4]))

    def test___getitem__(self):
        def test_index(expected, items, index):
            result = List(items)[index]
            assert result == expected

        def test_slice(expected, items, slice_obj):
            result = List(items)[slice_obj]
            assert result == expected
            assert isinstance(result, List)

        test_index(1, [1, 2, 3, 4, 5], 0)
        test_index(2, [1, 2, 3, 4, 5], 1)
        test_index(5, [1, 2, 3, 4, 5], -1)
        test_index(4, [1, 2, 3, 4, 5], -2)

        test_slice([2, 3, 4], [1, 2, 3, 4, 5], slice(1, 4))
        test_slice([1, 2, 3], [1, 2, 3, 4, 5], slice(None, 3))
        test_slice([3, 4, 5], [1, 2, 3, 4, 5], slice(2, None))
        test_slice([1, 3, 5], [1, 2, 3, 4, 5], slice(None, None, 2))
        test_slice([], [1, 2, 3, 4, 5], slice(10, 20))

    def test___iter__(self):
        def test(items):
            iterator = iter(List(items))
            assert isinstance(iterator, Iter)
            assert list(iterator) == items

        test([])
        test([1, 2, 3])

        items = []
        for item in List([1, 2, 3]):
            items.append(item)
        assert items == [1, 2, 3]

    def test___mul__(self):
        def test(expected, items, n):
            result = List(items) * n
            assert isinstance(result, List)
            assert list(result) == expected

        test([1, 2, 1, 2, 1, 2], [1, 2], 3)
        test([], [], 5)
        test(["a", "a", "a", "a"], ["a"], 4)
        test([], [1, 2, 3], 0)
        test([], [1, 2], -1)

    def test___rmul__(self):
        def test(expected, n, items):
            result = n * List(items)
            assert isinstance(result, List)
            assert list(result) == expected

        test([1, 2, 1, 2, 1, 2], 3, [1, 2])
        test([], 5, [])
        test(["a", "a", "a", "a"], 4, ["a"])
        test([], 0, [1, 2, 3])
        test([], -1, [1, 2])

    def test_all(self):
        test = by_eq(List, "all")

        test(True, [])
        test(True, [1, 2, 3, True])
        test(False, [0, False, None, ""])
        test(False, [1, 2, 0, 3])
        test(True, [True, True, True])
        test(False, [True, False, True])
        test(True, [1, 3, 5, 7], key=is_odd)
        test(False, [1, 2, 3, 5], key=is_odd)

    def test_any(self):
        test = by_eq(List, "any")

        test(False, [])
        test(True, [1, 2, 3, True])
        test(False, [0, False, None, ""])
        test(True, [0, False, 1, None])
        test(False, [False, False, False])
        test(True, [False, True, False])
        test(False, [2, 4, 6, 8], key=is_odd)
        test(True, [2, 3, 4, 6], key=is_odd)

    def test_append(self):
        test = by_iter_eq(List, "append")

        test([1, 2, 3], [1, 2], 3)
        test([1], [], 1)
        test([1, 2, None], [1, 2], None)
        test(["a", "b", "c"], ["a", "b"], "c")

        ls = List()
        assert ls.append(1) is ls

    def test_apply(self):
        test = by_eq(List, "apply")

        test(False, [], bool)
        test(True, [1], bool)
        test(3, [1, 2, 3], len)
        test(0, [], len)
        test(6, [1, 2, 3], sum)

    def test_apply_and_wrap(self):
        test = by_iter_eq(List, "apply_and_wrap")

        test([], [], identity)
        test([1], [1], identity)
        test([1, 2, 3], [1, 2, 3], identity)
        test([3, 2, 1], [1, 2, 3], reversed)

    def test_batch(self):
        test = by_iter_eq(List, "batch")

        with should_raise(RequirementException):
            List().batch(0)

        test([], [], 1)
        test([(1,)], [1], 1)
        test([(1,), (2,)], [1, 2], 1)
        test([(1, 2)], [1, 2], 2)
        test([(1, 2), (3,)], [1, 2, 3], 2)
        test([(1, 2, 3)], [1, 2, 3], 3)
        test([(1, 2, 3)], [1, 2, 3], 4)

    def test_chain(self):
        test = by_iter_eq(List, "chain")

        test([], [])
        test([], [], [])
        test([], [], [], [])
        test([1], [1], [])
        test([1, 2], [1, 2], [])
        test([1, 2, 3], [1, 2], [3])
        test([1, 2, 3, 4], [1, 2], [3, 4])
        test([1, 2, 3, 4, 5], [1], [2, 3], [4, 5])

    def test_clear(self):
        def test(items):
            lst = List(items)
            result = lst.clear()
            assert result is lst
            assert not lst

        test([])
        test([1])
        test([1, 2, 3])
        test(["a", "b", "c"])

    def test_combinations(self):
        test = by_iter_eq(List, "combinations")

        test([()], [], 0)
        test([()], [1], 0)
        test([()], [1, 2], 0)
        test([()], [1, 2, 3], 0)
        test([], [], 1)
        test([(1,)], [1], 1)
        test([(1,), (2,)], [1, 2], 1)
        test([(1,), (2,), (3,)], [1, 2, 3], 1)
        test([], [], 2)
        test([], [1], 2)
        test([(1, 2)], [1, 2], 2)
        test([(1, 2), (1, 3), (2, 3)], [1, 2, 3], 2)

        test([()], [], 0, with_replacement=True)
        test([()], [1], 0, with_replacement=True)
        test([], [], 1, with_replacement=True)
        test([(1,)], [1], 1, with_replacement=True)
        test([(1,), (2,)], [1, 2], 1, with_replacement=True)
        test([], [], 2, with_replacement=True)
        test([(1, 1)], [1], 2, with_replacement=True)
        test([(1, 1), (1, 2), (2, 2)], [1, 2], 2, with_replacement=True)

    def test_combine_if(self):
        test = by_iter_eq(List, "combine_if")

        test([], [], True, "map", double)
        test([], [], False, "map", double)
        test([2], [1], True, "map", double)
        test([1], [1], False, "map", double)
        test([2, 4], [1, 2], True, "map", double)
        test([1, 2], [1, 2], False, "map", double)
        test([2, 4, 6], [1, 2, 3], True, "map", double)
        test([1, 2, 3], [1, 2, 3], False, "map", double)

    def test_copy(self):
        def test(items):
            ls = List(items)
            ls2 = ls.copy()
            assert ls is not ls2
            assert ls == ls2

        test([])
        test([1])
        test([1, 2])
        test([2, 1])
        test([1, 2, 3])

    def test_count(self):
        test = by_eq(List, "count")

        test({}, [])
        test({1: 1}, [1])
        test({1: 2}, [1, 1])
        test({1: 1, 2: 1}, [1, 2])
        test({1: 2, 2: 1}, [1, 1, 2])
        test({"a": 1, "b": 2}, ["a", "b", "b"])
        test({"a": 2, "b": 1}, ["a", "b", "a"])

        test({1: 2, 2: 1}, ["a", "ab", "b"], len)
        test({True: 2, False: 1}, [1, 2, 0], bool)

    def test_cycle(self):
        def test(expected, items):
            actual = List(items).cycle().take(5).list()
            assert actual == expected

        test([], [])
        test([1, 1, 1, 1, 1], [1])
        test([1, 2, 1, 2, 1], [1, 2])
        test([1, 2, 3, 1, 2], [1, 2, 3])

    def test_default_dict(self):
        def test(items, *a, **kw):
            result = List(items).default_dict(*a, **kw)
            assert isinstance(result, DefaultDict)
            assert dict(result) == dict(items)

        a, b, c = enumerate("abc")

        test([], int)
        test([a], int)
        test([a, b], int)
        test([], list)
        test([("a", [1, 2])], list)
        test([("a", [1, 2]), ("b", [3, 4])], list)
        test([], str)
        test([("a", "hello")], str)
        test([("a", "hello"), ("b", "world")], str)

        result = List([("a", 1)]).default_dict(int)
        assert result["a"] == 1
        assert result["b"] == 0

        result = List().default_dict(list)
        result["new_list"].append(1)
        assert result["new_list"] == [1]

    def test_dict(self):
        test = by_eq(List, "dict")

        a, b, c = enumerate("abc")

        test({}, [])
        test({0: "a"}, [a])
        test({0: "a", 1: "b"}, [a, b])
        test({0: 1, 1: 2, 2: 3}, enumerate([1, 2, 3]))

    def test_discard(self):
        def test(expected_result, expected_list, items, item):
            ls = List(items)
            result = ls.discard(item)
            assert result == expected_result
            assert ls == expected_list

        test(False, [], [], 1)
        test(True, [], [1], 1)
        test(True, [2], [1, 2], 1)
        test(True, [1, 3], [1, 2, 3], 2)
        test(False, [1, 2, 3], [1, 2, 3], 4)

    def test_distinct(self):
        test = by_iter_eq(List, "distinct")

        test([], [])
        test([1], [1])
        test([1], [1, 1])
        test([1, 2], [1, 2])
        test([1, 2], [1, 1, 2])
        test([1, 2, 3], [1, 2, 3, 2, 1])
        test(["a"], ["a", "a"])
        test(["a", "b"], ["a", "b", "a"])

        test([1, 2], [1, 3, 2, 4], key=lambda x: x % 2)

    def test_do(self):
        def test(items):
            actual = []

            def f(item):
                actual.append(item)

            result = List(items).do(f)
            assert iter_eq(result, items)
            assert actual == items

        test([])
        test([1])
        test([1, 2, 3])
        test(["a", "b"])

    def test_drop(self):
        test = by_iter_eq(List, "drop")

        test([], [], 0)
        test([], [], 1)
        test([], [], 5)
        test([1], [1], 0)
        test([], [1], 1)
        test([], [1], 5)
        test([1, 2, 3], [1, 2, 3], 0)
        test([2, 3], [1, 2, 3], 1)
        test([3], [1, 2, 3], 2)
        test([], [1, 2, 3], 3)
        test([], [1, 2, 3], 5)

    def test_drop_while(self):
        test = by_iter_eq(List, "drop_while")

        test([], [], lambda x: x < 5)
        test([], [1], lambda x: x < 5)
        test([5], [5], lambda x: x < 5)
        test([5, 6], [1, 2, 5, 6], lambda x: x < 5)
        test([5, 1, 6], [1, 2, 5, 1, 6], lambda x: x < 5)
        test([1, 2, 3], [1, 2, 3], lambda x: x > 5)

    def test_enumerate(self):
        test = by_iter_eq(List, "enumerate")

        test([], [])
        test([(0, "a")], ["a"])
        test([(0, "a"), (1, "b")], ["a", "b"])
        test([(0, 1), (1, 2), (2, 3)], [1, 2, 3])

        test([], [], start=5)
        test([(5, "a")], ["a"], start=5)
        test([(5, "a"), (6, "b")], ["a", "b"], start=5)

    def test_extend(self):
        def test(expected, initial, items):
            lst = List(initial)
            result = lst.extend(items)
            assert result is lst
            assert list(lst) == expected

        test([], [], [])
        test([1], [], [1])
        test([1, 2], [1], [2])
        test([1, 2, 3], [1], [2, 3])
        test([1, 2, 3, 4], [1, 2], [3, 4])

    def test_filter(self):
        test = by_iter_eq(List, "filter")

        test([], [], lambda x: x > 0)
        test([], [1, 2, 3], lambda x: x > 5)
        test([1, 2, 3], [1, 2, 3], lambda x: x > 0)
        test([2, 3], [1, 2, 3], lambda x: x > 1)
        test([3], [1, 2, 3], lambda x: x > 2)
        test([1, 3], [1, 2, 3], is_odd)
        test([2], [1, 2, 3], is_even)

    def test_first(self):
        test = by_eq(List, "first")

        with should_raise(StopIteration):
            List([]).first()
        test(1, [1])
        test(1, [1, 2, 3])
        test("a", ["a", "b", "c"])

        with should_raise(StopIteration):
            List([]).first(predicate=lambda x: x > 0)
        with should_raise(StopIteration):
            List([1, 2, 3]).first(predicate=lambda x: x > 5)
        test(2, [1, 2, 3], predicate=lambda x: x > 1)
        test(1, [1, 2, 3], predicate=is_odd)

        test(None, [], default=None)
        test(None, [1, 2, 3], predicate=lambda x: x > 5, default=None)

    def test_flat_map(self):
        test = by_iter_eq(List, "flat_map")

        test([], [], lambda x: [x])
        test([1], [1], lambda x: [x])
        test([1, 1], [1], lambda x: [x, x])
        test([1, 2], [1, 2], lambda x: [x])
        test([1, 1, 2, 2], [1, 2], lambda x: [x, x])
        test([], [1, 2], lambda x: [])
        test([0, 0, 1], [2, 3], lambda x: range(x - 1))

    def test_flatten(self):
        test = by_iter_eq(List, "flatten")

        test([], [])
        test([1], [[1]])
        test([1, 2], [[1, 2]])
        test([1], [[1], []])
        test([1, 2, 3], [[1, 2], [3]])
        test([1, 2, 3], [[1], [2], [3]])
        test([], [[], []])

    def test_fold(self):
        test = by_eq(List, "fold")

        test(0, [], 0, add)
        test(1, [], 1, add)
        test(1, [1], 0, add)
        test(3, [1, 2], 0, add)
        test(6, [1, 2, 3], 0, add)
        test(1, [], 1, mul)
        test(2, [2], 1, mul)
        test(6, [2, 3], 1, mul)

    def test_fold_while(self):
        test = by_eq(List, "fold_while")

        test(0, [], 0, add, lambda acc: acc < 10)
        test(1, [], 1, add, lambda acc: acc < 10)
        test(0, [1], 0, add, lambda acc: acc < 1)
        test(1, [1, 2], 0, add, lambda acc: acc < 2)
        test(6, [1, 2, 3], 0, add, lambda acc: acc < 10)
        test(3, [1, 2, 3], 0, add, lambda acc: acc < 5)

    def test_for_each(self):
        def test(items):
            actual = []

            def f(item):
                actual.append(item)

            List(items).for_each(f)
            assert actual == items

        test([])
        test([1])
        test([1, 2, 3])
        test(["a", "b", "c"])

    def test_group_by(self):
        test = by_eq(List, "group_by")

        test({}, [])
        test({1: [1]}, [1])
        test({1: [1, 1]}, [1, 1])
        test({1: [1], 2: [2]}, [1, 2])
        test({1: [1, 1], 2: [2]}, [1, 1, 2])
        test({True: [1, 3], False: [2]}, [1, 2, 3], is_odd)
        test({1: ["a"], 2: ["ab"]}, ["a", "ab"], len)

    def test_insert(self):
        def test(expected, initial, index, item):
            lst = List(initial)
            result = lst.insert(index, item)
            assert result is lst
            assert list(lst) == expected

        test([1], [], 0, 1)
        test([1, 2], [2], 0, 1)
        test([2, 1], [2], 1, 1)
        test([1, 3, 2], [1, 2], 1, 3)
        test([1, 2, 3], [1, 2], 2, 3)

    def test_intersperse(self):
        test = by_iter_eq(List, "intersperse")

        test([], [], 0)
        test([1], [1], 0)
        test([1, 0, 2], [1, 2], 0)
        test([1, 0, 2, 0, 3], [1, 2, 3], 0)
        test(["a", ',', "b"], ["a", "b"], ',')

    def test_iter(self):
        def test(items):
            result = List(items).iter()
            assert isinstance(result, Iter)
            assert list(result) == items

        test([])
        test([1])
        test([1, 2, 3])
        test(["a", "b"])

    def test_last(self):
        test = by_eq(List, "last")

        with should_raise(StopIteration):
            List([]).last()
        test(1, [1])
        test(3, [1, 2, 3])
        test("c", ["a", "b", "c"])

        with should_raise(StopIteration):
            List([]).last(predicate=lambda x: x > 0)
        with should_raise(StopIteration):
            List([1, 2, 3]).last(predicate=lambda x: x > 5)
        test(3, [1, 2, 3], predicate=lambda x: x > 1)
        test(3, [1, 2, 3], predicate=is_odd)

        test(None, [], default=None)
        test(None, [1, 2, 3], predicate=lambda x: x > 5, default=None)

    def test_list(self):
        def test(items):
            ls = List(items)
            actual = ls.list()
            assert actual is ls

        test([])
        test([1])
        test([1, 2, 3])
        test(["a", "b"])

    def test_map(self):
        test = by_iter_eq(List, "map")

        test([], [], double)
        test([2], [1], double)
        test([2, 4], [1, 2], double)
        test([2, 4, 6], [1, 2, 3], double)
        test(["A"], ["a"], str.upper)
        test([1, 4, 9], [1, 2, 3], lambda x: x * x)

    def test_map_to_keys(self):
        test = by_eq(List, "map_to_keys")

        test({}, [], double)
        test({2: 1}, [1], double)
        test({2: 1}, [1, 1], double)
        test({2: 1, 4: 2}, [1, 2], double)
        test({4: 2, 2: 1}, [2, 1], double)

    def test_map_to_pairs(self):
        test = by_iter_eq(List, "map_to_pairs")

        test([], [], double)
        test([(1, 2)], [1], double)
        test([(1, 2), (1, 2)], [1, 1], double)
        test([(1, 2), (2, 4)], [1, 2], double)
        test([(2, 4), (1, 2)], [2, 1], double)

    def test_map_to_values(self):
        test = by_eq(List, "map_to_values")

        test({}, [], double)
        test({1: 2}, [1], double)
        test({1: 2}, [1, 1], double)
        test({1: 2, 2: 4}, [1, 2], double)
        test({2: 4, 1: 2}, [2, 1], double)

    def test_max(self):
        test = by_eq(List, "max")

        with should_raise(ValueError):
            List([]).max()
        test(1, [1])
        test(3, [1, 2, 3])
        test(3, [3, 1, 2])
        test("c", ["a", "b", "c"])

        test("ccc", ["a", "bb", "ccc"], key=len)

    def test_min(self):
        test = by_eq(List, "min")

        with should_raise(ValueError):
            List([]).min()
        test(1, [1])
        test(1, [1, 2, 3])
        test(1, [3, 1, 2])
        test("a", ["a", "b", "c"])

        test("a", ["a", "bb", "ccc"], key=len)

    def test_min_max(self):
        test = by_eq(List, "min_max")

        with should_raise(ValueError):
            List([]).min_max()
        test((1, 1), [1])
        test((1, 3), [1, 2, 3])
        test((1, 3), [3, 1, 2])
        test(("a", "c"), ["a", "b", "c"])

        test(("a", "ccc"), ["a", "bb", "ccc"], key=len)

    def test_only(self):
        test = by_eq(List, "only")

        with should_raise(ValueError):
            List([]).only()
        test(1, [1])
        with should_raise(ValueError):
            List([1, 2]).only()
        with should_raise(ValueError):
            List([1, 2, 3]).only()

        with should_raise(ValueError):
            List([]).only(predicate=is_odd)
        test(1, [1], predicate=is_odd)
        test(1, [1, 2], predicate=is_odd)
        with should_raise(ValueError):
            List([1, 3]).only(predicate=is_odd)

    def test_partition(self):
        def test(expected_true, expected_false, items, predicate):
            actual_true, actual_false = List(items).partition(predicate)
            assert iter_eq(actual_true, expected_true)
            assert iter_eq(actual_false, expected_false)

        test([], [], [], is_odd)
        test([1], [], [1], is_odd)
        test([], [2], [2], is_odd)
        test([1], [2], [1, 2], is_odd)
        test([1, 3], [2], [1, 2, 3], is_odd)
        test([1, 3, 5], [2, 4], [1, 2, 3, 4, 5], is_odd)

    def test_permutations(self):
        test = by_iter_eq(List, "permutations")

        test([()], [], 0)
        test([()], [1], 0)
        test([()], [1, 2], 0)
        test([], [], 1)
        test([(1,)], [1], 1)
        test([(1,), (2,)], [1, 2], 1)
        test([], [], 2)
        test([], [1], 2)
        test([(1, 2), (2, 1)], [1, 2], 2)
        test([(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)], [1, 2, 3], 2)

    def test_powerset(self):
        test = by_iter_eq(List, "powerset")

        test([()], [])
        test([(), (1,)], [1])
        test([(), (1,), (2,), (1, 2)], [1, 2])
        test([(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)], [1, 2, 3])

    def test_product(self):
        test = by_iter_eq(List, "product")

        test([], [])
        test([], [], [])
        test([], [1], [])
        test([], [], [1])
        test([(1, "a")], [1], ["a"])
        test([(1, "a"), (1, "b")], [1], ["a", "b"])
        test([(1, "a"), (2, "a")], [1, 2], ["a"])
        test([(1, "a"), (1, "b"), (2, "a"), (2, "b")], [1, 2], ["a", "b"])

        test([], [], repeat=2)
        test([()], [1], repeat=0)
        test([(1,)], [1], repeat=1)
        test([(1, 1)], [1], repeat=2)
        test([(1, 1), (1, 2), (2, 1), (2, 2)], [1, 2], repeat=2)

    def test_reduce(self):
        test = by_eq(List, "reduce")

        with should_raise(ValueError):
            List([]).reduce(add)
        test(1, [1], add)
        test(3, [1, 2], add)
        test(6, [1, 2, 3], add)
        test(1, [1], mul)
        test(2, [1, 2], mul)
        test(6, [1, 2, 3], mul)

    def test_repeat(self):
        def test(expected, items, n):
            actual = List(items).repeat(n).take(len(expected)).list()
            assert actual == expected

        test([], [], 3)
        test([], [1], 0)
        test([1, 1, 1], [1], 3)
        test([1, 2, 1, 2, 1], [1, 2], 3)

    def test_reverse(self):
        def test(expected, initial):
            lst = List(initial)
            result = lst.reverse()
            assert result is lst
            assert list(lst) == expected

        test([], [])
        test([1], [1])
        test([2, 1], [1, 2])
        test([3, 2, 1], [1, 2, 3])
        test(["c", "b", "a"], ["a", "b", "c"])

    def test_reversed(self):
        test = by_iter_eq(List, "reversed")

        test([], [])
        test([3], [3])
        test([2, 1], [1, 2])
        test([3, 2, 1], [1, 2, 3])
        test(["c", "b", "a"], ["a", "b", "c"])
        test([5, 4, 3, 2, 1], [1, 2, 3, 4, 5])

    def test_set(self):
        def test(items):
            result = List(items).set()
            assert isinstance(result, Set)
            assert sorted(list(result)) == sorted(list(set(items)))

        test([])
        test([1])
        test([1, 2, 3])
        test([1, 1, 2, 2, 3])
        test(["a", "b", "a"])

    def test_size(self):
        test = by_eq(List, "size")

        test(0, [])
        test(1, [1])
        test(3, [1, 2, 3])
        test(5, ["a", "b", "c", "d", "e"])

    def test_sliding(self):
        test = by_iter_eq(List, "sliding")

        test([], [], 2)
        test([], [1], 2)
        test([(1, 2)], [1, 2], 2)
        test([(1, 2), (2, 3)], [1, 2, 3], 2)
        test([(1, 2, 3)], [1, 2, 3], 3)
        test([(1, 2, 3), (2, 3, 4)], [1, 2, 3, 4], 3)
        test([(1, 2, 3), (2, 3, 4), (3, 4, 5)], [1, 2, 3, 4, 5], 3)

    def test_sliding_by_timestamp(self):
        now = 0

        def clock():
            nonlocal now
            now += 1
            return now

        def test(expected, items, window_size):
            nonlocal now
            now = 0
            stamp = timestamp(clock)
            result = List(items).sliding_by_timestamp(window_size, stamp=stamp)
            actual = [list(window) for window in result]
            assert actual == expected

        test([], [], 2)
        test([[1]], [1], 2)
        test([[1, 2]], [1, 2], 3)
        test([[1, 2], [2, 3]], [1, 2, 3], 2)

    def test_sort(self):
        def test(expected, initial, *args, **kwargs):
            lst = List(initial)
            result = lst.sort(*args, **kwargs)
            assert result is lst
            assert list(lst) == expected

        test([], [])
        test([1], [1])
        test([1, 2, 3], [3, 1, 2])
        test([1, 2, 2, 3], [2, 3, 1, 2])
        test(["a", "b", "c"], ["c", "a", "b"])
        test(["a", "bb", "ccc"], ["ccc", "a", "bb"], key=len)
        test([3, 2, 1], [1, 2, 3], reverse=True)

    def test_sorted(self):
        test = by_iter_eq(List, "sorted")

        test([], [])
        test([1], [1])
        test([1, 2, 3], [3, 1, 2])
        test([1, 2, 3], [1, 2, 3])
        test(["a", "b", "c"], ["c", "a", "b"])
        test([3, 2, 1], [1, 2, 3], reverse=True)
        test(["c", "b", "a"], ["a", "b", "c"], reverse=True)
        test(["a", "bb", "ccc"], ["bb", "ccc", "a"], key=len)
        test(["ccc", "bb", "a"], ["bb", "ccc", "a"], key=len, reverse=True)

    def test_take(self):
        test = by_iter_eq(List, "take")

        test([], [], 0)
        test([], [], 3)
        test([], [1, 2, 3], 0)
        test([1], [1, 2, 3], 1)
        test([1, 2], [1, 2, 3], 2)
        test([1, 2, 3], [1, 2, 3], 3)
        test([1, 2, 3], [1, 2, 3], 5)

    def test_take_while(self):
        test = by_iter_eq(List, "take_while")

        test([], [], lambda x: x < 5)
        test([], [5, 6, 7], lambda x: x < 5)
        test([1, 2], [1, 2, 5, 6], lambda x: x < 5)
        test([1, 2, 3], [1, 2, 3], lambda x: x < 5)
        test([1], [1, 5, 2], lambda x: x < 5)

    def test_timestamp(self):
        now = 0

        def clock():
            nonlocal now
            now += 1
            return now

        def test(expected, items):
            nonlocal now
            now = 0
            actual = List(items).timestamp(clock=clock)
            assert iter_eq(actual, expected)

        test([], [])
        test([(1, "a")], ["a"])
        test([(1, "a"), (2, "b")], ["a", "b"])
        test([(1, 1), (2, 2), (3, 3)], [1, 2, 3])

    def test_tqdm(self):
        # tqdm shouldn't change the input, and shouldn't raise any exceptions
        def test(items):
            actual = List(items).tqdm()
            assert iter_eq(actual, items)

        test([])
        test([1])
        test([1, 2, 3])
        test(["a", "b", "c"])

    def test_transpose(self):
        test = by_iter_eq(List, "transpose")

        test([], [])
        test([(1,), (2,), (3,)], [[1, 2, 3]])
        test([(1, 4), (2, 5), (3, 6)], [[1, 2, 3], [4, 5, 6]])
        test([(1, 4, 7), (2, 5, 8), (3, 6, 9)], [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_tuple(self):
        def test(items):
            actual = List(items).tuple()
            assert isinstance(actual, tuple)
            assert isinstance(actual, Tuple)
            assert iter_eq(actual, items)

        test([])
        test([1])
        test([1, 2, 3])
        test(["a", "b"])

    def test_unzip(self):
        pass  # direct alias of transpose, does not need further testing

    def test_zip(self):
        test = by_iter_eq(List, "zip")

        test([], [], [])
        test([], [1], [])
        test([], [], [1])
        test([(1, "a")], [1], ["a"])
        test([(1, "a"), (2, "b")], [1, 2], ["a", "b"])
        test([(1, "a")], [1, 2], ["a"])
        test([(1, "a")], [1], ["a", "b"])

    def test_zip_longest(self):
        test = by_iter_eq(List, "zip_longest")

        test([], [], [])
        test([(1, None)], [1], [])
        test([(None, "a")], [], ["a"])
        test([(1, "a")], [1], ["a"])
        test([(1, "a"), (2, "b")], [1, 2], ["a", "b"])
        test([(1, "a"), (2, None)], [1, 2], ["a"])
        test([(1, "a"), (None, "b")], [1], ["a", "b"])


class TestRange:
    def test___contains__(self):
        test = by_eq(Range, "__contains__")

        test(True, range(10), 5)
        test(False, range(10), 15)
        test(True, range(5, 10), 7)
        test(False, range(5, 10), 3)
        test(True, range(0, 10, 2), 4)
        test(False, range(0, 10, 2), 5)

    def test___eq__(self):
        test = by_eq(Range, "__eq__")

        test(True, range(10), Range(10))
        test(True, range(10), range(10))
        test(False, range(10), Range(5))
        test(False, range(10), range(5))
        test(True, range(5, 10), Range(5, 10))
        test(False, range(5, 10), Range(5, 15))
        test(True, range(0, 10, 2), Range(0, 10, 2))
        test(False, range(0, 10, 2), Range(0, 10, 3))

    def test___getitem__(self):
        test = by_eq(Range, "__getitem__")

        test(5, range(10), 5)
        test(7, range(5, 10), 2)
        test(4, range(0, 10, 2), 2)
        test(Range(3, 5), range(10), slice(3, 5))

    def test___hash__(self):
        def test(*a, **kw):
            expected = hash(range(*a, **kw))
            actual = hash(Range(*a, **kw))
            assert actual == expected

        test(10)
        test(5, 10)
        test(0, 10, 2)

    def test___init__(self):
        def test(expected, *a, **kw):
            r = Range(*a, **kw)
            assert iter_eq(r, expected)

        test([0, 1, 2], 3)
        test([5, 6, 7, 8, 9], 5, 10)
        test([0, 2, 4, 6, 8], 0, 10, 2)
        test([10, 9, 8, 7, 6], 10, 5, -1)
        test([0, 1, 2], range(3))
        test([0, 1, 2], Range(3))

        with should_raise(RequirementException):
            Range(range(3), step=2)

    def test___iter__(self):
        test = by_iter_eq(Range, "__iter__")

        test([0, 1, 2, 3, 4], range(5))
        test([5, 6, 7, 8, 9], range(5, 10))
        test([0, 2, 4, 6, 8], range(0, 10, 2))

    def test___len__(self):
        test = by_eq(Range, "__len__")

        test(5, range(5))
        test(5, range(5, 10))
        test(5, range(0, 10, 2))
        test(0, range(0))
        test(0, range(5, 3))

    def test___repr__(self):
        test = by_eq(Range, "__repr__")

        test("Range(0, 5)", range(5))
        test("Range(5, 10)", range(5, 10))
        test("Range(0, 10, 2)", range(0, 10, 2))

    def test___reversed__(self):
        test = by_iter_eq(Range, "reversed")

        test([4, 3, 2, 1, 0], range(5))
        test([9, 8, 7, 6, 5], range(5, 10))
        test([8, 6, 4, 2, 0], range(0, 10, 2))

    def test_start(self):
        test = by_member_eq(Range, "start")

        test(0, range(5))
        test(5, range(5, 10))
        test(0, range(0, 10, 2))

    def test_stop(self):
        test = by_member_eq(Range, "stop")

        test(5, range(5))
        test(10, range(5, 10))
        test(10, range(0, 10, 2))

    def test_step(self):
        test = by_member_eq(Range, "step")

        test(1, range(5))
        test(1, range(5, 10))
        test(2, range(0, 10, 2))

    def test_all(self):
        test = by_eq(Range, "all")

        test(True, range(0))
        test(False, range(3))
        test(True, range(1, 3))
        test(False, range(0, 3))
        test(True, range(1, 5, 2), is_odd)
        test(False, range(0, 4, 2), is_odd)
        test(False, range(0, 10, 3), is_odd)

    def test_any(self):
        test = by_eq(Range, "any")

        test(False, range(0))
        test(True, range(3))
        test(True, range(1, 3))
        test(True, range(0, 3))
        test(True, range(1, 5, 2), is_odd)
        test(False, range(0, 4, 2), is_odd)
        test(True, range(0, 10, 3), is_odd)

    def test_apply(self):
        test = by_eq(Range, "apply")

        test(0, range(0), len)
        test(5, range(5), len)
        test(5, range(5, 10), len)
        test(5, range(0, 10, 2), len)
        test(0, range(5, 2), len)
        test(3, range(5, 2, -1), len)

    def test_apply_and_wrap(self):
        test = by_iter_eq(Range, "apply_and_wrap")

        test([4, 3, 2, 1, 0], range(5), reversed)
        test([9, 8, 7, 6, 5], range(5, 10), reversed)
        test([8, 6, 4, 2, 0], range(0, 10, 2), reversed)

    def test_batch(self):
        test = by_iter_eq(Range, "batch")

        test([range(0)], range(0), 2)
        test([range(2)], range(2), 2)
        test([range(2)], range(2), 3)
        test([range(2), range(2, 3)], range(3), 2)
        test([range(1), range(1, 2), range(2, 3)], range(3), 1)

    def test_chain(self):
        test = by_iter_eq(Range, "chain")

        test([10, 11], range(0), [10, 11])
        test([], range(0), [])
        test([0, 1, 2], range(3), [])
        test([0, 1, 2, 10, 11], range(3), [10, 11])
        test([0, 1, 2, 5, 6, 10, 11], range(3), range(5, 7), [10, 11])

    def test_combinations(self):
        test = by_iter_eq(Range, "combinations")

        test([()], range(0), 0)
        test([()], range(1), 0)
        test([()], range(2), 0)
        test([()], range(3), 0)
        test([], range(0), 1)
        test([], range(5), 6)
        test([(0,)], range(1), 1)
        test([(0,), (1,)], range(2), 1)
        test([(0,), (1,), (2,)], range(3), 1)
        test([(0, 1)], range(2), 2)
        test([(0, 1), (0, 2), (1, 2)], range(3), 2)
        test([()], range(0), 0, with_replacement=True)
        test([()], range(1), 0, with_replacement=True)
        test([()], range(2), 0, with_replacement=True)
        test([()], range(3), 0, with_replacement=True)
        test([], range(0), 1, with_replacement=True)
        test([(0,)], range(1), 1, with_replacement=True)
        test([(0,), (1,)], range(2), 1, with_replacement=True)
        test([(0,), (1,), (2,)], range(3), 1, with_replacement=True)
        test([], range(0), 2, with_replacement=True)
        test([(0, 0)], range(1), 2, with_replacement=True)
        test([(0, 0), (0, 1), (1, 1)], range(2), 2, with_replacement=True)
        test([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)], range(3), 2, with_replacement=True)
        test([(5,), (6,), (7,)], range(5, 8), 1)
        test([(5, 6), (5, 7), (6, 7)], range(5, 8), 2)
        test([(0,), (2,), (4,)], range(0, 6, 2), 1)
        test([(0, 2), (0, 4), (2, 4)], range(0, 6, 2), 2)
        test([(0, 0), (0, 2), (0, 4), (2, 2), (2, 4), (4, 4)], range(0, 6, 2), 2, with_replacement=True)

    def test_combine_if(self):
        test = by_iter_eq(Range, "combine_if")

        test([], range(0), False, "map", double)
        test([], range(0), True, "map", double)
        test([0], range(1), False, "map", double)
        test([0], range(1), True, "map", double)
        test([1], range(1, 2), False, "map", double)
        test([2], range(1, 2), True, "map", double)
        test([0, 1, 2], range(3), False, "map", double)
        test([0, 2, 4], range(3), True, "map", double)
        test([1, 2, 3], range(1, 4), False, "map", double)
        test([2, 4, 6], range(1, 4), True, "map", double)

        test([], range(0), False, "filter", is_even)
        test([], range(0), True, "filter", is_even)
        test([1], range(1, 2), False, "filter", is_even)
        test([], range(1, 2), True, "filter", is_even)
        test([0, 1, 2], range(3), False, "filter", is_even)
        test([0, 2], range(3), True, "filter", is_even)
        test([1, 2, 3], range(1, 4), False, "filter", is_even)
        test([2], range(1, 4), True, "filter", is_even)

        test([1, 2], range(1, 3), False, "take", 5)
        test([1], range(1, 3), True, "take", 1)
        test([0, 2, 4], range(0, 6, 2), False, "take", 5)
        test([0, 2], range(0, 6, 2), True, "take", 2)

    def test_count(self):
        test = by_eq(Range, "count")

        test({}, range(0))
        test({0: 1, 1: 1, 2: 1}, range(3))
        test({2: 1, 3: 1, 4: 1}, range(2, 5))
        test({0: 1, 2: 1, 4: 1}, range(0, 6, 2))
        test({1: 1, 3: 1}, range(1, 5, 2))

        test({True: 1, False: 1}, range(2), is_even)
        test({False: 2}, range(1, 5, 2), is_even)
        test({True: 3}, range(0, 6, 2), is_even)

    def test_cycle(self):
        def test(expected, items):
            actual = Range(items).cycle().take(10)
            assert iter_eq(actual, expected)

        # test([], range(0))
        test([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], range(3))
        test([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], range(5))
        test([5, 6, 7, 5, 6, 7, 5, 6, 7, 5], range(5, 8))
        test([0, 2, 4, 0, 2, 4, 0, 2, 4, 0], range(0, 6, 2))
        test([1, 3, 5, 1, 3, 5, 1, 3, 5, 1], range(1, 6, 2))
        test([2, 2, 2, 2, 2, 2, 2, 2, 2, 2], range(2, 3))

    def test_do(self):
        def test(items):
            actual = []

            def f(item):
                actual.append(item)

            result = Range(items).do(f)
            assert iter_eq(result, items)
            assert iter_eq(actual, items)

        test(range(0))
        test(range(1))
        test(range(5))
        test(range(5, 10))
        test(range(0, 10, 2))
        test(range(5, 2, -1))

    def test_drop(self):
        test = by_eq(Range, "drop")

        test(range(5), range(5), 0)
        test(range(1, 5), range(5), 1)
        test(range(2, 5), range(5), 2)
        test(range(5, 5), range(5), 5)
        test(range(5, 5), range(5), 10)

        test(range(5, 10), range(5, 10), 0)
        test(range(7, 10), range(5, 10), 2)
        test(range(10, 10), range(5, 10), 5)
        test(range(10, 10), range(5, 10), 10)

        test(range(0, 10, 2), range(0, 10, 2), 0)
        test(range(2, 10, 2), range(0, 10, 2), 1)
        test(range(4, 10, 2), range(0, 10, 2), 2)
        test(range(10, 10, 2), range(0, 10, 2), 5)

    def test_drop_while(self):
        test = by_iter_eq(Range, "drop_while")

        test([], range(0), less_than(5))
        test([5, 6, 7], range(0, 8), less_than(5))
        test([3, 4, 5], range(1, 6), less_than(3))
        test([], range(1, 5), less_than(10))
        test([1, 2, 3, 4], range(1, 5), less_than(1))

        test([6, 8], range(0, 10, 2), less_than(6))
        test([], range(2, 10, 2), less_than(20))
        test([2, 4, 6, 8], range(2, 10, 2), less_than(2))

        test([], range(5), less_than(10))
        test([5, 6, 7, 8, 9], range(10), less_than(5))
        test([0, 1, 2, 3, 4], range(5), less_than(0))

    def test_enumerate(self):
        test = by_iter_eq(Range, "enumerate")

        test([], range(0))
        test([(0, 0)], range(1))
        test([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)], range(5))
        test([(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)], range(5, 10))
        test([(0, 0), (1, 2), (2, 4), (3, 6), (4, 8)], range(0, 10, 2))

        test([(1, 0)], range(1), start=1)
        test([(2, 0), (3, 1), (4, 2), (5, 3), (6, 4)], range(5), start=2)
        test([(10, 5), (11, 6), (12, 7)], range(5, 8), start=10)

    def test_filter(self):
        test = by_iter_eq(Range, "filter")

        test([], range(0), is_even)
        test([0, 2, 4], range(5), is_even)
        test([1, 3], range(5), is_odd)
        test([6, 8], range(5, 10), is_even)
        test([5, 7, 9], range(5, 10), is_odd)
        test([0, 2, 4, 6, 8], range(0, 10, 2), is_even)
        test([], range(1, 10, 2), is_even)
        test([0, 1, 2, 3, 4], range(10), less_than(5))
        test([5, 6, 7, 8, 9], range(10), greater_than(4))

    def test_first(self):
        test = by_eq(Range, "first")

        test(0, range(5))
        test(5, range(5, 10))
        test(0, range(0, 10, 2))
        test(1, range(1, 2))

        test(0, range(5), is_even)
        test(1, range(5), is_odd)
        test(6, range(5, 10), is_even)
        test(5, range(5, 10), is_odd)

        test("default", range(0), default="default")
        test("default", range(1, 5, 2), is_even, "default")
        test("default", range(5), greater_than(10), "default")

        with should_raise(StopIteration):
            Range(0).first()
        with should_raise(StopIteration):
            Range(1, 5, 2).first(is_even)

    def test_flat_map(self):
        test = by_iter_eq(Range, "flat_map")

        test([], range(0), tower)
        test([], range(1), tower)
        test([1], range(2), tower)
        test([1, 2, 2], range(3), tower)
        test([1, 2, 2, 3, 3, 3], range(4), tower)
        test([2, 2, 3, 3, 3, 4, 4, 4, 4], range(2, 5), tower)
        test([2, 2, 4, 4, 4, 4], range(0, 6, 2), tower)

        test([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], range(5), lambda x: [x, x])
        test([], range(5), lambda x: [])
        test([0, 1, 2, 3, 4], range(5), lambda x: [x])

    def test_fold(self):
        test = by_eq(Range, "fold")

        test(0, range(0), 0, add)
        test(1, range(0), 1, add)
        test(10, range(1, 5), 0, add)
        test(11, range(1, 5), 1, add)
        test(24, range(1, 5), 1, mul)
        test(120, range(1, 6), 1, mul)
        test(9, range(2, 5), 0, add)
        test(6, range(0, 6, 2), 0, add)

    def test_fold_while(self):
        test = by_eq(Range, "fold_while")

        test(0, range(0), 0, add, lambda acc: acc < 10)
        test(1, range(0), 1, add, lambda acc: acc < 10)
        test(0, range(1, 2), 0, add, lambda acc: acc < 1)
        test(1, range(1, 3), 0, add, lambda acc: acc < 2)
        test(6, range(1, 4), 0, add, lambda acc: acc < 10)
        test(3, range(1, 4), 0, add, lambda acc: acc < 5)

    def test_for_each(self):
        def test(items):
            actual = []

            def f(item):
                actual.append(item)

            Range(items).for_each(f)
            assert iter_eq(actual, items)

        test(range(0))
        test(range(1, 2))
        test(range(1, 4))

    def test_group_by(self):
        test = by_eq(Range, "group_by")

        test({}, range(0))
        test({1: [1]}, range(1, 2))
        test({1: [1], 2: [2]}, range(1, 3))
        test({True: [1, 3], False: [2]}, range(1, 4), is_odd)

    def test_index(self):
        test = by_eq(Range, "index")

        with should_raise(ValueError):
            Range(0).index(1)
        test(0, range(1), 0)
        test(0, range(2), 0)
        test(0, range(3), 0)
        test(2, range(3), 2)
        test(1, range(2, 4), 3)

    def test_intersperse(self):
        test = by_iter_eq(Range, "intersperse")

        test([], range(0), 0)
        test([1], range(1, 2), 0)
        test([1, 0, 2], range(1, 3), 0)
        test([1, 0, 2, 0, 3], range(1, 4), 0)

    def test_iter(self):
        def test(items):
            actual = Range(items).iter()
            assert isinstance(actual, Iter)
            assert iter_eq(actual, items)

        test(range(0))
        test(range(1))
        test(range(3))

    def test_last(self):
        test = by_eq(Range, "last")

        test(4, range(5))
        test(9, range(5, 10))
        test(8, range(0, 10, 2))
        test(1, range(1, 2))

        test(4, range(5), is_even)
        test(3, range(5), is_odd)
        test(8, range(5, 10), is_even)
        test(9, range(5, 10), is_odd)

        test("default", range(0), default="default")
        test("default", range(1, 5, 2), is_even, "default")
        test("default", range(5), greater_than(10), "default")

        with should_raise(StopIteration):
            Range(0).last()
        with should_raise(StopIteration):
            Range(1, 5, 2).last(is_even)

    def test_list(self):
        def test(items):
            actual = Range(items).list()
            assert isinstance(actual, List)
            assert iter_eq(actual, items)

        test(range(0))
        test(range(1))
        test(range(5))
        test(range(1, 6))
        test(range(0, 10, 2))
        test(range(1, 8, 3))
        test(range(10, 0, -1))
        test(range(5, -5, -2))

    def test_map(self):
        test = by_iter_eq(Range, "map")

        test([], range(0), double)
        test([0], range(1), double)
        test([0, 2, 4, 6, 8], range(5), double)
        test([10, 12, 14, 16, 18], range(5, 10), double)
        test([0, 4, 8, 12, 16], range(0, 10, 2), double)
        test([2, 6, 10], range(1, 6, 2), double)

        test([1, 4, 9, 16, 25], range(1, 6), lambda x: x * x)
        test([0, 1, 4, 9, 16], range(5), lambda x: x * x)
        test("01234", range(5), str)

    def test_map_to_keys(self):
        test = by_eq(Range, "map_to_keys")

        test({}, range(0), double)
        test({0: 0}, range(1), double)
        test({0: 0, 2: 1, 4: 2}, range(3), double)
        test({10: 5, 12: 6, 14: 7}, range(5, 8), double)

    def test_map_to_pairs(self):
        test = by_iter_eq(Range, "map_to_pairs")

        test([], range(0), double)
        test([(0, 0)], range(1), double)
        test([(0, 0), (1, 2), (2, 4)], range(3), double)
        test([(5, 10), (6, 12), (7, 14)], range(5, 8), double)

    def test_map_to_values(self):
        test = by_eq(Range, "map_to_values")

        test({}, range(0), double)
        test({0: 0}, range(1), double)
        test({0: 0, 1: 2, 2: 4}, range(3), double)
        test({5: 10, 6: 12, 7: 14}, range(5, 8), double)

    def test_max(self):
        test = by_eq(Range, "max")

        with should_raise(ValueError):
            Range(0).max()
        test("default", range(0), default="default")
        test(0, range(1))
        test(4, range(5))
        test(9, range(5, 10))
        test(8, range(0, 10, 2))
        test(5, range(1, 7, 2))

        test(1, range(-5, 2))
        test(-10, range(-15, -9))

    def test_min(self):
        test = by_eq(Range, "min")

        with should_raise(ValueError):
            Range(0).min()
        test("default", range(0), default="default")
        test(0, range(1))
        test(0, range(5))
        test(5, range(5, 10))
        test(0, range(0, 10, 2))
        test(1, range(1, 7, 2))

        test(-5, range(-5, 2))
        test(-15, range(-15, -9))

    def test_min_max(self):
        test = by_eq(Range, "min_max")

        with should_raise(ValueError):
            Range(0).min_max()
        test("default", range(0), default="default")
        test((0, 0), range(1))
        test((0, 4), range(5))
        test((5, 9), range(5, 10))
        test((0, 8), range(0, 10, 2))
        test((1, 5), range(1, 7, 2))

        test((-5, 1), range(-5, 2))
        test((-15, -10), range(-15, -9))
        test((10, 10), range(10, 11))

    def test_only(self):
        test = by_eq(Range, "only")

        with should_raise(ValueError):
            Range(0).only()
        with should_raise(ValueError):
            Range(5).only(is_even)
        test(1, range(1, 2))
        test(0, range(1), is_even)
        test(1, range(1, 3), is_odd)

    def test_partition(self):
        def test(expected_true, expected_false, items, predicate):
            actual_true, actual_false = Range(items).partition(predicate)
            assert iter_eq(actual_true, expected_true)
            assert iter_eq(actual_false, expected_false)

        test([], [], range(0), is_odd)
        test([1], [], range(1, 2), is_odd)
        test([], [2], range(2, 3), is_odd)
        test([1], [2], range(1, 3), is_odd)
        test([1, 3], [2], range(1, 4), is_odd)
        test([1, 3, 5], [2, 4], range(1, 6), is_odd)

    def test_permutations(self):
        test = by_iter_eq(Range, "permutations")

        test([()], range(0), 0)
        test([()], range(1, 2), 0)
        test([()], range(1, 3), 0)
        test([], range(0), 1)
        test([(1,)], range(1, 2), 1)
        test([(1,), (2,)], range(1, 3), 1)
        test([], range(0), 2)
        test([], range(1, 2), 2)
        test([(1, 2), (2, 1)], range(1, 3), 2)
        test([(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)], range(1, 4), 2)

    def test_powerset(self):
        test = by_iter_eq(Range, "powerset")

        test([()], range(0))
        test([(), (1,)], range(1, 2))
        test([(), (1,), (2,), (1, 2)], range(1, 3))
        test([(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)], range(1, 4))

    def test_product(self):
        test = by_iter_eq(Range, "product")

        test([], range(0))
        test([], range(0), [])
        test([], range(1, 2), [])
        test([], range(0), [1])
        test([(1, "a")], range(1, 2), ["a"])
        test([(1, "a"), (1, "b")], range(1, 2), ["a", "b"])
        test([(1, "a"), (2, "a")], range(1, 3), ["a"])
        test([(1, "a"), (1, "b"), (2, "a"), (2, "b")], range(1, 3), ["a", "b"])

        test([], range(0), repeat=2)
        test([()], range(1, 2), repeat=0)
        test([(1,)], range(1, 2), repeat=1)
        test([(1, 1)], range(1, 2), repeat=2)
        test([(1, 1), (1, 2), (2, 1), (2, 2)], range(1, 3), repeat=2)

    def test_reduce(self):
        test = by_eq(Range, "reduce")

        with should_raise(ValueError):
            Range(0).reduce(add)
        test(0, range(1), add)
        test(1, range(1, 2), add)
        test(6, range(1, 4), add)
        test(0, range(1), mul)
        test(6, range(1, 4), mul)

    def test_repeat(self):
        test = by_iter_eq(Range, "repeat")

        test([], range(0), 3)
        test([0, 0, 0], range(1), 3)
        test([0, 1, 2, 0, 1, 2], range(3), 2)
        test([1, 2, 3, 1, 2, 3, 1, 2, 3], range(1, 4), 3)

    def test_reversed(self):
        test = by_eq(Range, "reversed")

        test(range(0), range(0))
        test(range(4, -1, -1), range(5))
        test(range(9, 4, -1), range(5, 10))
        test(range(1, 0, -1), range(1, 2))

        test(range(8, -2, -2), range(0, 10, 2))
        test(range(9, -1, -2), range(1, 11, 2))
        test(range(15, 2, -3), range(3, 18, 3))

        test(range(10, 0, 1), range(1, 11, -1))
        test(range(5, 15, 1), range(14, 4, -1))

    def test_set(self):
        def test(items):
            actual = Range(items).set()
            assert isinstance(actual, Set)
            assert sorted(list(actual)) == sorted(list(set(items)))

        test(range(0))
        test(range(1, 2))
        test(range(1, 4))

    def test_size(self):
        test = by_eq(Range, "size")

        test(0, range(0))
        test(1, range(1))
        test(5, range(5))
        test(5, range(5, 10))
        test(1, range(1, 2))
        test(5, range(0, 10, 2))
        test(3, range(1, 7, 2))
        test(7, range(-5, 2))
        test(5, range(-15, -10))

    def test_sliding(self):
        test = by_iter_eq(Range, "sliding")

        test([], range(0), 2)
        test([], range(1, 2), 2)
        test([range(1, 3)], range(1, 3), 2)
        test([range(1, 3), range(2, 4)], range(1, 4), 2)
        test([range(1, 4)], range(1, 4), 3)
        test([range(1, 4), range(2, 5)], range(1, 5), 3)
        test([range(1, 4), range(2, 5), range(3, 6)], range(1, 6), 3)

    def test_sliding_by_timestamp(self):
        now = 0

        def clock():
            nonlocal now
            now += 1
            return now

        def test(expected, items, window_size):
            nonlocal now
            now = 0
            stamp = timestamp(clock)
            result = Range(items).sliding_by_timestamp(window_size, stamp=stamp)
            actual = [list(window) for window in result]
            assert actual == expected

        test([], range(0), 2)
        test([[1]], range(1, 2), 2)
        test([[1, 2]], range(1, 3), 3)
        test([[1, 2], [2, 3]], range(1, 4), 2)

    def test_take(self):
        test = by_eq(Range, "take")

        test(range(0), range(5), 0)
        test(range(1), range(5), 1)
        test(range(2), range(5), 2)
        test(range(5), range(5), 5)
        test(range(5), range(5), 10)

        test(range(5, 5), range(5, 10), 0)
        test(range(5, 7), range(5, 10), 2)
        test(range(5, 10), range(5, 10), 5)
        test(range(5, 10), range(5, 10), 10)

        test(range(0, 0, 2), range(0, 10, 2), 0)
        test(range(0, 2, 2), range(0, 10, 2), 1)
        test(range(0, 4, 2), range(0, 10, 2), 2)
        test(range(0, 6, 2), range(0, 10, 2), 3)

    def test_take_while(self):
        test = by_iter_eq(Range, "take_while")

        test([], range(0), lambda x: x < 5)
        test([], range(5, 8), lambda x: x < 5)
        test([1, 2, 3], range(1, 4), lambda x: x < 5)
        test([1, 2, 3, 4], range(1, 7), lambda x: x < 5)

    def test_timestamp(self):
        now = 0

        def clock():
            nonlocal now
            now += 1
            return now

        def test(expected, items):
            nonlocal now
            now = 0
            actual = Range(items).timestamp(clock=clock)
            assert iter_eq(actual, expected)

        test([], range(0))
        test([(1, 0)], range(1))
        test([(1, 0), (2, 1)], range(2))
        test([(1, 0), (2, 1), (3, 2)], range(3))

    def test_tqdm(self):
        # tqdm shouldn't change the input, and shouldn't raise any exceptions
        def test(items):
            actual = Range(items).tqdm()
            assert iter_eq(actual, items)

        test(range(0))
        test(range(1))
        test(range(2))

    def test_tuple(self):
        def test(items):
            actual = Range(items).tuple()
            assert isinstance(actual, tuple)
            assert isinstance(actual, Tuple)
            assert iter_eq(actual, items)

        test(range(0))
        test(range(1))
        test(range(3))

    def test_zip(self):
        test = by_iter_eq(Range, "zip")

        test([], range(0), [])
        test([], range(1), [])
        test([], range(0), [1])
        test([(0, "a")], range(1), ["a"])
        test([(0, "a"), (1, "b")], range(2), ["a", "b"])
        test([(0, "a")], range(2), ["a"])
        test([(0, "a")], range(1), ["a", "b"])

    def test_zip_longest(self):
        test = by_iter_eq(Range, "zip_longest")

        test([], range(0), [])
        test([(0, None)], range(1), [])
        test([(None, "a")], range(0), ["a"])
        test([(0, "a")], range(1), ["a"])
        test([(0, "a"), (1, "b")], range(2), ["a", "b"])
        test([(0, "a"), (1, None)], range(2), ["a"])
        test([(0, "a"), (None, "b")], range(1), ["a", "b"])


# for deterministic testing
@patch("ajprax.collections.Set.__iter__", lambda self: Iter(sorted(set.__iter__(self))))
class TestSet:
    def test___add__(self):
        test = by_eq(Set, "__add__")

        assert Set([1, 2]) + [3, 4] == {1, 2, 3, 4}
        assert isinstance(Set([1, 2]) + [3, 4], Set)

        test(set(), set(), set())
        test({1}, set(), {1})
        test({1}, {1}, set())
        test({1, 2}, {1}, {2})
        test({1, 2}, {2}, {1})

    def test___and__(self):
        test = by_eq(Set, "__and__")

        assert Set({1, 2}) & {2, 3} == {2}
        assert isinstance(Set({1, 2}) & {2, 3}, Set)

        test(set(), set(), set())
        test(set(), set(), {1})
        test(set(), {1}, set())
        test(set(), {1}, {2})
        test(set(), {2}, {1})
        test({2}, {1, 2}, {2, 3})

    def test___iadd__(self):
        test = by_eq(Set, "__iadd__")

        s = Set()
        s += {1}
        assert s == {1}
        assert isinstance(s, Set)

        test(set(), set(), set())
        test({1}, set(), {1})
        test({1}, {1}, set())
        test({1, 2}, {1}, {2})
        test({1, 2}, {2}, {1})

    def test___iter__(self):
        def test(items):
            iterator = iter(Set(items))
            assert isinstance(iterator, Iter)
            assert set(iterator) == items

        test(set())
        test({1, 2, 3})

        items = set()
        for item in Set({1, 2, 3}):
            items.add(item)
        assert items == {1, 2, 3}

    def test___isub__(self):
        test = by_eq(Set, "__isub__")

        s = Set({1, 2})
        s -= {1}
        assert s == {2}
        assert isinstance(s, Set)

        test(set(), set(), set())
        test(set(), set(), {1})
        test({1}, {1}, set())
        test({1}, {1}, {2})
        test({2}, {2}, {1})

    def test___or__(self):
        test = by_eq(Set, "__or__")

        assert Set({1}) | {2} == {1, 2}
        assert isinstance(Set({1}) | {2}, Set)

        test(set(), set(), set())
        test({1}, set(), {1})
        test({1}, {1}, set())
        test({1, 2}, {1}, {2})
        test({1, 2}, {2}, {1})
        test({1}, {1}, {1})

    def test___sub__(self):
        test = by_eq(Set, "__sub__")

        assert Set({1, 2}) - {2} == {1}
        assert isinstance(Set({1, 2}) - {2}, Set)

        test(set(), set(), set())
        test(set(), set(), {1})
        test({1}, {1}, set())
        test({1}, {1}, {2})
        test({2}, {2}, {1})

    def test___xor__(self):
        test = by_eq(Set, "__xor__")

        assert Set({1, 2}) ^ {2, 3} == {1, 3}
        assert isinstance(Set({1, 2}) ^ {2, 3}, Set)

        test(set(), set(), set())
        test({1}, set(), {1})
        test({1}, {1}, set())
        test({1, 2}, {1}, {2})
        test({1, 3}, {1, 2}, {2, 3})

    def test_add(self):
        test = by_eq(Set, "__add__")

        assert Set({1}) + {2} == {1, 2}
        assert isinstance(Set({1}) + {2}, Set)

        test(set(), set(), set())
        test({1}, set(), {1})
        test({1}, {1}, set())
        test({1, 2}, {1}, {2})
        test({1, 2}, {2}, {1})

    def test_all(self):
        test = by_eq(Set, "all")

        test(True, [])
        test(True, [1, 2, 3, True])
        test(False, [1, 2, 0, 3])
        test(True, [True, True, True])
        test(False, [True, False, True])
        test(True, [1, 3, 5, 7], key=is_odd)
        test(False, [1, 2, 3, 5], key=is_odd)

    def test_any(self):
        test = by_eq(Set, "any")

        test(False, [])
        test(True, [1, 2, 3, True])
        test(False, [False, False, False])
        test(True, [False, True, False])
        test(False, [2, 4, 6, 8], key=is_odd)
        test(True, [2, 3, 4, 6], key=is_odd)

    def test_apply(self):
        test = by_eq(Set, "apply")

        test(False, [], bool)
        test(True, [1], bool)
        test(3, [1, 2, 3], len)
        test(0, [], len)
        test(6, [1, 2, 3], sum)

    def test_apply_and_wrap(self):
        test = by_iter_eq(Set, "apply_and_wrap")

        test(set(), [], identity)
        test({1}, [1], identity)
        test({1, 2, 3}, [1, 2, 3], identity)

    def test_batch(self):
        test = by_iter_eq(Set, "batch")

        with should_raise(RequirementException):
            Set().batch(0)

        test(set(), [], 1)
        test([{1}], [1], 1)
        test([{1}, {2}], [1, 2], 1)
        test([{1, 2}], [1, 2], 2)
        test([{1, 2}, {3}], [1, 2, 3], 2)
        test([{1, 2, 3}], [1, 2, 3], 3)
        test([{1, 2, 3}], [1, 2, 3], 4)

    def test_chain(self):
        test = by_iter_eq(Set, "chain")

        test([], [])
        test([], [], [])
        test([], [], [], [])
        test([1], [1], [])
        test([1, 2], [1, 2], [])
        test([1, 2, 3], [1, 2], [3])
        test([1, 2, 3, 4], [1, 2], [3, 4])
        test([1, 2, 3, 4, 5], [1], [2, 3], [4, 5])

    def test_clear(self):
        def test(items):
            s = Set(items)
            result = s.clear()
            assert result is s
            assert not s

        test([])
        test([1])
        test([1, 2, 3])
        test(["a", "b", "c"])

    def test_combinations(self):
        test = by_iter_eq(Set, "combinations")

        test([set()], [], 0)
        test([set()], [1], 0)
        test([set()], [1, 2], 0)
        test([set()], [1, 2, 3], 0)
        test([], [], 1)
        test([{1}], [1], 1)
        test([{1}, {2}], [1, 2], 1)
        test([{1}, {2}, {3}], [1, 2, 3], 1)
        test([], [], 2)
        test([], [1], 2)
        test([{1, 2}], [1, 2], 2)
        test([{1, 2}, {1, 3}, {2, 3}], [1, 2, 3], 2)

    def test_combine_if(self):
        test = by_iter_eq(Set, "combine_if")

        test(set(), [], True, "map", double)
        test(set(), [], False, "map", double)
        test({2}, [1], True, "map", double)
        test({1}, [1], False, "map", double)
        test({2, 4}, [1, 2], True, "map", double)
        test({1, 2}, [1, 2], False, "map", double)
        test({2, 4, 6}, [1, 2, 3], True, "map", double)
        test({1, 2, 3}, [1, 2, 3], False, "map", double)

    def test_copy(self):
        def test(items):
            s = Set(items)
            s2 = s.copy()
            assert s is not s2
            assert s == s2

        test([])
        test([1])
        test([1, 2])
        test([2, 1])
        test([1, 2, 3])

    def test_count(self):
        test = by_eq(Set, "count")

        test({1: 2, 2: 1}, ["a", "ab", "b"], len)
        test({True: 2, False: 1}, [1, 2, 0], bool)

    def test_cycle(self):
        def test(expected, items):
            actual = Set(items).cycle().take(5).list()
            assert actual == expected

        test([], [])
        test([1, 1, 1, 1, 1], [1])
        test([1, 2, 1, 2, 1], [1, 2])
        test([1, 2, 3, 1, 2], [1, 2, 3])

    def test_default_dict(self):
        def test(items, *a, **kw):
            result = Set(items).default_dict(*a, **kw)
            assert isinstance(result, DefaultDict)
            assert dict(result) == dict(items)

        a, b, c = enumerate("abc")

        test([], int)
        test([a], int)
        test([a, b], int)
        test([], list)
        test([("a", (1, 2))], list)
        test([("a", (1, 2)), ("b", (3, 4))], list)
        test([], str)
        test([("a", "hello")], str)
        test([("a", "hello"), ("b", "world")], str)

        result = Set([("a", 1)]).default_dict(int)
        assert result["a"] == 1
        assert result["b"] == 0

        result = Set().default_dict(list)
        result["new_list"].append(1)
        assert result["new_list"] == [1]

    def test_dict(self):
        test = by_eq(Set, "dict")

        a, b, c = enumerate("abc")

        test({}, [])
        test({0: "a"}, [a])
        test({0: "a", 1: "b"}, [a, b])
        test({0: 1, 1: 2, 2: 3}, enumerate([1, 2, 3]))

    def test_discard(self):
        def test(expected_result, expected_set, items, item):
            s = Set(items)
            result = s.discard(item)
            assert result == expected_result
            assert s == expected_set

        test(False, set(), [], 1)
        test(True, set(), [1], 1)
        test(True, {2}, [1, 2], 1)
        test(True, {1, 3}, [1, 2, 3], 2)
        test(False, {1, 2, 3}, [1, 2, 3], 4)

    def test_distinct(self):
        def test(expect_one_of, items, *a, **kw):
            actual = Set(items).distinct(*a, **kw)
            assert actual in expect_one_of

        test(({1, 2}, {1, 4}, {3, 2}, {3, 4}), [1, 2, 3, 4], key=lambda x: x % 2)
        test(({"a", "ab"}, {"b", "ab"}), ["a", "ab", "b"], len)

    def test_do(self):
        def test(items):
            actual = set()

            def f(item):
                actual.add(item)

            result = Set(items).do(f)
            assert result == items
            assert actual == items

        test(set())
        test({1})
        test({1, 2, 3})
        test({"a", "b"})

    def test_drop(self):
        test = by_eq(Set, "drop")

        test(set(), [], 0)
        test(set(), [], 1)
        test(set(), [], 5)
        test({1}, [1], 0)
        test(set(), [1], 1)
        test(set(), [1], 5)
        test({1, 2, 3}, [1, 2, 3], 0)
        test({2, 3}, [1, 2, 3], 1)
        test({3}, [1, 2, 3], 2)
        test(set(), [1, 2, 3], 3)
        test(set(), [1, 2, 3], 5)

    def test_drop_while(self):
        test = by_eq(Set, "drop_while")

        test(set(), [], lambda x: x < 5)
        test(set(), [1], lambda x: x < 5)
        test({5}, [5], lambda x: x < 5)
        test({5, 6}, [1, 2, 5, 6], lambda x: x < 5)
        test({1, 2, 3}, [1, 2, 3], lambda x: x > 5)

    def test_enumerate(self):
        test = by_eq(Set, "enumerate")

        test(set(), [])
        test({(0, "a")}, ["a"])
        test({(0, "a"), (1, "b")}, ["a", "b"])
        test({(0, 1), (1, 2), (2, 3)}, [1, 2, 3])

        test(set(), [], start=5)
        test({(5, "a")}, ["a"], start=5)
        test({(5, "a"), (6, "b")}, ["a", "b"], start=5)

    def test_filter(self):
        test = by_eq(Set, "filter")

        test(set(), [], lambda x: x > 0)
        test(set(), [1, 2, 3], lambda x: x > 5)
        test({1, 2, 3}, [1, 2, 3], lambda x: x > 0)
        test({2, 3}, [1, 2, 3], lambda x: x > 1)
        test({3}, [1, 2, 3], lambda x: x > 2)
        test({1, 3}, [1, 2, 3], is_odd)
        test({2}, [1, 2, 3], is_even)

    def test_first(self):
        test = by_eq(Set, "first")

        with should_raise(StopIteration):
            Set([]).first()
        test(1, [1])
        test(1, [1, 2, 3])
        test("a", ["a", "b", "c"])

        with should_raise(StopIteration):
            Set([]).first(predicate=lambda x: x > 0)
        with should_raise(StopIteration):
            Set([1, 2, 3]).first(predicate=lambda x: x > 5)
        test(2, [1, 2, 3], predicate=lambda x: x > 1)
        test(1, [1, 2, 3], predicate=is_odd)

        test(None, [], default=None)
        test(None, [1, 2, 3], predicate=lambda x: x > 5, default=None)

    def test_flat_map(self):
        test = by_eq(Set, "flat_map")

        test(set(), [], lambda x: [x])
        test({1}, [1], lambda x: [x])
        test({1, 2}, [1], lambda x: [x, x + 1])
        test({1, 2}, [1, 2], lambda x: [x])
        test({1, 2, 3}, [1, 2], lambda x: [x, x + 1])
        test(set(), [1, 2], lambda x: [])

    def test_flatten(self):
        test = by_eq(Set, "flatten")

        test(set(), [])
        test({1}, [(1,)])
        test({1, 2}, [(1, 2)])
        test({1}, [(1,), ()])
        test({1, 2, 3}, [(1, 2), (3,)])
        test({1, 2, 3}, [(1,), (2,), (3,)])
        test(set(), [(), ()])

    def test_fold(self):
        test = by_eq(Set, "fold")

        test(0, [], 0, add)
        test(1, [], 1, add)
        test(1, [1], 0, add)
        test(3, [1, 2], 0, add)
        test(6, [1, 2, 3], 0, add)
        test(1, [], 1, mul)
        test(2, [2], 1, mul)
        test(6, [2, 3], 1, mul)

    def test_fold_while(self):
        test = by_eq(Set, "fold_while")

        test(0, [], 0, add, lambda acc: acc < 10)
        test(1, [], 1, add, lambda acc: acc < 10)
        test(0, [1], 0, add, lambda acc: acc < 1)
        test(1, [1, 2], 0, add, lambda acc: acc < 2)
        test(6, [1, 2, 3], 0, add, lambda acc: acc < 10)
        test(3, [1, 2, 3], 0, add, lambda acc: acc < 5)

    def test_for_each(self):
        def test(items):
            actual = set()

            def f(item):
                actual.add(item)

            Set(items).for_each(f)
            assert actual == items

        test(set())
        test({1})
        test({1, 2, 3})
        test({"a", "b", "c"})

    def test_group_by(self):
        test = by_eq(Set, "group_by")

        test({True: [1, 3], False: [2]}, [1, 2, 3], is_odd)
        test({1: ["a"], 2: ["ab"]}, ["a", "ab"], len)

    def test_iter(self):
        def test(items):
            result = Set(items).iter()
            assert isinstance(result, Iter)
            assert set(result) == items

        test(set())
        test({1})
        test({1, 2, 3})
        test({"a", "b"})

    def test_last(self):
        test = by_eq(Set, "last")

        with should_raise(StopIteration):
            Set([]).last()
        test(1, [1])
        test(3, [1, 2, 3])
        test("c", ["a", "b", "c"])

        with should_raise(StopIteration):
            Set([]).last(predicate=lambda x: x > 0)
        with should_raise(StopIteration):
            Set([1, 2, 3]).last(predicate=lambda x: x > 5)
        test(3, [1, 2, 3], predicate=lambda x: x > 1)
        test(3, [1, 2, 3], predicate=is_odd)

        test(None, [], default=None)
        test(None, [1, 2, 3], predicate=lambda x: x > 5, default=None)

    def test_list(self):
        def test(items):
            s = Set(items)
            actual = s.list()
            assert isinstance(actual, list)
            assert isinstance(actual, List)
            assert set(actual) == items

        test(set())
        test({1})
        test({1, 2, 3})
        test({"a", "b"})

    def test_map(self):
        test = by_eq(Set, "map")

        test(set(), [], double)
        test({2}, [1], double)
        test({2, 4}, [1, 2], double)
        test({2, 4, 6}, [1, 2, 3], double)
        test({"A"}, ["a"], str.upper)
        test({1, 4, 9}, [1, 2, 3], lambda x: x * x)

    def test_map_to_keys(self):
        test = by_eq(Set, "map_to_keys")

        test({}, [], double)
        test({2: 1}, [1], double)
        test({2: 1}, [1, 1], double)
        test({2: 1, 4: 2}, [1, 2], double)
        test({4: 2, 2: 1}, [2, 1], double)

    def test_map_to_pairs(self):
        test = by_iter_eq(Set, "map_to_pairs")

        test([], [], double)
        test([(1, 2)], [1], double)
        test([(1, 2), (2, 4)], [1, 2], double)

    def test_map_to_values(self):
        test = by_eq(Set, "map_to_values")

        test({}, [], double)
        test({1: 2}, [1], double)
        test({1: 2}, [1, 1], double)
        test({1: 2, 2: 4}, [1, 2], double)
        test({2: 4, 1: 2}, [2, 1], double)

    def test_max(self):
        test = by_eq(Set, "max")

        with should_raise(ValueError):
            Set([]).max()
        test(1, [1])
        test(3, [1, 2, 3])
        test(3, [3, 1, 2])
        test("c", ["a", "b", "c"])

        test("ccc", ["a", "bb", "ccc"], key=len)

    def test_min(self):
        test = by_eq(Set, "min")

        with should_raise(ValueError):
            Set([]).min()
        test(1, [1])
        test(1, [1, 2, 3])
        test(1, [3, 1, 2])
        test("a", ["a", "b", "c"])

        test("a", ["a", "bb", "ccc"], key=len)

    def test_min_max(self):
        test = by_eq(Set, "min_max")

        with should_raise(ValueError):
            Set([]).min_max()
        test((1, 1), [1])
        test((1, 3), [1, 2, 3])
        test((1, 3), [3, 1, 2])
        test(("a", "c"), ["a", "b", "c"])

        test(("a", "ccc"), ["a", "bb", "ccc"], key=len)

    def test_only(self):
        test = by_eq(Set, "only")

        with should_raise(ValueError):
            Set([]).only()
        test(1, [1])
        with should_raise(ValueError):
            Set([1, 2]).only()
        with should_raise(ValueError):
            Set([1, 2, 3]).only()

        with should_raise(ValueError):
            Set([]).only(predicate=is_odd)
        test(1, [1], predicate=is_odd)
        test(1, [1, 2], predicate=is_odd)
        with should_raise(ValueError):
            Set([1, 3]).only(predicate=is_odd)

    def test_partition(self):
        def test(expected_true, expected_false, items, predicate):
            actual_true, actual_false = Set(items).partition(predicate)
            assert iter_eq(actual_true, expected_true)
            assert iter_eq(actual_false, expected_false)

        test([], [], [], is_odd)
        test([1], [], [1], is_odd)
        test([], [2], [2], is_odd)
        test([1], [2], [1, 2], is_odd)
        test([1, 3], [2], [1, 2, 3], is_odd)
        test([1, 3, 5], [2, 4], [1, 2, 3, 4, 5], is_odd)

    def test_permutations(self):
        test = by_iter_eq(Set, "permutations")

        test([()], [], 0)
        test([()], [1], 0)
        test([()], [1, 2], 0)
        test([], [], 1)
        test([(1,)], [1], 1)
        test([(1,), (2,)], [1, 2], 1)
        test([], [], 2)
        test([], [1], 2)
        test([(1, 2), (2, 1)], [1, 2], 2)
        test([(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)], [1, 2, 3], 2)

    def test_powerset(self):
        test = by_iter_eq(Set, "powerset")

        test([()], [])
        test([(), (1,)], [1])
        test([(), (1,), (2,), (1, 2)], [1, 2])
        test([(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)], [1, 2, 3])

    def test_product(self):
        test = by_iter_eq(Set, "product")

        test([], range(0), [])
        test([], range(0), [1, 2])
        test([], range(1), [])
        test([(0, 1), (0, 2)], range(1), [1, 2])
        test([(0, 1), (0, 2), (1, 1), (1, 2)], range(2), [1, 2])
        test([(0, 1), (1, 1)], range(2), [1])
        test([(0, 0), (0, 1), (1, 0), (1, 1)], range(2), range(2))
        test([(0, 0)], range(1), repeat=2)
        test([(0, 0), (0, 1), (1, 0), (1, 1)], range(2), repeat=2)

    def test_reduce(self):
        test = by_eq(Set, "reduce")

        with should_raise(ValueError):
            Set([]).reduce(add)
        test(1, [1], add)
        test(3, [1, 2], add)
        test(6, [1, 2, 3], add)
        test(1, [1], mul)
        test(2, [1, 2], mul)
        test(6, [1, 2, 3], mul)

    def test_repeat(self):
        def test(expected, items, n):
            actual = Set(items).repeat(n).take(len(expected)).list()
            assert actual == expected

        test([], [], 3)
        test([], [1], 0)
        test([1, 1, 1], [1], 3)
        test([1, 2, 1, 2, 1], [1, 2], 3)

    def test_set(self):
        def test(items):
            s = Set(items)
            actual = s.set()
            assert actual is s

        test([])
        test([1])
        test([1, 2, 3])
        test(["a", "b"])

    def test_size(self):
        test = by_eq(Set, "size")

        test(0, [])
        test(1, [1])
        test(3, [1, 2, 3])
        test(5, ["a", "b", "c", "d", "e"])

    def test_sliding(self):
        test = by_iter_eq(Set, "sliding")

        test([], [], 2)
        test([], [1], 2)
        test([{1, 2}], [1, 2], 2)
        test([{1, 2}, {2, 3}], [1, 2, 3], 2)
        test([{1, 2, 3}], [1, 2, 3], 3)
        test([{1, 2, 3}, {2, 3, 4}], [1, 2, 3, 4], 3)
        test([{1, 2, 3}, {2, 3, 4}, {3, 4, 5}], [1, 2, 3, 4, 5], 3)

    def test_sliding_by_timestamp(self):
        now = 0

        def clock():
            nonlocal now
            now += 1
            return now

        def test(expected, items, window_size):
            nonlocal now
            now = 0
            stamp = timestamp(clock)
            result = Set(items).sliding_by_timestamp(window_size, stamp=stamp)
            actual = [set(window) for window in result]
            assert actual == expected

        test([], [], 2)
        test([{1}], [1], 2)
        test([{1, 2}], [1, 2], 3)
        test([{1, 2}, {2, 3}], [1, 2, 3], 2)

    def test_take(self):
        test = by_iter_eq(Set, "take")

        test([], [], 0)
        test([], [], 3)
        test([], [1, 2, 3], 0)
        test([1], [1, 2, 3], 1)
        test([1, 2], [1, 2, 3], 2)
        test([1, 2, 3], [1, 2, 3], 3)
        test([1, 2, 3], [1, 2, 3], 5)

    def test_take_while(self):
        test = by_iter_eq(Set, "take_while")

        test([], [], lambda x: x < 5)
        test([], [5, 6, 7], lambda x: x < 5)
        test([1, 2], [1, 2, 5, 6], lambda x: x < 5)
        test([1, 2, 3], [1, 2, 3], lambda x: x < 5)

    def test_timestamp(self):
        now = 0

        def clock():
            nonlocal now
            now += 1
            return now

        def test(expected, items):
            nonlocal now
            now = 0
            actual = Set(items).timestamp(clock=clock)
            assert iter_eq(actual, expected)

        test([], [])
        test([(1, "a")], ["a"])
        test([(1, "a"), (2, "b")], ["a", "b"])
        test([(1, 1), (2, 2), (3, 3)], [1, 2, 3])

    def test_tqdm(self):
        # tqdm shouldn't change the input, and shouldn't raise any exceptions
        def test(items):
            actual = Set(items).tqdm()
            assert iter_eq(actual, items)

        test([])
        test([1])
        test([1, 2, 3])
        test(["a", "b", "c"])

    def test_transpose(self):
        test = by_iter_eq(Set, "transpose")

        test([], [])
        test([(1,), (2,), (3,)], [(1, 2, 3)])
        test([(1, 4), (2, 5), (3, 6)], [(1, 2, 3), (4, 5, 6)])
        test([(1, 4, 7), (2, 5, 8), (3, 6, 9)], [(1, 2, 3), (4, 5, 6), (7, 8, 9)])

    def test_tuple(self):
        def test(items):
            actual = Set(items).tuple()
            assert isinstance(actual, tuple)
            assert isinstance(actual, Tuple)
            assert iter_eq(actual, items)

        test([])
        test([1])
        test([1, 2, 3])
        test(["a", "b"])

    def test_unzip(self):
        pass  # direct alias of transpose

    def test_update(self):
        def test(expected, items, *a, **kw):
            s = Set(items)
            actual = s.update(*a, **kw)
            assert actual is s
            assert actual == expected

        test(set(), [], [])
        test({1}, {1}, [])
        test({1}, [], {1})
        test({1, 2, 3}, {1, 2}, {2, 3})

    def test_zip(self):
        test = by_eq(Set, "zip")

        test(set(), [], [])
        test(set(), [1], [])
        test(set(), [], [1])
        test({(1, "a")}, [1], ["a"])
        test({(1, "a"), (2, "b")}, [1, 2], ["a", "b"])
        test({(1, "a")}, [1, 2], ["a"])
        test({(1, "a")}, [1], ["a", "b"])

    def test_zip_longest(self):
        test = by_eq(Set, "zip_longest")

        test(set(), [], [])
        test({(1, None)}, [1], [])
        test({(None, "a")}, [], ["a"])
        test({(1, "a")}, [1], ["a"])
        test({(1, "a"), (2, "b")}, [1, 2], ["a", "b"])
        test({(1, "a"), (2, None)}, [1, 2], ["a"])
        test({(1, "a"), (None, "b")}, [1], ["a", "b"])


class TestTuple:
    def test___add__(self):
        def test(expected, left, right):
            actual = Tuple(left) + right
            assert iter_eq(actual, expected)
            assert isinstance(actual, Tuple)

        test([1, 2, 3, 4], [1, 2], [3, 4])
        test([], [], [])
        test([1, 2], [1, 2], [])
        test([1, 2], [], [1, 2])
        test([1, 2, 3, 4], [1, 2], Tuple([3, 4]))

    def test___getitem__(self):
        def test_index(expected, items, index):
            result = Tuple(items)[index]
            assert result == expected

        def test_slice(expected, items, slice_obj):
            actual = Tuple(items)[slice_obj]
            assert iter_eq(actual, expected)
            assert isinstance(actual, Tuple)

        test_index(1, [1, 2, 3, 4, 5], 0)
        test_index(2, [1, 2, 3, 4, 5], 1)
        test_index(5, [1, 2, 3, 4, 5], -1)
        test_index(4, [1, 2, 3, 4, 5], -2)

        test_slice([2, 3, 4], [1, 2, 3, 4, 5], slice(1, 4))
        test_slice([1, 2, 3], [1, 2, 3, 4, 5], slice(None, 3))
        test_slice([3, 4, 5], [1, 2, 3, 4, 5], slice(2, None))
        test_slice([1, 3, 5], [1, 2, 3, 4, 5], slice(None, None, 2))
        test_slice([], [1, 2, 3, 4, 5], slice(10, 20))

    def test___iter__(self):
        def test(expected, items):
            iterator = iter(Tuple(items))
            assert isinstance(iterator, Iter)
            assert list(iterator) == expected

        test([1, 2, 3], [1, 2, 3])
        test([], [])

        items = []
        for item in Tuple([1, 2, 3]):
            items.append(item)
        assert items == [1, 2, 3]

    def test___mul__(self):
        def test(expected, items, n):
            actual = Tuple(items) * n
            assert iter_eq(actual, expected)
            assert isinstance(actual, Tuple)

        test([1, 2, 1, 2, 1, 2], [1, 2], 3)
        test([], [], 5)
        test(["a", "a", "a", "a"], ["a"], 4)
        test([], [1, 2, 3], 0)
        test([], [1, 2], -1)

    def test___rmul__(self):
        def test(expected, n, items):
            actual = n * Tuple(items)
            assert iter_eq(actual, expected)
            assert isinstance(actual, Tuple)

        test([1, 2, 1, 2, 1, 2], 3, [1, 2])
        test([], 5, [])
        test(["a", "a", "a", "a"], 4, ["a"])
        test([], 0, [1, 2, 3])
        test([], -1, [1, 2])

    def test_all(self):
        test = by_eq(Tuple, "all")

        test(True, [])
        test(True, [1, 2, 3, True])
        test(False, [0, False, None, ""])
        test(False, [1, 2, 0, 3])
        test(True, [True, True, True])
        test(False, [True, False, True])
        test(True, [1, 3, 5, 7], key=is_odd)
        test(False, [1, 2, 3, 5], key=is_odd)

    def test_any(self):
        test = by_eq(Tuple, "any")

        test(False, [])
        test(True, [1, 2, 3, True])
        test(False, [0, False, None, ""])
        test(True, [0, False, 1, None])
        test(False, [False, False, False])
        test(True, [False, True, False])
        test(False, [2, 4, 6, 8], key=is_odd)
        test(True, [2, 3, 4, 6], key=is_odd)

    def test_apply(self):
        test = by_eq(Tuple, "apply")

        test(False, [], bool)
        test(True, [1], bool)
        test(3, [1, 2, 3], len)
        test(0, [], len)
        test(6, [1, 2, 3], sum)

    def test_apply_and_wrap(self):
        test = by_iter_eq(Tuple, "apply_and_wrap")

        test([], [], identity)
        test([1], [1], identity)
        test([1, 2, 3], [1, 2, 3], identity)
        test([3, 2, 1], [1, 2, 3], reversed)

    def test_batch(self):
        test = by_iter_eq(Tuple, "batch")

        with should_raise(RequirementException):
            Tuple().batch(0)

        test([], [], 1)
        test([(1,)], [1], 1)
        test([(1,), (2,)], [1, 2], 1)
        test([(1, 2)], [1, 2], 2)
        test([(1, 2), (3,)], [1, 2, 3], 2)
        test([(1, 2, 3)], [1, 2, 3], 3)
        test([(1, 2, 3)], [1, 2, 3], 4)

    def test_chain(self):
        test = by_iter_eq(Tuple, "chain")

        test([], [])
        test([], [], [])
        test([], [], [], [])
        test([1], [1], [])
        test([1, 2], [1, 2], [])
        test([1, 2, 3], [1, 2], [3])
        test([1, 2, 3, 4], [1, 2], [3, 4])
        test([1, 2, 3, 4, 5], [1], [2, 3], [4, 5])

    def test_combinations(self):
        test = by_iter_eq(Tuple, "combinations")

        test([()], [], 0)
        test([()], [1], 0)
        test([()], [1, 2], 0)
        test([()], [1, 2, 3], 0)
        test([], [], 1)
        test([(1,)], [1], 1)
        test([(1,), (2,)], [1, 2], 1)
        test([(1,), (2,), (3,)], [1, 2, 3], 1)
        test([], [], 2)
        test([], [1], 2)
        test([(1, 2)], [1, 2], 2)
        test([(1, 2), (1, 3), (2, 3)], [1, 2, 3], 2)

        test([()], [], 0, with_replacement=True)
        test([()], [1], 0, with_replacement=True)
        test([], [], 1, with_replacement=True)
        test([(1,)], [1], 1, with_replacement=True)
        test([(1,), (2,)], [1, 2], 1, with_replacement=True)
        test([], [], 2, with_replacement=True)
        test([(1, 1)], [1], 2, with_replacement=True)
        test([(1, 1), (1, 2), (2, 2)], [1, 2], 2, with_replacement=True)

    def test_combine_if(self):
        test = by_iter_eq(Tuple, "combine_if")

        test([], [], True, "map", double)
        test([], [], False, "map", double)
        test([2], [1], True, "map", double)
        test([1], [1], False, "map", double)
        test([2, 4], [1, 2], True, "map", double)
        test([1, 2], [1, 2], False, "map", double)
        test([2, 4, 6], [1, 2, 3], True, "map", double)
        test([1, 2, 3], [1, 2, 3], False, "map", double)

    def test_count(self):
        test = by_eq(Tuple, "count")

        test({}, [])
        test({1: 1}, [1])
        test({1: 2}, [1, 1])
        test({1: 1, 2: 1}, [1, 2])
        test({1: 2, 2: 1}, [1, 1, 2])
        test({"a": 1, "b": 2}, ["a", "b", "b"])
        test({"a": 2, "b": 1}, ["a", "b", "a"])

        test({1: 2, 2: 1}, ["a", "ab", "b"], len)
        test({True: 2, False: 1}, [1, 2, 0], bool)

    def test_cycle(self):
        def test(expected, items):
            actual = Tuple(items).cycle().take(5).list()
            assert actual == expected

        test([], [])
        test([1, 1, 1, 1, 1], [1])
        test([1, 2, 1, 2, 1], [1, 2])
        test([1, 2, 3, 1, 2], [1, 2, 3])

    def test_default_dict(self):
        def test(items, *a, **kw):
            result = Tuple(items).default_dict(*a, **kw)
            assert isinstance(result, DefaultDict)
            assert dict(result) == dict(items)

        a, b, c = enumerate("abc")

        test([], int)
        test([a], int)
        test([a, b], int)
        test([], list)
        test([("a", [1, 2])], list)
        test([("a", [1, 2]), ("b", [3, 4])], list)
        test([], str)
        test([("a", "hello")], str)
        test([("a", "hello"), ("b", "world")], str)

        result = Tuple([("a", 1)]).default_dict(int)
        assert result["a"] == 1
        assert result["b"] == 0

        result = Tuple().default_dict(list)
        result["new_list"].append(1)
        assert result["new_list"] == [1]

    def test_dict(self):
        test = by_eq(Tuple, "dict")

        a, b, c = enumerate("abc")

        test({}, [])
        test({0: "a"}, [a])
        test({0: "a", 1: "b"}, [a, b])
        test({0: 1, 1: 2, 2: 3}, enumerate([1, 2, 3]))

    def test_distinct(self):
        test = by_iter_eq(Tuple, "distinct")

        test([], [])
        test([1], [1])
        test([1], [1, 1])
        test([1, 2], [1, 2])
        test([1, 2], [1, 1, 2])
        test([1, 2, 3], [1, 2, 3, 2, 1])
        test(["a"], ["a", "a"])
        test(["a", "b"], ["a", "b", "a"])

    def test_do(self):
        def test(items):
            actual = []

            def f(item):
                actual.append(item)

            result = Tuple(items).do(f)
            assert iter_eq(result, items)
            assert actual == items

        test([])
        test([1])
        test([1, 2, 3])
        test(["a", "b"])

    def test_drop(self):
        test = by_iter_eq(Tuple, "drop")

        test([], [], 0)
        test([], [], 1)
        test([], [], 5)
        test([1], [1], 0)
        test([], [1], 1)
        test([], [1], 5)
        test([1, 2, 3], [1, 2, 3], 0)
        test([2, 3], [1, 2, 3], 1)
        test([3], [1, 2, 3], 2)
        test([], [1, 2, 3], 3)
        test([], [1, 2, 3], 5)

    def test_drop_while(self):
        test = by_iter_eq(Tuple, "drop_while")

        test([], [], lambda x: x < 5)
        test([], [1], lambda x: x < 5)
        test([5], [5], lambda x: x < 5)
        test([5, 6], [1, 2, 5, 6], lambda x: x < 5)
        test([5, 1, 6], [1, 2, 5, 1, 6], lambda x: x < 5)
        test([1, 2, 3], [1, 2, 3], lambda x: x > 5)

    def test_enumerate(self):
        test = by_iter_eq(Tuple, "enumerate")

        test([], [])
        test([(0, "a")], ["a"])
        test([(0, "a"), (1, "b")], ["a", "b"])
        test([(0, 1), (1, 2), (2, 3)], [1, 2, 3])

        test([], [], start=5)
        test([(5, "a")], ["a"], start=5)
        test([(5, "a"), (6, "b")], ["a", "b"], start=5)

    def test_filter(self):
        test = by_iter_eq(Tuple, "filter")

        test([], [], lambda x: x > 0)
        test([], [1, 2, 3], lambda x: x > 5)
        test([1, 2, 3], [1, 2, 3], lambda x: x > 0)
        test([2, 3], [1, 2, 3], lambda x: x > 1)
        test([3], [1, 2, 3], lambda x: x > 2)
        test([1, 3], [1, 2, 3], is_odd)
        test([2], [1, 2, 3], is_even)

    def test_first(self):
        test = by_eq(Tuple, "first")

        with should_raise(StopIteration):
            Tuple([]).first()
        test(1, [1])
        test(1, [1, 2, 3])
        test("a", ["a", "b", "c"])

        with should_raise(StopIteration):
            Tuple([]).first(predicate=lambda x: x > 0)
        with should_raise(StopIteration):
            Tuple([1, 2, 3]).first(predicate=lambda x: x > 5)
        test(2, [1, 2, 3], predicate=lambda x: x > 1)
        test(1, [1, 2, 3], predicate=is_odd)

        test(None, [], default=None)
        test(None, [1, 2, 3], predicate=lambda x: x > 5, default=None)

    def test_flat_map(self):
        test = by_iter_eq(Tuple, "flat_map")

        test([], [], lambda x: [x])
        test([1], [1], lambda x: [x])
        test([1, 1], [1], lambda x: [x, x])
        test([1, 2], [1, 2], lambda x: [x])
        test([1, 1, 2, 2], [1, 2], lambda x: [x, x])
        test([], [1, 2], lambda x: [])
        test([0, 0, 1], [2, 3], lambda x: list(range(x - 1)))

    def test_flatten(self):
        test = by_iter_eq(Tuple, "flatten")

        test([], [])
        test([1], [[1]])
        test([1, 2], [[1, 2]])
        test([1], [[1], []])
        test([1, 2, 3], [[1, 2], [3]])
        test([1, 2, 3], [[1], [2], [3]])
        test([], [[], []])

    def test_fold(self):
        test = by_eq(Tuple, "fold")

        test(0, [], 0, add)
        test(1, [], 1, add)
        test(1, [1], 0, add)
        test(3, [1, 2], 0, add)
        test(6, [1, 2, 3], 0, add)
        test(1, [], 1, mul)
        test(2, [2], 1, mul)
        test(6, [2, 3], 1, mul)

    def test_fold_while(self):
        test = by_eq(Tuple, "fold_while")

        test(0, [], 0, add, lambda acc: acc < 10)
        test(1, [], 1, add, lambda acc: acc < 10)
        test(0, [1], 0, add, lambda acc: acc < 1)
        test(1, [1, 2], 0, add, lambda acc: acc < 2)
        test(6, [1, 2, 3], 0, add, lambda acc: acc < 10)
        test(3, [1, 2, 3], 0, add, lambda acc: acc < 5)

    def test_for_each(self):
        def test(items):
            actual = []

            def f(item):
                actual.append(item)

            Tuple(items).for_each(f)
            assert actual == items

        test([])
        test([1])
        test([1, 2, 3])
        test(["a", "b", "c"])

    def test_group_by(self):
        test = by_eq(Tuple, "group_by")

        test({}, [])
        test({1: [1]}, [1])
        test({1: [1, 1]}, [1, 1])
        test({1: [1], 2: [2]}, [1, 2])
        test({1: [1, 1], 2: [2]}, [1, 1, 2])
        test({True: [1, 3], False: [2]}, [1, 2, 3], is_odd)
        test({1: ["a"], 2: ["ab"]}, ["a", "ab"], len)

    def test_intersperse(self):
        test = by_iter_eq(Tuple, "intersperse")

        test([], [], 0)
        test([1], [1], 0)
        test([1, 0, 2], [1, 2], 0)
        test([1, 0, 2, 0, 3], [1, 2, 3], 0)
        test(["a", ',', "b"], ["a", "b"], ',')

    def test_iter(self):
        def test(items):
            result = Tuple(items).iter()
            assert isinstance(result, Iter)
            assert list(result) == items

        test([])
        test([1])
        test([1, 2, 3])
        test(["a", "b"])

    def test_last(self):
        test = by_eq(Tuple, "last")

        with should_raise(StopIteration):
            Tuple([]).last()
        test(1, [1])
        test(3, [1, 2, 3])
        test("c", ["a", "b", "c"])

        with should_raise(StopIteration):
            Tuple([]).last(predicate=lambda x: x > 0)
        with should_raise(StopIteration):
            Tuple([1, 2, 3]).last(predicate=lambda x: x > 5)
        test(3, [1, 2, 3], predicate=lambda x: x > 1)
        test(3, [1, 2, 3], predicate=is_odd)

        test(None, [], default=None)
        test(None, [1, 2, 3], predicate=lambda x: x > 5, default=None)

    def test_list(self):
        def test(items):
            actual = Tuple(items).list()
            assert isinstance(actual, list)
            assert isinstance(actual, List)
            assert iter_eq(actual, items)

        test([])
        test([1])
        test([1, 2, 3])
        test(["a", "b"])

    def test_map(self):
        test = by_iter_eq(Tuple, "map")

        test([], [], double)
        test([2], [1], double)
        test([2, 4], [1, 2], double)
        test([2, 4, 6], [1, 2, 3], double)
        test(["A"], ["a"], str.upper)
        test([1, 4, 9], [1, 2, 3], lambda x: x * x)

    def test_map_to_keys(self):
        test = by_eq(Tuple, "map_to_keys")

        test({}, [], double)
        test({2: 1}, [1], double)
        test({2: 1}, [1, 1], double)
        test({2: 1, 4: 2}, [1, 2], double)
        test({4: 2, 2: 1}, [2, 1], double)

    def test_map_to_pairs(self):
        test = by_iter_eq(Tuple, "map_to_pairs")

        test([], [], double)
        test([(1, 2)], [1], double)
        test([(1, 2), (1, 2)], [1, 1], double)
        test([(1, 2), (2, 4)], [1, 2], double)
        test([(2, 4), (1, 2)], [2, 1], double)

    def test_map_to_values(self):
        test = by_eq(Tuple, "map_to_values")

        test({}, [], double)
        test({1: 2}, [1], double)
        test({1: 2}, [1, 1], double)
        test({1: 2, 2: 4}, [1, 2], double)
        test({2: 4, 1: 2}, [2, 1], double)

    def test_max(self):
        test = by_eq(Tuple, "max")

        with should_raise(ValueError):
            Tuple([]).max()
        test(1, [1])
        test(3, [1, 2, 3])
        test(3, [3, 1, 2])
        test("c", ["a", "b", "c"])

        test("ccc", ["a", "bb", "ccc"], key=len)

    def test_min(self):
        test = by_eq(Tuple, "min")

        with should_raise(ValueError):
            Tuple([]).min()
        test(1, [1])
        test(1, [1, 2, 3])
        test(1, [3, 1, 2])
        test("a", ["a", "b", "c"])

        test("a", ["a", "bb", "ccc"], key=len)

    def test_min_max(self):
        test = by_eq(Tuple, "min_max")

        with should_raise(ValueError):
            Tuple([]).min_max()
        test((1, 1), [1])
        test((1, 3), [1, 2, 3])
        test((1, 3), [3, 1, 2])
        test(("a", "c"), ["a", "b", "c"])

        test(("a", "ccc"), ["a", "bb", "ccc"], key=len)

    def test_only(self):
        test = by_eq(Tuple, "only")

        with should_raise(ValueError):
            Tuple([]).only()
        test(1, [1])
        with should_raise(ValueError):
            Tuple([1, 2]).only()
        with should_raise(ValueError):
            Tuple([1, 2, 3]).only()

        with should_raise(ValueError):
            Tuple([]).only(predicate=is_odd)
        test(1, [1], predicate=is_odd)
        test(1, [1, 2], predicate=is_odd)
        with should_raise(ValueError):
            Tuple([1, 3]).only(predicate=is_odd)

    def test_partition(self):
        def test(expected_true, expected_false, items, predicate):
            true_list, false_list = Tuple(items).partition(predicate)
            assert list(true_list) == expected_true
            assert list(false_list) == expected_false

        test([], [], [], is_odd)
        test([1], [], [1], is_odd)
        test([], [2], [2], is_odd)
        test([1], [2], [1, 2], is_odd)
        test([1, 3], [2], [1, 2, 3], is_odd)
        test([1, 3, 5], [2, 4], [1, 2, 3, 4, 5], is_odd)

    def test_permutations(self):
        test = by_iter_eq(Tuple, "permutations")

        test([()], [], 0)
        test([()], [1], 0)
        test([()], [1, 2], 0)
        test([], [], 1)
        test([(1,)], [1], 1)
        test([(1,), (2,)], [1, 2], 1)
        test([], [], 2)
        test([], [1], 2)
        test([(1, 2), (2, 1)], [1, 2], 2)
        test([(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)], [1, 2, 3], 2)

    def test_powerset(self):
        test = by_iter_eq(Tuple, "powerset")

        test([()], [])
        test([(), (1,)], [1])
        test([(), (1,), (2,), (1, 2)], [1, 2])
        test([(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)], [1, 2, 3])

    def test_product(self):
        test = by_iter_eq(Tuple, "product")

        test([], [])
        test([], [], [])
        test([], [1], [])
        test([], [], [1])
        test([(1, "a")], [1], ["a"])
        test([(1, "a"), (1, "b")], [1], ["a", "b"])
        test([(1, "a"), (2, "a")], [1, 2], ["a"])
        test([(1, "a"), (1, "b"), (2, "a"), (2, "b")], [1, 2], ["a", "b"])

        test([], [], repeat=2)
        test([()], [1], repeat=0)
        test([(1,)], [1], repeat=1)
        test([(1, 1)], [1], repeat=2)
        test([(1, 1), (1, 2), (2, 1), (2, 2)], [1, 2], repeat=2)

    def test_reduce(self):
        test = by_eq(Tuple, "reduce")

        with should_raise(ValueError):
            Tuple([]).reduce(add)
        test(1, [1], add)
        test(3, [1, 2], add)
        test(6, [1, 2, 3], add)
        test(1, [1], mul)
        test(2, [1, 2], mul)
        test(6, [1, 2, 3], mul)

    def test_repeat(self):
        def test(expected, items, n):
            actual = Tuple(items).repeat(n).take(len(expected)).list()
            assert actual == expected

        test([], [], 3)
        test([], [1], 0)
        test([1, 1, 1], [1], 3)
        test([1, 2, 1, 2, 1], [1, 2], 3)

    def test_reversed(self):
        test = by_iter_eq(Tuple, "reversed")

        test([], [])
        test([3], [3])
        test([2, 1], [1, 2])
        test([3, 2, 1], [1, 2, 3])
        test(["c", "b", "a"], ["a", "b", "c"])
        test([5, 4, 3, 2, 1], [1, 2, 3, 4, 5])

    def test_set(self):
        def test(items):
            result = Tuple(items).set()
            assert isinstance(result, Set)
            assert sorted(list(result)) == sorted(list(set(items)))

        test([])
        test([1])
        test([1, 2, 3])
        test([1, 1, 2, 2, 3])
        test(["a", "b", "a"])

    def test_size(self):
        test = by_eq(Tuple, "size")

        test(0, [])
        test(1, [1])
        test(3, [1, 2, 3])
        test(5, ["a", "b", "c", "d", "e"])

    def test_sliding(self):
        test = by_iter_eq(Tuple, "sliding")

        test([], [], 2)
        test([], [1], 2)
        test([(1, 2)], [1, 2], 2)
        test([(1, 2), (2, 3)], [1, 2, 3], 2)
        test([(1, 2, 3)], [1, 2, 3], 3)
        test([(1, 2, 3), (2, 3, 4)], [1, 2, 3, 4], 3)
        test([(1, 2, 3), (2, 3, 4), (3, 4, 5)], [1, 2, 3, 4, 5], 3)

    def test_sliding_by_timestamp(self):
        now = 0

        def clock():
            nonlocal now
            now += 1
            return now

        def test(expected, items, window_size):
            nonlocal now
            now = 0
            stamp = timestamp(clock)
            result = Tuple(items).sliding_by_timestamp(window_size, stamp=stamp)
            actual = [list(window) for window in result]
            assert actual == expected

        test([], [], 2)
        test([[1]], [1], 2)
        test([[1, 2]], [1, 2], 3)
        test([[1, 2], [2, 3]], [1, 2, 3], 2)

    def test_sorted(self):
        test = by_iter_eq(Tuple, "sorted")

        test([], [])
        test([1], [1])
        test([1, 2, 3], [3, 1, 2])
        test([1, 2, 3], [1, 2, 3])
        test(["a", "b", "c"], ["c", "a", "b"])
        test([3, 2, 1], [1, 2, 3], reverse=True)
        test(["c", "b", "a"], ["a", "b", "c"], reverse=True)
        test(["a", "bb", "ccc"], ["bb", "ccc", "a"], key=len)
        test(["ccc", "bb", "a"], ["bb", "ccc", "a"], key=len, reverse=True)

    def test_take(self):
        test = by_iter_eq(Tuple, "take")

        test([], [], 0)
        test([], [], 3)
        test([], [1, 2, 3], 0)
        test([1], [1, 2, 3], 1)
        test([1, 2], [1, 2, 3], 2)
        test([1, 2, 3], [1, 2, 3], 3)
        test([1, 2, 3], [1, 2, 3], 5)

    def test_take_while(self):
        test = by_iter_eq(Tuple, "take_while")

        test([], [], lambda x: x < 5)
        test([], [5, 6, 7], lambda x: x < 5)
        test([1, 2], [1, 2, 5, 6], lambda x: x < 5)
        test([1, 2, 3], [1, 2, 3], lambda x: x < 5)
        test([1], [1, 5, 2], lambda x: x < 5)

    def test_timestamp(self):
        now = 0

        def clock():
            nonlocal now
            now += 1
            return now

        def test(expected, items):
            nonlocal now
            now = 0
            actual = Tuple(items).timestamp(clock=clock)
            assert iter_eq(actual, expected)

        test([], [])
        test([(1, "a")], ["a"])
        test([(1, "a"), (2, "b")], ["a", "b"])
        test([(1, 1), (2, 2), (3, 3)], [1, 2, 3])

    def test_tqdm(self):
        # tqdm shouldn't change the input, and shouldn't raise any exceptions
        def test(items):
            actual = Tuple(items).tqdm()
            assert iter_eq(actual, items)

        test([])
        test([1])
        test([1, 2, 3])
        test(["a", "b", "c"])

    def test_transpose(self):
        test = by_iter_eq(Tuple, "transpose")

        test([], [])
        test([(1,), (2,), (3,)], [[1, 2, 3]])
        test([(1, 4), (2, 5), (3, 6)], [[1, 2, 3], [4, 5, 6]])
        test([(1, 4, 7), (2, 5, 8), (3, 6, 9)], [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_tuple(self):
        def test(items):
            tup = Tuple(items)
            actual = tup.tuple()
            assert isinstance(actual, tuple)
            assert isinstance(actual, Tuple)
            assert actual == tup
            assert actual is tup

        test([])
        test([1])
        test([1, 2, 3])
        test(["a", "b", "c"])

    def test_unzip(self):
        pass  # direct alias of transpose, does not need further testing

    def test_zip(self):
        test = by_iter_eq(Tuple, "zip")

        test([], [])
        test([], [], [])
        test([], [1], [])
        test([], [], [1])
        test([("a", 1)], "a", [1])
        test([("a", 1), ("b", 2)], "ab", [1, 2])
        test([("a", 1)], "ab", [1])
        test([("a", 1)], "a", [1, 2])

    def test_zip_longest(self):
        test = by_iter_eq(Tuple, "zip_longest")

        test([], [])
        test([], [], [])
        test([(1,)], [1])
        test([(1, None)], [1], [])
        test([(1, 0)], [1], [], fillvalue=0)
        test([(None, 1)], [], [1])
        test([(0, 1)], [], [1], fillvalue=0)
        test([(1, 2)], [1], [2])
        test([(1, 3), (2, None)], [1, 2], [3])
        test([(1, 3), (2, 4)], [1, 2], [3, 4])
