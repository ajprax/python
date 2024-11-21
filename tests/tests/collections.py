from operator import add, mul

from ajprax.collections import Dict
from ajprax.collections import Iter
from ajprax.collections import count, repeated, repeatedly, timestamp
from ajprax.hof import identity
from ajprax.hof import t
from ajprax.require import RequirementException
from tests import by_iter_eq, by_eq, double, is_even, is_odd, less_than, should_raise, tower, greater_than
from tests import iter_eq
from tests import key_less_than, key_is_odd, double_key, double_value


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


class TestDefaultDict:
    pass


class TestDeque:
    pass


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

    def test_apply_and_iter(self):
        test = by_iter_eq(Dict, "apply_and_iter")

        test([], [], lambda d: d.keys())
        test(["a"], [("a", 1)], lambda d: d.keys())
        test([1], [("a", 1)], lambda d: d.values())

    def test_batch(self):
        test = by_iter_eq(Dict, "batch")

        for i in range(2, 5):
            test([], [], i)
            test([(("a", 1),)], [("a", 1)], i)
            test([(("a", 1), ("b", 2))], [("a", 1), ("b", 2)], i)

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

    def test_count_values(self):
        test = by_eq(Dict, "count_values")

        a, b, c = enumerate("abc")

        test({}, [])
        test({"a": 1}, [a])
        test({"a": 1, "b": 1}, [a, b])
        test({"a": 2}, [a, (1, "a")])
        test({"a": 2, "b": 1}, [a, (1, "a"), (2, "b")])

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

    def test_dict(self):
        def test(items):
            d1 = Dict(items)
            d2 = d1.dict()
            assert d1 == d2
            assert d1 is not d2

        a, b, c = enumerate("abc")

        test([])
        test([a])
        test([b])
        test([a, b])
        test([b, a])
        test([a, b, c])

    def test_distinct_values(self):
        test = by_iter_eq(Dict, "distinct_values")

        a, b, c = enumerate("abc")

        test([], [])
        test([a], [a])
        test([b], [b])
        test([a, b], [a, b])
        test([b, a], [b, a])
        test([a, b, c], [a, b, c])
        test([a], [a, a])
        test([a, b], [a, a, b])
        test([a, b], [a, b, a])
        test([], [], key=lambda c: ord(c) % 2)
        test([a], [a], key=lambda c: ord(c) % 2)
        test([b], [b], key=lambda c: ord(c) % 2)
        test([a, b], [a, b], key=lambda c: ord(c) % 2)
        test([b, a], [b, a], key=lambda c: ord(c) % 2)
        test([a, b], [a, b, c], key=lambda c: ord(c) % 2)
        test([a], [a, a], key=lambda c: ord(c) % 2)
        test([a, b], [a, a, b], key=lambda c: ord(c) % 2)
        test([a, b], [a, b, a], key=lambda c: ord(c) % 2)

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
        test = by_iter_eq(Dict, "drop")

        a, b, c = enumerate("abc")

        test([], [], 0)
        test([], [], 1)
        test([], [], 4)
        test([a], [a], 0)
        test([], [a], 1)
        test([], [a], 4)
        test([b], [b], 0)
        test([], [b], 1)
        test([], [b], 4)
        test([a, b], [a, b], 0)
        test([b], [a, b], 1)
        test([], [a, b], 4)
        test([b, a], [b, a], 0)
        test([a], [b, a], 1)
        test([], [b, a], 4)
        test([a, b, c], [a, b, c], 0)
        test([b, c], [a, b, c], 1)
        test([], [a, b, c], 4)

    def test_drop_while(self):
        test = by_iter_eq(Dict, "drop_while")

        a, b, c = enumerate("abc")

        test([], [], t(key_less_than(1)))
        test([], [a], t(key_less_than(1)))
        test([b], [b], t(key_less_than(1)))
        test([b], [a, b], t(key_less_than(1)))
        test([b, a], [b, a], t(key_less_than(1)))
        test([b, c], [a, b, c], t(key_less_than(1)))
        test([], [], t(key_less_than(2)))
        test([], [a], t(key_less_than(2)))
        test([], [b], t(key_less_than(2)))
        test([], [a, b], t(key_less_than(2)))
        test([], [b, a], t(key_less_than(2)))
        test([c], [a, b, c], t(key_less_than(2)))

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
        test = by_iter_eq(Dict, "filter")

        a, b, c = enumerate("abc")

        test([], [], t(key_less_than(1)))
        test([a], [a], t(key_less_than(1)))
        test([], [b], t(key_less_than(1)))
        test([a], [a, b], t(key_less_than(1)))
        test([a], [b, a], t(key_less_than(1)))
        test([a], [a, b, c], t(key_less_than(1)))
        test([], [], t(key_less_than(2)))
        test([a], [a], t(key_less_than(2)))
        test([b], [b], t(key_less_than(2)))
        test([a, b], [a, b], t(key_less_than(2)))
        test([b, a], [b, a], t(key_less_than(2)))
        test([a, b], [a, b, c], t(key_less_than(2)))

    def test_filter_keys(self):
        test = by_iter_eq(Dict, "filter_keys")

        a, b, c = enumerate("abc")

        test([], [], less_than(1))
        test([a], [a], less_than(1))
        test([], [b], less_than(1))
        test([a], [a, b], less_than(1))
        test([a], [b, a], less_than(1))
        test([a], [a, b, c], less_than(1))
        test([], [], less_than(2))
        test([a], [a], less_than(2))
        test([b], [b], less_than(2))
        test([a, b], [a, b], less_than(2))
        test([b, a], [b, a], less_than(2))
        test([a, b], [a, b, c], less_than(2))

    def test_filter_values(self):
        test = by_iter_eq(Dict, "filter_values")

        a, b, c = enumerate("abc")

        test([], [], less_than("b"))
        test([a], [a], less_than("b"))
        test([], [b], less_than("b"))
        test([a], [a, b], less_than("b"))
        test([a], [b, a], less_than("b"))
        test([a], [a, b, c], less_than("b"))
        test([], [], less_than("c"))
        test([a], [a], less_than("c"))
        test([b], [b], less_than("c"))
        test([a, b], [a, b], less_than("c"))
        test([b, a], [b, a], less_than("c"))
        test([a, b], [a, b, c], less_than("c"))

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
        test = by_iter_eq(Dict, "flat_map")

        a, b, c = enumerate("abc")

        def key_range(key, value):
            return [(k, value) for k in range(key)]

        test([], [], t(key_range))
        test([], [a], t(key_range))
        test([(0, "b")], [b], t(key_range))
        test([(0, "b")], [a, b], t(key_range))
        test([(0, "b")], [b, a], t(key_range))
        test([(0, "b"), (0, "c"), (1, "c")], [a, b, c], t(key_range))

    def test_flat_map_keys(self):
        test = by_iter_eq(Dict, "flat_map_keys")

        a, b, c = enumerate("abc")

        test([], [], range)
        test([], [a], range)
        test([(0, "b")], [b], range)
        test([(0, "b")], [a, b], range)
        test([(0, "b")], [b, a], range)
        test([(0, "b"), (0, "c"), (1, "c")], [a, b, c], range)

    def test_flat_map_values(self):
        test = by_iter_eq(Dict, "flat_map_values")

        a, b, c = enumerate("abc")

        def letter_tower(c):
            n = ord(c) - ord("a")
            return c * n

        test([], [], letter_tower)
        test([], [a], letter_tower)
        test([b], [b], letter_tower)
        test([b], [a, b], letter_tower)
        test([b], [b, a], letter_tower)
        test([b, c, c], [a, b, c], letter_tower)

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
        pass

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
        pass

    def test_intersection(self):
        pass

    def test_intersperse(self):
        pass

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

    def test_iter_keys(self):
        test = by_iter_eq(Dict, "iter_keys")

        a, b, c = enumerate("abc")

        test([], [])
        test([0], [a])
        test([1], [b])
        test([0, 1], [a, b])
        test([1, 0], [b, a])
        test([0, 1, 2], [a, b, c])

    def test_iter_values(self):
        test = by_iter_eq(Dict, "iter_values")

        a, b, c = enumerate("abc")

        test([], [])
        test(["a"], [a])
        test(["b"], [b])
        test(["a", "b"], [a, b])
        test(["b", "a"], [b, a])
        test(["a", "b", "c"], [a, b, c])

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
            from ajprax.collections import List
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
        test = by_iter_eq(Dict, "map")

        a, b, c = enumerate("abc")

        test([], [], t(double_key))
        test([a], [a], t(double_key))
        test([(2, "b")], [b], t(double_key))
        test([a, (2, "b")], [a, b], t(double_key))
        test([(2, "b"), a], [b, a], t(double_key))
        test([a, (2, "b"), (4, "c")], [a, b, c], t(double_key))
        test([], [], t(double_value))
        test([(0, "aa")], [a], t(double_value))
        test([(1, "bb")], [b], t(double_value))
        test([(0, "aa"), (1, "bb")], [a, b], t(double_value))
        test([(1, "bb"), (0, "aa")], [b, a], t(double_value))
        test([(0, "aa"), (1, "bb"), (2, "cc")], [a, b, c], t(double_value))

    def test_map_keys(self):
        test = by_iter_eq(Dict, "map_keys")

        a, b, c = enumerate("abc")

        test([], [], double)
        test([a], [a], double)
        test([(2, "b")], [b], double)
        test([a, (2, "b")], [a, b], double)
        test([(2, "b"), a], [b, a], double)
        test([a, (2, "b"), (4, "c")], [a, b, c], double)

    def test_map_values(self):
        test = by_iter_eq(Dict, "map_values")

        a, b, c = enumerate("abc")

        test([], [], double)
        test([(0, "aa")], [a], double)
        test([(1, "bb")], [b], double)
        test([(0, "aa"), (1, "bb")], [a, b], double)
        test([(1, "bb"), (0, "aa")], [b, a], double)
        test([(0, "aa"), (1, "bb"), (2, "cc")], [a, b, c], double)

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
        pass

    def test_permutations(self):
        pass

    def test_powerset(self):
        pass

    def test_product(self):
        pass

    def test_put(self):
        pass

    def test_repeat(self):
        pass

    def test_set(self):
        pass

    def test_size(self):
        pass

    def test_sliding(self):
        pass

    def test_sliding_by_timestamp(self):
        pass

    def test_take(self):
        pass

    def test_take_while(self):
        pass

    def test_timestamp(self):
        pass

    def test_tqdm(self):
        pass

    def test_tuple(self):
        pass

    def test_union(self):
        pass

    def test_update(self):
        pass

    def test_zip(self):
        pass

    def test_zip_longest(self):
        pass


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
        pass

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

    def test_apply_and_iter(self):
        test = by_iter_eq(Iter, "apply_and_iter")

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
        from ajprax.collections import List

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
        # repeat > 1, but it does allow multiple iterables and repeat > 1 and it repeats all iterables. Effectively it
        # takes the product of the iterables and products the result `repeat` times, flattening the result.
        test([(1, 3, 1, 3), (1, 3, 1, 4), (1, 3, 2, 3), (1, 3, 2, 4), (1, 4, 1, 3), (1, 4, 1, 4), (1, 4, 2, 3),
              (1, 4, 2, 4), (2, 3, 1, 3), (2, 3, 1, 4), (2, 3, 2, 3), (2, 3, 2, 4), (2, 4, 1, 3), (2, 4, 1, 4),
              (2, 4, 2, 3), (2, 4, 2, 4), ], [1, 2], [3, 4], repeat=2, )
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
        from ajprax.collections import Set
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
        pass

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
        pass

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
        from ajprax.collections import Tuple
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
        pass

    def test___getitem__(self):
        pass

    def test___iter__(self):
        pass

    def test___mul__(self):
        pass

    def test___rmul__(self):
        pass

    def test_all(self):
        pass

    def test_any(self):
        pass

    def test_append(self):
        pass

    def test_apply(self):
        pass

    def test_apply_and_iter(self):
        pass

    def test_batch(self):
        pass

    def test_chain(self):
        pass

    def test_clear(self):
        pass

    def test_combinations(self):
        pass

    def test_combine_if(self):
        pass

    def test_count(self):
        pass

    def test_cycle(self):
        pass

    def test_dict(self):
        pass

    def test_discard(self):
        pass

    def test_distinct(self):
        pass

    def test_do(self):
        pass

    def test_drop(self):
        pass

    def test_drop_while(self):
        pass

    def test_enumerate(self):
        pass

    def test_extend(self):
        pass

    def test_filter(self):
        pass

    def test_first(self):
        pass

    def test_flat_map(self):
        pass

    def test_flatten(self):
        pass

    def test_fold(self):
        pass

    def test_fold_while(self):
        pass

    def test_for_each(self):
        pass

    def test_group_by(self):
        pass

    def test_insert(self):
        pass

    def test_intersperse(self):
        pass

    def test_iter(self):
        pass

    def test_last(self):
        pass

    def test_list(self):
        pass

    def test_map(self):
        pass

    def test_map_to_keys(self):
        pass

    def test_map_to_pairs(self):
        pass

    def test_map_to_values(self):
        pass

    def test_max(self):
        pass

    def test_min(self):
        pass

    def test_min_max(self):
        pass

    def test_only(self):
        pass

    def test_partition(self):
        pass

    def test_permutations(self):
        pass

    def test_powerset(self):
        pass

    def test_product(self):
        pass

    def test_reduce(self):
        pass

    def test_repeat(self):
        pass

    def test_reverse(self):
        pass

    def test_set(self):
        pass

    def test_size(self):
        pass

    def test_sliding(self):
        pass

    def test_sliding_by_timestamp(self):
        pass

    def test_sort(self):
        pass

    def test_take(self):
        pass

    def test_take_while(self):
        pass

    def test_timestamp(self):
        pass

    def test_tqdm(self):
        pass

    def test_transpose(self):
        pass

    def test_tuple(self):
        pass

    def test_unzip(self):
        pass  # direct alias of transpose, does not need further testing

    def test_zip(self):
        pass

    def test_zip_longest(self):
        pass


class TestRange:
    def test___contains__(self):
        pass

    def test___eq__(self):
        pass

    def test___getitem__(self):
        pass

    def test___hash__(self):
        pass

    def test___iter__(self):
        pass

    def test___len__(self):
        pass

    def test___repr__(self):
        pass

    def test___reversed__(self):
        pass

    def test_start(self):
        pass

    def test_stop(self):
        pass

    def test_step(self):
        pass

    def test_all(self):
        pass

    def test_any(self):
        pass

    def test_apply(self):
        pass

    def test_apply_and_iter(self):
        pass

    def test_batch(self):
        pass

    def test_chain(self):
        pass

    def test_combinations(self):
        pass

    def test_combine_if(self):
        pass

    def test_count(self):
        pass

    def test_cycle(self):
        pass

    def test_dict(self):
        pass

    def test_distinct(self):
        pass

    def test_do(self):
        pass

    def test_drop(self):
        pass

    def test_drop_while(self):
        pass

    def test_enumerate(self):
        pass

    def test_filter(self):
        pass

    def test_first(self):
        pass

    def test_flat_map(self):
        pass

    def test_flatten(self):
        pass

    def test_fold(self):
        pass

    def test_fold_while(self):
        pass

    def test_for_each(self):
        pass

    def test_group_by(self):
        pass

    def test_intersperse(self):
        pass

    def test_last(self):
        pass

    def test_list(self):
        pass

    def test_map(self):
        pass

    def test_map_to_keys(self):
        pass

    def test_map_to_pairs(self):
        pass

    def test_map_to_values(self):
        pass

    def test_max(self):
        pass

    def test_min(self):
        pass

    def test_min_max(self):
        pass

    def test_only(self):
        pass

    def test_partition(self):
        pass

    def test_permutations(self):
        pass

    def test_powerset(self):
        pass

    def test_product(self):
        pass

    def test_reduce(self):
        pass

    def test_repeat(self):
        pass

    def test_reverse(self):
        pass

    def test_set(self):
        pass

    def test_size(self):
        pass

    def test_sliding(self):
        pass

    def test_sliding_by_timestamp(self):
        pass

    def test_take(self):
        pass

    def test_take_while(self):
        pass

    def test_timestamp(self):
        pass

    def test_tqdm(self):
        pass

    def test_transpose(self):
        pass

    def test_tuple(self):
        pass

    def test_zip(self):
        pass

    def test_zip_longest(self):
        pass

    def test_index(self):
        pass

    def test_iter(self):
        pass


class TestSet:
    def test___iter__(self):
        pass

    def test_add(self):
        pass

    def test_all(self):
        pass

    def test_any(self):
        pass

    def test_apply(self):
        pass

    def test_apply_and_iter(self):
        pass

    def test_batch(self):
        pass

    def test_chain(self):
        pass

    def test_clear(self):
        pass

    def test_combinations(self):
        pass

    def test_combine_if(self):
        pass

    def test_count(self):
        pass

    def test_cycle(self):
        pass

    def test_dict(self):
        pass

    def test_discard(self):
        pass

    def test_distinct(self):
        pass

    def test_do(self):
        pass

    def test_drop(self):
        pass

    def test_drop_while(self):
        pass

    def test_enumerate(self):
        pass

    def test_filter(self):
        pass

    def test_first(self):
        pass

    def test_flat_map(self):
        pass

    def test_flatten(self):
        pass

    def test_fold(self):
        pass

    def test_fold_while(self):
        pass

    def test_for_each(self):
        pass

    def test_group_by(self):
        pass

    def test_intersperse(self):
        pass

    def test_iter(self):
        pass

    def test_last(self):
        pass

    def test_list(self):
        pass

    def test_map(self):
        pass

    def test_map_to_keys(self):
        pass

    def test_map_to_pairs(self):
        pass

    def test_map_to_values(self):
        pass

    def test_max(self):
        pass

    def test_min(self):
        pass

    def test_min_max(self):
        pass

    def test_only(self):
        pass

    def test_partition(self):
        pass

    def test_permutations(self):
        pass

    def test_powerset(self):
        pass

    def test_product(self):
        pass

    def test_reduce(self):
        pass

    def test_repeat(self):
        pass

    def test_set(self):
        pass

    def test_size(self):
        pass

    def test_sliding(self):
        pass

    def test_sliding_by_timestamp(self):
        pass

    def test_take(self):
        pass

    def test_take_while(self):
        pass

    def test_timestamp(self):
        pass

    def test_tqdm(self):
        pass

    def test_transpose(self):
        pass

    def test_tuple(self):
        pass

    def test_update(self):
        pass

    def test_zip(self):
        pass

    def test_zip_longest(self):
        pass


class TestTuple:
    def test___add__(self):
        pass

    def test___mul__(self):
        pass

    def test___rmul__(self):
        pass

    def test___getitem__(self):
        pass

    def test___iter__(self):
        pass

    def test_all(self):
        pass

    def test_any(self):
        pass

    def test_apply(self):
        pass

    def test_apply_and_iter(self):
        pass

    def test_batch(self):
        pass

    def test_chain(self):
        pass

    def test_combinations(self):
        pass

    def test_combine_if(self):
        pass

    def test_count(self):
        pass

    def test_cycle(self):
        pass

    def test_dict(self):
        pass

    def test_distinct(self):
        pass

    def test_do(self):
        pass

    def test_drop(self):
        pass

    def test_drop_while(self):
        pass

    def test_enumerate(self):
        pass

    def test_filter(self):
        pass

    def test_first(self):
        pass

    def test_flat_map(self):
        pass

    def test_flatten(self):
        pass

    def test_fold(self):
        pass

    def test_fold_while(self):
        pass

    def test_for_each(self):
        pass

    def test_group_by(self):
        pass

    def test_intersperse(self):
        pass

    def test_iter(self):
        pass

    def test_last(self):
        pass

    def test_list(self):
        pass

    def test_map(self):
        pass

    def test_map_to_keys(self):
        pass

    def test_map_to_pairs(self):
        pass

    def test_map_to_values(self):
        pass

    def test_max(self):
        pass

    def test_min(self):
        pass

    def test_min_max(self):
        pass

    def test_only(self):
        pass

    def test_partition(self):
        pass

    def test_permutations(self):
        pass

    def test_powerset(self):
        pass

    def test_product(self):
        pass

    def test_reduce(self):
        pass

    def test_repeat(self):
        pass

    def test_reverse(self):
        pass

    def test_set(self):
        pass

    def test_size(self):
        pass

    def test_sliding(self):
        pass

    def test_sliding_by_timestamp(self):
        pass

    def test_sorted(self):
        pass

    def test_take(self):
        pass

    def test_take_while(self):
        pass

    def test_timestamp(self):
        pass

    def test_tqdm(self):
        pass

    def test_transpose(self):
        pass

    def test_tuple(self):
        pass

    def test_unzip(self):
        pass

    def test_zip(self):
        pass

    def test_zip_longest(self):
        pass
