from contextlib import contextmanager
from itertools import zip_longest

from ajprax.sentinel import Unset


@contextmanager
def should_raise(exc, s=Unset):
    try:
        yield
        assert False, "should have raised"
    except exc as e:
        if s is not Unset:
            assert str(e) == s


def iter_eq(a, b):
    return all(a == b for a, b in zip_longest(a, b, fillvalue=Unset))


def is_even(i):
    return not i % 2


def is_odd(i):
    return i % 2


def key_is_odd(key, value):
    return key % 2


def double(i):
    return i * 2


def double_key(key, value):
    return key * 2, value


def double_value(key, value):
    return key, value * 2


def tower(i):
    return [i] * i


def less_than(n):
    def test(i):
        return i < n

    return test


def key_less_than(n):
    def test(key, value):
        return key < n

    return test


def greater_than(n):
    def test(i):
        return i > n

    return test


def by_eq(cls, method):
    def test(expected, items, *a, **kw):
        actual = getattr(cls(items), method)(*a, **kw)
        assert actual == expected

    return test


def by_iter_eq(cls, method):
    def test(expected, items, *a, **kw):
        actual = tuple(getattr(cls(items), method)(*a, **kw))
        expected = tuple(expected)
        assert actual == expected, (actual, expected)

    return test


def by_member_eq(cls, member):
    def test(expected, items):
        actual = getattr(cls(items), member)
        assert actual == expected

    return test
