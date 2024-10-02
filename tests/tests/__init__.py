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


def double(i):
    return i * 2


def tower(i):
    return [i] * i
