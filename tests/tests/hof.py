from ajprax.hof import *
from tests import should_raise


def test_t():
    def add(a, b):
        return a + b

    args = [(0, 1), (2, 3), (4, 5)]

    with should_raise(TypeError):
        list(map(add, args))
    assert list(map(t(add), args)) == [1, 5, 9]


def test_kw():
    def add(a=0, b=0):
        return a + b

    kwargs = [{"a": 0, "b": 1}, {"a": 2}, {"b": 5}]
    with should_raise(TypeError):
        list(map(add, kwargs))

    assert list(map(kw(add), kwargs)) == [1, 2, 5]
