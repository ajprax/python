import builtins
from io import StringIO

from ajprax.print import print


def test_print():
    def test(*a, **kw):
        mine = StringIO()
        builtin = StringIO()
        print(*a, **kw, file=mine)
        builtins.print(*a, **kw, file=builtin)
        assert mine.read() == builtin.read()

    test()
    test(1)
    test("a")
    test(1, 2)
    test(1, "a")
    test(1, "a", sep="-")
    test(1, "a", end="\r")
