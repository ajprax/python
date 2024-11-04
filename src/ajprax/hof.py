def identity(x):
    return x


def kw(f):
    """
    Convert a function to take a single dictionary of keyword arguments

    Similar to, but more explicit that `t`
    Example:
         def add(a=0, b=0):
             return a + b

        # imagine that the standard library had nice collections
        kwargs = [dict(a=5), dict(b=10), dict(a=5, b=10)]
        kwargs.map(lambda d: add(d.get("a", 0), d.get("b", 0)))
        # no manual decomposition of the dict, use functions that weren't written for use with HOFs
        kwargs.map(kw(add))
        # use more readable lambdas
        kwargs.map(kw(lambda a=0, b=0: a + b))
    """

    def kw(kw):
        return f(**kw)

    return kw


def t(f):
    """
    Convert a function of N arguments to a function of one N-degree tuple

    Useful with higher order functions that expect a single-argument function as input. This is given as an alternative
    to implementing a star_map, star_for_each, star_flat_map, etc functions for all collections.

    Example:
        def add(a, b):
            return a + b

        # imagine that the standard library had nice collections
        pairs = [(1, 2), (2, 4), (3, 6)]
        pairs.map(lambda pair: add(pair[0], pair[1]))
        # no manual decomposition of the pair, use functions that weren't written for use with HOFs
        pairs.map(t(add))
        # use more readable lambdas
        pairs.map(t(lambda a, b: a + b))
    """

    def t(t):
        return f(*t)

    return t
