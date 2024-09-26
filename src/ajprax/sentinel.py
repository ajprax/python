class Sentinel(object):
    def __new__(cls, *a, **kw):
        return cls

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__


class Unset(Sentinel):
    pass
