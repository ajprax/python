from typing import TypeVar


S = TypeVar("S", bound="Sentinel")


class Sentinel(object):
    def __new__(cls: type[S], *a: object, **kw: object) -> type[S]:
        return cls

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return self.__class__.__name__


class Unset(Sentinel):
    pass
