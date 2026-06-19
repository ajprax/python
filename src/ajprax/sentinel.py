from typing import TypeVar


S = TypeVar("S", bound="Sentinel")


class _SentinelMeta(type):
    def __str__(cls) -> str:
        return cls.__name__

    def __repr__(cls) -> str:
        return cls.__name__


class Sentinel(metaclass=_SentinelMeta):
    def __new__(cls: type[S], *a: object, **kw: object) -> type[S]:
        return cls


class Unset(Sentinel):
    pass
