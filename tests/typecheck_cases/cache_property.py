from ajprax.cache import cache


class C:
    def __init__(self) -> None:
        self.raw = 1

    @cache
    @property
    def value(self) -> int:
        return self.raw

    @value.setter
    def value(self, value: int) -> None:
        self.raw = value

    @cache(key=lambda: "value")
    @property
    def keyed_value(self) -> int:
        return self.raw


instance = C()
value: int = instance.value
keyed_value: int = instance.keyed_value
C.value.clear(instance)
