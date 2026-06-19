from contextlib import contextmanager
from os import PathLike, remove
from shutil import copy, move
from tempfile import NamedTemporaryFile
from typing import Callable, Iterator, Literal, Optional, Protocol, TypeVar, Union, overload

from ajprax.collections import Iter
from ajprax.require import require

Path = Union[str, PathLike[str]]
T = TypeVar("T")


class _AtomicFile(Protocol):
    @overload
    def write(self, data: str, /) -> object:
        ...

    @overload
    def write(self, data: bytes, /) -> object:
        ...

    def flush(self) -> object:
        ...


class _Hasher(Protocol):
    def update(self, data: bytes, /) -> object:
        ...

    def digest(self) -> bytes:
        ...


@contextmanager
def atomic_write(filename: Path, mode: str = "w") -> Iterator[_AtomicFile]:
    with NamedTemporaryFile(mode, delete=False) as temp:
        yield temp
    move(temp.name, filename)


@contextmanager
def backup(filename: Path, backup_filename: Optional[Path] = None) -> Iterator[None]:
    if backup_filename is None:
        backup_filename = f"{filename}.backup"

    copy(filename, backup_filename)
    try:
        yield
    except:
        move(backup_filename, filename)
        raise
    else:
        remove(backup_filename)


@overload
def contents(filename: Path, binary: Literal[False] = False, chunk_size: int = 4096) -> Iter[str]:
    ...


@overload
def contents(filename: Path, binary: Literal[True], chunk_size: int = 4096) -> Iter[bytes]:
    ...


@overload
def contents(filename: Path, binary: bool, chunk_size: int = 4096) -> Iter[Union[str, bytes]]:
    ...


def contents(filename: Path, binary: bool = False, chunk_size: int = 4096) -> Iter[Union[str, bytes]]:
    require(chunk_size > 0, chunk_size=chunk_size)

    def gen() -> Iterator[Union[str, bytes]]:
        with open(filename, "rb" if binary else "r") as f:
            chunk = f.read(chunk_size)
            while chunk:
                yield chunk
                chunk = f.read(chunk_size)

    return Iter(gen())


@overload
def _hash(filename: Path, hasher: Callable[[], _Hasher], encode: None = None) -> bytes:
    ...


@overload
def _hash(filename: Path, hasher: Callable[[], _Hasher], encode: Callable[[bytes], T]) -> T:
    ...


def _hash(
    filename: Path,
    hasher: Callable[[], _Hasher],
    encode: Optional[Callable[[bytes], T]] = None,
) -> Union[bytes, T]:
    h = hasher()
    for chunk in contents(filename, binary=True):
        h.update(chunk)
    h = h.digest()
    if encode:
        h = encode(h)
    return h


@overload
def hash(filename: Path, hasher: Callable[[], _Hasher], encode: None = None) -> bytes:
    ...


@overload
def hash(filename: Path, hasher: Callable[[], _Hasher], encode: Callable[[bytes], T]) -> T:
    ...


def hash(
    filename: Path,
    hasher: Callable[[], _Hasher],
    encode: Optional[Callable[[bytes], T]] = None,
) -> Union[bytes, T]:
    if encode is None:
        return _hash(filename, hasher)
    return _hash(filename, hasher, encode)


try:
    import crcmod
except ImportError:
    pass
else:
    @overload
    def crc32c(filename: Path, encode: None = None) -> bytes:
        ...

    @overload
    def crc32c(filename: Path, encode: Callable[[bytes], T]) -> T:
        ...

    def crc32c(filename: Path, encode: Optional[Callable[[bytes], T]] = None) -> Union[bytes, T]:
        if encode is None:
            return _hash(filename, lambda: crcmod.predefined.Crc("crc-32c"))
        return _hash(filename, lambda: crcmod.predefined.Crc("crc-32c"), encode)


@overload
def md5(filename: Path, encode: None = None) -> bytes:
    ...


@overload
def md5(filename: Path, encode: Callable[[bytes], T]) -> T:
    ...


def md5(filename: Path, encode: Optional[Callable[[bytes], T]] = None) -> Union[bytes, T]:
    from hashlib import md5
    if encode is None:
        return _hash(filename, lambda: md5())
    return _hash(filename, lambda: md5(), encode)


@overload
def sha1(filename: Path, encode: None = None) -> bytes:
    ...


@overload
def sha1(filename: Path, encode: Callable[[bytes], T]) -> T:
    ...


def sha1(filename: Path, encode: Optional[Callable[[bytes], T]] = None) -> Union[bytes, T]:
    from hashlib import sha1
    if encode is None:
        return _hash(filename, lambda: sha1())
    return _hash(filename, lambda: sha1(), encode)


@overload
def sha256(filename: Path, encode: None = None) -> bytes:
    ...


@overload
def sha256(filename: Path, encode: Callable[[bytes], T]) -> T:
    ...


def sha256(filename: Path, encode: Optional[Callable[[bytes], T]] = None) -> Union[bytes, T]:
    from hashlib import sha256
    if encode is None:
        return _hash(filename, lambda: sha256())
    return _hash(filename, lambda: sha256(), encode)
