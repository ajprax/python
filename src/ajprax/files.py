from ajprax.collections import Iter
from ajprax.require import require

try:
    import crcmod
except ImportError:
    crcmod = False


def contents(filename, binary=False, chunk_size=4096):
    require(chunk_size > 0, chunk_size=chunk_size)

    def gen():
        with open(filename, "rb" if binary else "r") as f:
            chunk = f.read(chunk_size)
            while chunk:
                yield chunk
                chunk = f.read(chunk_size)

    return Iter(gen())


def hash(filename, hasher, encode=None):
    h = hasher()
    for chunk in contents(filename, binary=True):
        h.update(chunk)
    h = h.digest()
    if encode:
        h = encode(h)
    return h


if crcmod:
    def crc32c(filename, encode=None):
        return hash(filename, lambda: crcmod.predefined.Crc("crc-32c"), encode=encode)


def md5(filename, encode=None):
    from hashlib import md5
    return hash(filename, md5, encode=encode)


def sha1(filename, encode=None):
    from hashlib import sha1
    return hash(filename, sha1, encode=encode)


def sha256(filename, encode=None):
    from hashlib import sha256
    return hash(filename, sha256, encode=encode)
