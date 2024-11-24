try:
    import crcmod
except ImportError:
    crcmod = False


def hash(filename, hasher, encode=None):
    h = hasher()
    with open(filename, "rb") as f:
        chunk = f.read(4096)
        while len(chunk) > 0:
            h.update(chunk)
            chunk = f.read(4096)
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
