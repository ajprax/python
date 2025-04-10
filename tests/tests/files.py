from base64 import b64encode
from os import path

from ajprax.files import contents, crc32c, md5, sha1, sha256
from tests import iter_eq


def asset(name):
    return path.join(path.dirname(__file__), "assets", name)


def test_contents():
    def test(filename, chunks, *a, **kw):
        assert iter_eq(contents(filename, *a, **kw), chunks)

    test(asset("empty"), [])
    test(asset("empty"), [], binary=True)
    test(asset("empty"), [], chunk_size=1)
    test(asset("empty"), [], binary=True, chunk_size=5)
    test(asset("stuff"), ["some text or something"])
    test(asset("stuff"), [b"some text or something"], binary=True)
    test(asset("stuff"), ["some ", "text ", "or so", "methi", "ng"], chunk_size=5)
    test(asset("stuff"), [b"some ", b"text ", b"or so", b"methi", b"ng"], binary=True, chunk_size=5)


def test_hashers():
    def test(hash, empty, stuff):
        def test(filename, raw):
            assert hash(asset(filename)) == raw
            assert hash(asset(filename), b64encode) == b64encode(raw)

        test("empty", empty)
        test("stuff", stuff)

    test(
        crc32c,
        b"\x00\x00\x00\x00",
        b"\xf7\x88\xc4\x84",
    )
    test(
        md5,
        b"\xd4\x1d\x8c\xd9\x8f\x00\xb2\x04\xe9\x80\t\x98\xec\xf8B~",
        b"P\xd22\x8b\x9a\x85!\xc2\xefm\xc8\xde\x06\xba\x12\xa9"
    )
    test(
        sha1,
        b"\xda9\xa3\xee^kK\r2U\xbf\xef\x95`\x18\x90\xaf\xd8\x07\t",
        b"\xcaFe\x8a\x95\x192\xa2\xcbu\x89\xed\x7fs\xbf|\x86\xb3\x14\x80"
    )
    test(
        sha256,
        b"\xe3\xb0\xc4B\x98\xfc\x1c\x14\x9a\xfb\xf4\xc8\x99o\xb9$'\xaeA\xe4d\x9b\x93L\xa4\x95\x99\x1bxR\xb8U",
        b"o\x11z\xfe\xd94\xc27{5\xdf\x98q\xa6\xfc\xb1\x95\xcc\xa27#\xe7\xca\xd4\xbe\xf2\x0f6\x1e\xd7K\xed",
    )
