from ajprax.sentinel import Unset


def test_is():
    assert Unset is Unset
    assert Unset() is Unset
    assert Unset is Unset()
    assert Unset() is Unset()
