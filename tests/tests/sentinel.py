from ajprax.sentinel import Unset


def test_is():
    assert Unset is Unset
    assert Unset() is Unset
    assert Unset is Unset()
    assert Unset() is Unset()


def test_string_representation():
    assert str(Unset) == "Unset"
    assert repr(Unset) == "Unset"
    assert str(Unset()) == "Unset"
    assert repr(Unset()) == "Unset"
