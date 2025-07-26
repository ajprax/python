from ajprax.cache import cache


def test_cache():
    count = 0

    def test(expected_count, expected, *a, **kw):
        actual = add(*a, **kw)
        actual_count = count
        assert actual == expected
        assert actual_count == expected_count

    @cache()
    def add(a, b):
        nonlocal count
        count += 1
        return a + b

    assert count == 0
    test(1, 3, 1, 2)
    test(1, 3, 1, 2)
    test(2, 3, 2, 1)
    test(2, 3, 1, b=2)
    test(2, 3, a=1, b=2)
    test(2, 3, a=2, b=1)
    test(3, 4, 1, 3)
    test(4, 4, 2, 2)

    count = 0

    @cache()
    def add(a, b=2):
        nonlocal count
        count += 1
        return a + b

    assert count == 0
    test(1, 3, 1)
    test(1, 3, a=1)
    test(1, 3, 1, 2)
    test(1, 3, a=1, b=2)

    count = 0

    def key(a, b):
        return a, b

    @cache(key)
    def add(a, b):
        nonlocal count
        count += 1
        return a + b

    assert count == 0
    test(1, 3, 1, 2)
    test(1, 3, 1, 2)
    test(2, 3, 2, 1)
    test(2, 3, 1, b=2)
    test(2, 3, a=1, b=2)
    test(2, 3, a=2, b=1)
    test(3, 4, 1, 3)
    test(4, 4, 2, 2)

    count = 0

    def key(a, b=5):
        return a, b

    @cache(key)
    def add(a, b=2):
        nonlocal count
        count += 1
        return a + b

    assert count == 0
    test(1, 3, 1, 2)
    test(1, 3, 1, 2)
    test(2, 3, 2, 1)
    test(2, 3, 1, b=2)
    test(2, 3, a=1, b=2)
    test(2, 3, a=2, b=1)
    test(3, 4, 1, 3)
    test(4, 4, 2, 2)
