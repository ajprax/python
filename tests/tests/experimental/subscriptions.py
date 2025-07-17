from ajprax.sentinel import Unset
from ajprax.experimental.subscriptions import Notifications, Events, Value
from tests import is_odd, tower, double


class TestNotifications:
    def test_subscribe_and_notify(self):
        def test(*expected_args, **expected_kwargs):
            notifications = Notifications()

            def callback(*actual_args, **actual_kwargs):
                assert expected_args == actual_args
                assert expected_kwargs == actual_kwargs

            notifications.subscribe(callback)
            notifications.notify(*expected_args, **expected_kwargs)

        test()
        test(0)
        test(a=1)
        test(0, 1)
        test([0, 1])
        test(0, a=1)

    def test_unsubscribe(self):
        def test():
            assert False, "callback should not be called"

        notifications = Notifications()
        notifications.subscribe(test)
        try:
            notifications.notify()
        except AssertionError:
            pass
        notifications.unsubscribe(test)
        notifications.notify()

        with notifications.subscribe(test):
            try:
                notifications.notify()
            except AssertionError:
                pass
        notifications.notify()

        unsub = notifications.subscribe(test)
        try:
            notifications.notify()
        except AssertionError:
            pass
        unsub()
        notifications.notify()


class TestEvents:
    def test_subscribe(self):
        def test(expected):
            def callback(actual):
                assert actual == expected

            events = Events()
            events.subscribe(callback)
            events.send(expected)

        test(0)
        test(1)
        test([0, 1])

    def test_filter(self):
        def test(fn, input, expected):
            def callback(actual):
                assert actual == expected

            events = Events()
            filtered = events.filter(fn)
            filtered.subscribe(callback)
            events.send(input)

        test(is_odd, 0, None)
        test(is_odd, 1, 1)
        test(is_odd, 2, None)
        test(is_odd, 3, 3)

    def test_flat_map(self):
        def test(fn, input, expected):
            actuals = []

            def callback(actual):
                actuals.append(actual)

            events = Events()
            flat_mapped = events.flat_map(fn)
            flat_mapped.subscribe(callback)
            events.send(input)
            assert actuals == expected

        test(tower, 0, [])
        test(tower, 1, [1])
        test(tower, 2, [2, 2])

    def test_map(self):
        def test(fn, input, expected):
            def callback(actual):
                assert actual == expected

            events = Events()
            mapped = events.map(fn)
            mapped.subscribe(callback)
            events.send(input)

        test(double, 0, 0)
        test(double, 1, 2)
        test(double, 2, 4)
        test(double, "a", "aa")
        test(double, [0], [0, 0])

    def test_on(self):
        def test(input):
            called = False

            def callback(*a, **kw):
                nonlocal called
                called = True
                assert not a
                assert not kw

            events = Events()
            on = events.on()
            on.subscribe(callback)
            assert not called
            events.send(input)
            assert called

        test(0)
        test(1)
        test([0, 1])


class TestValue:
    def test_initial(self):
        value = Value(0)
        assert value.value == 0
        value = Value()
        assert value.value is Unset

    def test_subscribe(self):
        def test(actual):
            assert actual == 0

        value = Value()
        value.subscribe(test)
        value.set(0)
        try:
            value.set(1)
        except AssertionError:
            pass

    def test_changed(self):
        def test():
            nonlocal count
            count += 1
            assert count == expected

        count = 0
        value = Value()
        value.changed().subscribe(test)
        expected = 1
        value.set(0)
        expected = 2
        value.set(1)
        expected = 3
        value.set(0)

    def test_changes(self):
        def test(actual):
            nonlocal actual_count
            actual_count += 1
            assert actual == expected
            assert expected_count == actual_count

        value = Value()
        value.changes().subscribe(test)
        actual_count = 0
        for expected_count, expected in enumerate((0, 1, 0), 1):
            value.set(expected)

    def test_map(self):
        def test(fn, input, expected):
            def callback(actual):
                assert actual == expected

            value = Value()
            mapped = value.map(fn)
            mapped.subscribe(callback)
            value.set(input)

        test(double, 0, 0)
        test(double, 1, 2)
        test(double, 2, 4)
        test(double, "a", "aa")
        test(double, [0], [0, 0])

    def test_on(self):
        def test(input):
            called = False

            def callback(*a, **kw):
                nonlocal called
                called = True
                assert not a
                assert not kw

            value = Value()
            on = value.on()
            on.subscribe(callback)
            assert not called
            value.set(input)
            assert called

        test(0)
        test(1)
        test([0, 1])

    def test_zip(self):
        def test(actual):
            assert actual == expected

        va = Value()
        vb = Value()
        zipped = va.zip(vb)
        zipped.subscribe(test)
        expected = (0, 1)
        va.set(0)
        vb.set(1)
