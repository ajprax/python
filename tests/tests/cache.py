from functools import partial

from ajprax.cache import cache


def add(a, b):
    return a + b


class TestCache:
    def _test_default_key(self, f):
        def test(expected_count, expected_value, f, *a, **kw):
            actual_value = f(*a, **kw)
            actual_count = self.count
            assert actual_value == expected_value
            assert actual_count == expected_count

        self.count = 0
        test(1, [1, 1], f, 1, 2)
        test(1, [1, 1], f, 1, b=2)
        test(1, [1, 1], f, a=1, b=2)
        test(2, [1, 1, 1], f, 1, 3)
        test(3, [2], f, 2, 1)

    def _test_custom_key(self, f):
        def test(expected_count, expected_value, f, *a, **kw):
            actual_value = f(*a, **kw)
            actual_count = self.count
            assert actual_value == expected_value
            assert actual_count == expected_count

        self.count = 0
        test(1, [1, 1], f, 1, 2)
        test(1, [1, 1], f, 1, b=2)
        test(1, [1, 1], f, a=1, b=2)
        test(2, [1, 1, 1], f, 1, 3)
        test(2, [1, 1], f, 2, 1)

    def test_function(self):
        @cache()
        def f(a, b):
            self.count += 1
            return [a] * b

        self._test_default_key(f)

    def test_function_custom(self):
        @cache(key=add)
        def f(a, b):
            self.count += 1
            return [a] * b

        self._test_custom_key(f)

    def test_instance_method(self):
        class C:
            @cache(method=True)
            def f(_, a, b):
                self.count += 1
                return [a] * b

        self._test_default_key(C().f)

    def test_instance_method_custom(self):
        class C:
            @cache(key=add, method=True)
            def f(_, a, b):
                self.count += 1
                return [a] * b

        self._test_custom_key(C().f)

    def test_instance_method_partial(self):
        class C:
            @cache(method=True)
            def f(_, a, b):
                self.count += 1
                return [a] * b

        self._test_default_key(partial(C.f, C()))

    def test_instance_method_partial_custom(self):
        class C:
            @cache(key=add, method=True)
            def f(_, a, b):
                self.count += 1
                return [a] * b

        self._test_custom_key(partial(C.f, C()))

    def test_classmethod(self):
        class C:
            @classmethod
            @cache(method=True)
            def f(_, a, b):
                self.count += 1
                return [a] * b

        self._test_default_key(C.f)

    def test_classmethod_custom(self):
        class C:
            @classmethod
            @cache(key=add, method=True)
            def f(_, a, b):
                self.count += 1
                return [a] * b

        self._test_custom_key(C.f)

    def test_classmethod_via_instance(self):
        class C:
            @classmethod
            @cache(method=True)
            def f(_, a, b):
                self.count += 1
                return [a] * b

        self._test_default_key(C().f)

    def test_classmethod_via_instance_custom(self):
        class C:
            @classmethod
            @cache(key=add, method=True)
            def f(_, a, b):
                self.count += 1
                return [a] * b

        self._test_custom_key(C().f)

    def test_classmethod_subclass(self):
        class C:
            @classmethod
            @cache(method=True)
            def f(_, a, b):
                self.count += 1
                return [a] * b

        class D(C):
            pass

        self._test_default_key(D.f)

        # C.f should have nothing in it, so calling it will increase self.count
        count = self.count
        C.f(1, 2)
        assert self.count == count + 1

    def test_classmethod_subclass_custom(self):
        class C:
            @classmethod
            @cache(key=add, method=True)
            def f(_, a, b):
                self.count += 1
                return [a] * b

        class D(C):
            pass

        self._test_custom_key(D.f)

        # C.f should have nothing in it, so calling it will increase self.count
        count = self.count
        C.f(1, 2)
        assert self.count == count + 1

    def test_classmethod_subclass_via_instance(self):
        class C:
            @classmethod
            @cache(method=True)
            def f(_, a, b):
                self.count += 1
                return [a] * b

        class D(C):
            pass

        self._test_default_key(D().f)

        # the cache should live with the class
        count = self.count
        D.f(1, 2)
        assert self.count == count

        # C.f should have nothing in it, so calling it will increase self.count
        count = self.count
        C.f(1, 2)
        assert self.count == count + 1

    def test_classmethod_subclass_via_instance_custom(self):
        class C:
            @classmethod
            @cache(key=add, method=True)
            def f(_, a, b):
                self.count += 1
                return [a] * b

        class D(C):
            pass

        self._test_custom_key(D().f)

        # the cache should live with the class
        count = self.count
        D.f(1, 2)
        assert self.count == count

        # C.f should have nothing in it, so calling it will increase self.count
        count = self.count
        C.f(1, 2)
        assert self.count == count + 1

    def test_staticmethod(self):
        class C:
            @staticmethod
            @cache()
            def f(a, b):
                self.count += 1
                return [a] * b

        self._test_default_key(C.f)

    def test_staticmethod_custom(self):
        class C:
            @staticmethod
            @cache(key=add)
            def f(a, b):
                self.count += 1
                return [a] * b

        self._test_custom_key(C.f)

    def test_staticmethod_via_instance(self):
        class C:
            @staticmethod
            @cache()
            def f(a, b):
                self.count += 1
                return [a] * b

        self._test_default_key(C().f)

        # the cache should live with the class
        count = self.count
        C.f(1, 2)
        assert self.count == count

    def test_staticmethod_via_instance_custom(self):
        class C:
            @staticmethod
            @cache(key=add)
            def f(a, b):
                self.count += 1
                return [a] * b

        self._test_custom_key(C().f)

        # the cache should live with the class
        count = self.count
        C.f(1, 2)
        assert self.count == count

    def test_staticmethod_subclass(self):
        class C:
            @staticmethod
            @cache()
            def f(a, b):
                self.count += 1
                return [a] * b

        class D(C):
            pass

        self._test_default_key(D.f)

        # for static methods, the cache is shared with subclasses
        count = self.count
        C.f(1, 2)
        assert self.count == count

    def test_staticmethod_subclass_custom(self):
        class C:
            @staticmethod
            @cache(key=add)
            def f(a, b):
                self.count += 1
                return [a] * b

        class D(C):
            pass

        self._test_custom_key(D.f)

        # for static methods, the cache is shared with subclasses
        count = self.count
        C.f(1, 2)
        assert self.count == count

    def test_staticmethod_subclass_via_instance(self):
        class C:
            @staticmethod
            @cache()
            def f(a, b):
                self.count += 1
                return [a] * b

        class D(C):
            pass

        self._test_default_key(D().f)

        # for static methods, the cache is shared with subclasses
        count = self.count
        C.f(1, 2)
        assert self.count == count

    def test_staticmethod_subclass_via_instance_custom(self):
        class C:
            @staticmethod
            @cache(key=add)
            def f(a, b):
                self.count += 1
                return [a] * b

        class D(C):
            pass

        self._test_custom_key(D().f)

        # for static methods, the cache is shared with subclasses
        count = self.count
        C.f(1, 2)
        assert self.count == count

    def test_default_values(self):
        @cache()
        def f(a, b=2):
            self.count += 1
            return [a] * b

        def test(expected_count, expected_value, f, *a, **kw):
            actual_value = f(*a, **kw)
            actual_count = self.count
            assert actual_value == expected_value
            assert actual_count == expected_count

        self.count = 0
        test(1, [1, 1], f, 1)
        test(1, [1, 1], f, 1)
        test(1, [1, 1], f, a=1)
        test(2, [1, 1, 1], f, 1, 3)
        test(3, [2], f, 2, 1)

    def test_default_values_custom(self):
        @cache(key=add)
        def f(a, b=2):
            self.count += 1
            return [a] * b

        def test(expected_count, expected_value, f, *a, **kw):
            actual_value = f(*a, **kw)
            actual_count = self.count
            assert actual_value == expected_value
            assert actual_count == expected_count

        self.count = 0
        test(1, [1, 1], f, 1)
        test(1, [1, 1], f, 1)
        test(1, [1, 1], f, a=1)
        test(2, [1, 1, 1], f, 1, 3)
        test(2, [1, 1], f, 2, 1)

    def test_two_instances(self):
        class C:
            @cache(method=True)
            def f(_, a, b):
                self.count += 1
                return [a] * b

        c = C()
        c2 = C()

        self._test_default_key(c.f)
        self._test_default_key(c2.f)


class TestSingleton:

    def _test(self, f):
        self.count = 0
        f()
        assert self.count == 1
        f()
        assert self.count == 1

    def test_function(self):
        @cache()
        def f():
            self.count += 1

        self._test(f)

    def test_instance_method(self):
        class C:
            @cache(method=True)
            def f(_):
                self.count += 1

        c = C()
        self._test(c.f)

    def test_instance_method_partial(self):
        class C:
            @cache(method=True)
            def f(_):
                self.count += 1

        c = C()
        self._test(partial(C.f, c))

    def test_classmethod(self):
        class C:
            @classmethod
            @cache(method=True)
            def f(cls):
                self.count += 1

        self._test(C.f)

    def test_classmethod_via_instance(self):
        class C:
            @classmethod
            @cache(method=True)
            def f(cls):
                self.count += 1

        c = C()
        self._test(c.f)

    def test_classmethod_subclass(self):
        class C:
            @classmethod
            @cache(method=True)
            def f(cls):
                self.count += 1

        class D(C):
            pass

        self._test(D.f)
        count = self.count
        C.f()
        assert self.count == count + 1

    def test_classmethod_subclass_via_instance(self):
        class C:
            @classmethod
            @cache(method=True)
            def f(cls):
                self.count += 1

        class D(C):
            pass

        self._test(D().f)
        count = self.count
        C.f()
        assert self.count == count + 1

    def test_staticmethod(self):
        class C:
            @staticmethod
            @cache()
            def f():
                self.count += 1

        self._test(C.f)

    def test_staticmethod_via_instance(self):
        class C:
            @staticmethod
            @cache()
            def f():
                self.count += 1

        self._test(C().f)

    def test_staticmethod_subclass(self):
        class C:
            @staticmethod
            @cache()
            def f():
                self.count += 1

        class D(C):
            pass

        self._test(D.f)

    def test_staticmethod_subclass_via_instance(self):
        class C:
            @staticmethod
            @cache()
            def f():
                self.count += 1

        class D(C):
            pass

        self._test(D().f)

    def test_two_instances(self):
        class C:
            @cache(method=True)
            def f(_):
                self.count += 1

        c = C()
        c2 = C()

        self._test(c.f)
        self._test(c2.f)

    def test_property(self):
        class C:
            @property
            @cache(method=True)
            def f(_):
                self.count += 1

        self.count = 0
        c = C()
        c.f
        assert self.count == 1
        c.f
        assert self.count == 1
