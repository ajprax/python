import gc
import threading
import time
import weakref
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import pytest

from ajprax.cache import DoNotCache, cache


TARGET_KINDS = ("function", "instance_method", "classmethod")
ARITIES = ("singleton", "dict")
KEY_MODES = ("default", "custom_identity")


@dataclass
class Subject:
    call: Callable
    clear: Callable
    remove: Callable
    call_count: Callable[[], int]
    descriptor: Optional[object] = None


def _cache_decorator_for(arity: str, key_mode: str, collapse_custom_key: bool = False):
    if key_mode == "default":
        return cache

    if arity == "singleton":
        return cache(key=lambda: "singleton-key")

    if collapse_custom_key:
        return cache(key=lambda a, b=2: a + b)
    return cache(key=lambda a, b=2: (a, b))


def _equivalent_key_call_forms(arity: str):
    if arity == "singleton":
        return [((), {}), ((), {})]
    return [
        ((1, 2), {}),
        ((1,), {"b": 2}),
        ((1,), {}),
        ((), {"a": 1, "b": 2}),
        ((), {"b": 2, "a": 1}),
    ]


def _primary_form(arity: str):
    if arity == "singleton":
        return (), {}
    return (1, 2), {}


def _missing_key_form(arity: str):
    if arity == "singleton":
        return (), {}
    return (9, 9), {}


def _build_subject(
    target_kind: str, arity: str, key_mode: str, collapse_custom_key: bool = False
) -> Tuple[Subject, Optional[Subject]]:
    decorator = _cache_decorator_for(arity, key_mode, collapse_custom_key=collapse_custom_key)

    if target_kind == "function":
        counter = {"count": 0}

        if arity == "singleton":

            @decorator
            def fn():
                counter["count"] += 1
                return counter["count"]

        else:

            @decorator
            def fn(a, b=2):
                counter["count"] += 1
                return counter["count"]

        primary = Subject(
            call=lambda *args, **kwargs: fn(*args, **kwargs),
            clear=lambda: fn.clear(),
            remove=lambda *args, **kwargs: fn.remove(*args, **kwargs),
            call_count=lambda: counter["count"],
            descriptor=fn,
        )
        return primary, None

    if target_kind == "instance_method":

        class Worker:
            def __init__(self):
                self.calls = 0

            if arity == "singleton":

                @decorator
                def compute(self):
                    self.calls += 1
                    return self.calls

            else:

                @decorator
                def compute(self, a, b=2):
                    self.calls += 1
                    return self.calls

        first = Worker()
        second = Worker()
        descriptor = Worker.__dict__["compute"]
        primary = Subject(
            call=lambda *args, **kwargs: first.compute(*args, **kwargs),
            clear=lambda: first.compute.clear(),
            remove=lambda *args, **kwargs: first.compute.remove(*args, **kwargs),
            call_count=lambda: first.calls,
            descriptor=descriptor,
        )
        secondary = Subject(
            call=lambda *args, **kwargs: second.compute(*args, **kwargs),
            clear=lambda: second.compute.clear(),
            remove=lambda *args, **kwargs: second.compute.remove(*args, **kwargs),
            call_count=lambda: second.calls,
            descriptor=descriptor,
        )
        return primary, secondary

    if target_kind == "classmethod":

        class Base:
            calls = 0

            if arity == "singleton":

                @decorator
                @classmethod
                def compute(cls):
                    cls.calls += 1
                    return cls.calls

            else:

                @decorator
                @classmethod
                def compute(cls, a, b=2):
                    cls.calls += 1
                    return cls.calls

        class Child(Base):
            calls = 0

        descriptor = Base.__dict__["compute"]
        primary = Subject(
            call=lambda *args, **kwargs: Base.compute(*args, **kwargs),
            clear=lambda: Base.compute.clear(),
            remove=lambda *args, **kwargs: Base.compute.remove(*args, **kwargs),
            call_count=lambda: Base.calls,
            descriptor=descriptor,
        )
        secondary = Subject(
            call=lambda *args, **kwargs: Child.compute(*args, **kwargs),
            clear=lambda: Child.compute.clear(),
            remove=lambda *args, **kwargs: Child.compute.remove(*args, **kwargs),
            call_count=lambda: Child.calls,
            descriptor=descriptor,
        )
        return primary, secondary

    raise ValueError("unknown target kind: {0}".format(target_kind))


@pytest.mark.parametrize("target_kind", TARGET_KINDS)
@pytest.mark.parametrize("arity", ARITIES)
@pytest.mark.parametrize("key_mode", KEY_MODES)
def test_memoization_and_key_canonicalization_matrix(target_kind, arity, key_mode):
    primary, _ = _build_subject(target_kind, arity, key_mode)
    forms = _equivalent_key_call_forms(arity)
    results = [primary.call(*args, **kwargs) for args, kwargs in forms]
    assert results == [1] * len(forms)
    assert primary.call_count() == 1


@pytest.mark.parametrize("target_kind", TARGET_KINDS)
@pytest.mark.parametrize("arity", ARITIES)
@pytest.mark.parametrize("key_mode", KEY_MODES)
def test_clear_matrix(target_kind, arity, key_mode):
    primary, _ = _build_subject(target_kind, arity, key_mode)
    args, kwargs = _primary_form(arity)

    assert primary.call(*args, **kwargs) == 1
    assert primary.call(*args, **kwargs) == 1
    assert primary.call_count() == 1

    primary.clear()

    assert primary.call(*args, **kwargs) == 2
    assert primary.call_count() == 2


@pytest.mark.parametrize("target_kind", TARGET_KINDS)
@pytest.mark.parametrize("arity", ARITIES)
@pytest.mark.parametrize("key_mode", KEY_MODES)
def test_remove_matrix(target_kind, arity, key_mode):
    primary, _ = _build_subject(target_kind, arity, key_mode)

    if arity == "singleton":
        primary.remove()
        assert primary.call_count() == 0

        assert primary.call() == 1
        primary.remove()
        assert primary.call() == 2
        return

    args, kwargs = _primary_form(arity)
    assert primary.call(*args, **kwargs) == 1

    primary.remove(1, b=2)
    assert primary.call(a=1, b=2) == 2

    before = primary.call_count()
    missing_args, missing_kwargs = _missing_key_form(arity)
    primary.remove(*missing_args, **missing_kwargs)
    assert primary.call(*args, **kwargs) == 2
    assert primary.call_count() == before


@pytest.mark.parametrize("target_kind", ("instance_method", "classmethod"))
@pytest.mark.parametrize("arity", ARITIES)
@pytest.mark.parametrize("key_mode", KEY_MODES)
def test_scope_isolation_matrix(target_kind, arity, key_mode):
    primary, secondary = _build_subject(target_kind, arity, key_mode)
    assert secondary is not None

    forms = _equivalent_key_call_forms(arity)
    primary_args, primary_kwargs = forms[0]

    assert primary.call(*primary_args, **primary_kwargs) == 1
    assert secondary.call(*primary_args, **primary_kwargs) == 1
    assert primary.call(*forms[1][0], **forms[1][1]) == 1
    assert secondary.call(*forms[1][0], **forms[1][1]) == 1

    primary.clear()
    assert primary.call(*primary_args, **primary_kwargs) == 2
    assert secondary.call(*primary_args, **primary_kwargs) == 1

    secondary.remove(*primary_args, **primary_kwargs)
    assert secondary.call(*primary_args, **primary_kwargs) == 2
    assert primary.call(*primary_args, **primary_kwargs) == 2


@pytest.mark.parametrize("target_kind", TARGET_KINDS)
def test_custom_key_can_coalesce_distinct_inputs_matrix(target_kind):
    primary, secondary = _build_subject(
        target_kind, arity="dict", key_mode="custom_identity", collapse_custom_key=True
    )

    assert primary.call(1, 2) == 1
    assert primary.call(2, 1) == 1
    assert primary.call_count() == 1

    if secondary is not None:
        assert secondary.call(2, 1) == 1
        assert secondary.call_count() == 1


def _bad_function_dict_key_mismatch():
    @cache(key=lambda a: a)
    def f(a, b=2):
        return a + b

    return f


def _bad_function_singleton_key_mismatch():
    @cache(key=lambda a: a)
    def f():
        return 1

    return f


def _bad_instance_dict_key_mismatch():
    class C:
        @cache(key=lambda self, a, b=2: (a, b))
        def f(self, a, b=2):
            return a + b

    return C


def _bad_instance_singleton_key_mismatch():
    class C:
        @cache(key=lambda self: "x")
        def f(self):
            return 1

    return C


def _bad_classmethod_dict_key_mismatch():
    class C:
        @cache(key=lambda cls, a, b=2: (a, b))
        @classmethod
        def f(cls, a, b=2):
            return a + b

    return C


def _bad_classmethod_singleton_key_mismatch():
    class C:
        @cache(key=lambda cls: "x")
        @classmethod
        def f(cls):
            return 1

    return C


@pytest.mark.parametrize(
    "factory",
    [
        _bad_function_dict_key_mismatch,
        _bad_function_singleton_key_mismatch,
        _bad_instance_dict_key_mismatch,
        _bad_instance_singleton_key_mismatch,
        _bad_classmethod_dict_key_mismatch,
        _bad_classmethod_singleton_key_mismatch,
    ],
)
def test_key_signature_disagreement_raises_type_error(factory):
    with pytest.raises(TypeError):
        factory()


def test_instance_cache_state_released_after_gc():
    class C:
        @cache
        def f(self, x):
            return x

    c = C()
    assert c.f(1) == 1

    descriptor = C.__dict__["f"]
    assert len(descriptor._instance_states) == 1

    ref = weakref.ref(c)
    del c
    gc.collect()

    assert ref() is None
    assert len(descriptor._instance_states) == 0


@pytest.mark.parametrize("target_kind", TARGET_KINDS)
@pytest.mark.parametrize("arity", ARITIES)
def test_same_key_concurrency_coalesces_matrix(target_kind, arity):
    calls = {"count": 0}
    entered = threading.Event()
    release = threading.Event()

    decorator = cache

    if target_kind == "function":
        if arity == "singleton":

            @decorator
            def f():
                calls["count"] += 1
                entered.set()
                release.wait(timeout=2)
                return calls["count"]

            invoke = lambda *args, **kwargs: f(*args, **kwargs)
            first_form = ((), {})
            second_form = ((), {})
        else:

            @decorator
            def f(a, b=2):
                calls["count"] += 1
                entered.set()
                release.wait(timeout=2)
                return calls["count"]

            invoke = lambda *args, **kwargs: f(*args, **kwargs)
            first_form = ((1, 2), {})
            second_form = ((1,), {"b": 2})

    elif target_kind == "instance_method":

        class C:
            if arity == "singleton":

                @decorator
                def f(self):
                    calls["count"] += 1
                    entered.set()
                    release.wait(timeout=2)
                    return calls["count"]

            else:

                @decorator
                def f(self, a, b=2):
                    calls["count"] += 1
                    entered.set()
                    release.wait(timeout=2)
                    return calls["count"]

        inst = C()
        invoke = lambda *args, **kwargs: inst.f(*args, **kwargs)
        first_form = ((), {}) if arity == "singleton" else ((1, 2), {})
        second_form = ((), {}) if arity == "singleton" else ((1,), {"b": 2})

    else:

        class C:
            if arity == "singleton":

                @decorator
                @classmethod
                def f(cls):
                    calls["count"] += 1
                    entered.set()
                    release.wait(timeout=2)
                    return calls["count"]

            else:

                @decorator
                @classmethod
                def f(cls, a, b=2):
                    calls["count"] += 1
                    entered.set()
                    release.wait(timeout=2)
                    return calls["count"]

        invoke = lambda *args, **kwargs: C.f(*args, **kwargs)
        first_form = ((), {}) if arity == "singleton" else ((1, 2), {})
        second_form = ((), {}) if arity == "singleton" else ((1,), {"b": 2})

    results = []

    def worker(args, kwargs):
        results.append(invoke(*args, **kwargs))

    t1 = threading.Thread(target=worker, args=first_form)
    t2 = threading.Thread(target=worker, args=second_form)
    t1.start()
    assert entered.wait(timeout=1)
    t2.start()
    time.sleep(0.05)
    assert calls["count"] == 1

    release.set()
    t1.join(timeout=2)
    t2.join(timeout=2)
    assert not t1.is_alive()
    assert not t2.is_alive()
    assert calls["count"] == 1
    assert sorted(results) == [1, 1]


@pytest.mark.parametrize("target_kind", TARGET_KINDS)
@pytest.mark.parametrize("arity", ARITIES)
def test_do_not_cache_waiter_retries_matrix(target_kind, arity):
    calls = {"count": 0}
    entered = threading.Event()
    release = threading.Event()

    decorator = cache

    if target_kind == "function":
        if arity == "singleton":

            @decorator
            def f():
                calls["count"] += 1
                if calls["count"] == 1:
                    entered.set()
                    release.wait(timeout=2)
                    raise DoNotCache("temporary")
                return calls["count"]

            invoke = lambda *args, **kwargs: f(*args, **kwargs)
            form = ((), {})
        else:

            @decorator
            def f(a, b=2):
                calls["count"] += 1
                if calls["count"] == 1:
                    entered.set()
                    release.wait(timeout=2)
                    raise DoNotCache("temporary")
                return calls["count"]

            invoke = lambda *args, **kwargs: f(*args, **kwargs)
            form = ((1, 2), {})

    elif target_kind == "instance_method":

        class C:
            if arity == "singleton":

                @decorator
                def f(self):
                    calls["count"] += 1
                    if calls["count"] == 1:
                        entered.set()
                        release.wait(timeout=2)
                        raise DoNotCache("temporary")
                    return calls["count"]

            else:

                @decorator
                def f(self, a, b=2):
                    calls["count"] += 1
                    if calls["count"] == 1:
                        entered.set()
                        release.wait(timeout=2)
                        raise DoNotCache("temporary")
                    return calls["count"]

        inst = C()
        invoke = lambda *args, **kwargs: inst.f(*args, **kwargs)
        form = ((), {}) if arity == "singleton" else ((1, 2), {})

    else:

        class C:
            if arity == "singleton":

                @decorator
                @classmethod
                def f(cls):
                    calls["count"] += 1
                    if calls["count"] == 1:
                        entered.set()
                        release.wait(timeout=2)
                        raise DoNotCache("temporary")
                    return calls["count"]

            else:

                @decorator
                @classmethod
                def f(cls, a, b=2):
                    calls["count"] += 1
                    if calls["count"] == 1:
                        entered.set()
                        release.wait(timeout=2)
                        raise DoNotCache("temporary")
                    return calls["count"]

        invoke = lambda *args, **kwargs: C.f(*args, **kwargs)
        form = ((), {}) if arity == "singleton" else ((1, 2), {})

    results = {}

    def first():
        args, kwargs = form
        results["first"] = invoke(*args, **kwargs)

    def second():
        args, kwargs = form
        results["second"] = invoke(*args, **kwargs)

    t1 = threading.Thread(target=first)
    t2 = threading.Thread(target=second)
    t1.start()
    assert entered.wait(timeout=1)
    t2.start()
    time.sleep(0.05)
    release.set()

    t1.join(timeout=2)
    t2.join(timeout=2)
    assert not t1.is_alive()
    assert not t2.is_alive()

    assert results["first"] == "temporary"
    assert results["second"] == 2
    assert calls["count"] == 2

    args, kwargs = form
    assert invoke(*args, **kwargs) == 2
    assert calls["count"] == 2


@pytest.mark.parametrize("target_kind", TARGET_KINDS)
@pytest.mark.parametrize("arity", ARITIES)
def test_non_do_not_cache_exception_not_cached_matrix(target_kind, arity):
    calls = {"count": 0}
    entered = threading.Event()
    release = threading.Event()
    exceptions = []

    class Boom(Exception):
        pass

    decorator = cache

    if target_kind == "function":
        if arity == "singleton":

            @decorator
            def f():
                calls["count"] += 1
                if calls["count"] == 1:
                    entered.set()
                    release.wait(timeout=2)
                    raise Boom("boom")
                return calls["count"]

            invoke = lambda *args, **kwargs: f(*args, **kwargs)
            form = ((), {})
        else:

            @decorator
            def f(a, b=2):
                calls["count"] += 1
                if calls["count"] == 1:
                    entered.set()
                    release.wait(timeout=2)
                    raise Boom("boom")
                return calls["count"]

            invoke = lambda *args, **kwargs: f(*args, **kwargs)
            form = ((1, 2), {})

    elif target_kind == "instance_method":

        class C:
            if arity == "singleton":

                @decorator
                def f(self):
                    calls["count"] += 1
                    if calls["count"] == 1:
                        entered.set()
                        release.wait(timeout=2)
                        raise Boom("boom")
                    return calls["count"]

            else:

                @decorator
                def f(self, a, b=2):
                    calls["count"] += 1
                    if calls["count"] == 1:
                        entered.set()
                        release.wait(timeout=2)
                        raise Boom("boom")
                    return calls["count"]

        inst = C()
        invoke = lambda *args, **kwargs: inst.f(*args, **kwargs)
        form = ((), {}) if arity == "singleton" else ((1, 2), {})

    else:

        class C:
            if arity == "singleton":

                @decorator
                @classmethod
                def f(cls):
                    calls["count"] += 1
                    if calls["count"] == 1:
                        entered.set()
                        release.wait(timeout=2)
                        raise Boom("boom")
                    return calls["count"]

            else:

                @decorator
                @classmethod
                def f(cls, a, b=2):
                    calls["count"] += 1
                    if calls["count"] == 1:
                        entered.set()
                        release.wait(timeout=2)
                        raise Boom("boom")
                    return calls["count"]

        invoke = lambda *args, **kwargs: C.f(*args, **kwargs)
        form = ((), {}) if arity == "singleton" else ((1, 2), {})

    def worker():
        try:
            args, kwargs = form
            invoke(*args, **kwargs)
        except Exception as exc:
            exceptions.append(exc)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    assert entered.wait(timeout=1)
    t2.start()
    time.sleep(0.05)
    release.set()

    t1.join(timeout=2)
    t2.join(timeout=2)
    assert not t1.is_alive()
    assert not t2.is_alive()
    assert len(exceptions) == 2
    assert all(isinstance(exc, Boom) for exc in exceptions)
    assert calls["count"] == 1

    args, kwargs = form
    assert invoke(*args, **kwargs) == 2
    assert calls["count"] == 2
