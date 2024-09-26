from ajprax.require import require, RequirementException


def test_require():
    def raises(exception, s):
        def test(*a, **kw):
            try:
                require(*a, **kw)
                assert False, "should have raised"
            except exception as e:
                assert str(e) == s

        return test

    require(True)
    require(True, "message")
    require(True, key="value")
    require(True, "message", key="value")
    require(True, _exc=ValueError)
    require(True, "message", _exc=ValueError, key="value")

    raises(RequirementException, "")(False)
    raises(RequirementException, "message")(False, "message")
    raises(RequirementException, "key=value")(False, key="value")
    raises(RequirementException, "message key=value")(False, "message", key="value")
    raises(ValueError, "")(False, _exc=ValueError)
    raises(ValueError, "message key=value")(False, "message", _exc=ValueError, key="value")
