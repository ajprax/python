class RequirementException(Exception):
    pass


def require(condition, message="", _exc=RequirementException, **kwargs):
    """
    Similar to `assert condition, message` but allows controlling the exception type and structured keyword arguments
    """
    if not condition:
        if message and kwargs:
            message += " "
        message += " ".join(f"{k}={v}" for k, v in kwargs.items())
        raise _exc(message)
