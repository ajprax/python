class RequirementException(Exception):
    pass


def require(
    condition: object,
    message: str = "",
    _exc: type[Exception] = RequirementException,
    **kwargs: object,
) -> None:
    """
    Similar to `assert condition, message` but allows controlling the exception type and structured keyword arguments
    """
    if not condition:
        if message and kwargs:
            message += " "
        message += " ".join(f"{k}={v}" for k, v in kwargs.items())
        raise _exc(message)
