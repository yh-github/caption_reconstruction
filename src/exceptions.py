class UserFacingError(Exception):
    """
    A base class for all exceptions that are considered user-correctable
    and should not produce a full stack trace.
    """
    pass
