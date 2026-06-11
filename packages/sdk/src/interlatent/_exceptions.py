class InterlatentError(Exception):
    """Base SDK exception."""


class APIError(InterlatentError):
    """HTTP/API error."""

    def __init__(self, message: str, *, status_code: int | None = None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class AuthenticationError(APIError):
    """Authentication/authorization failure."""


class NotFoundError(APIError):
    """Resource not found."""
