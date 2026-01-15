"""Colorful exceptions"""

# Definition of handy colours for printing
_default = "\x1b[00m"
_green = "\x1b[01;32m"
_red = "\x1b[01;31m"


class InvalidOperationError(Exception):
    """Exception class for meaningless operations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        """String representation."""
        return _red + self.message + _default
