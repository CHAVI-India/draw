import threading
from typing import Any, Callable, Optional, TypeVar, cast

VoidFunction = TypeVar("VoidFunction", bound=Callable[..., None])


class Debouncer:
    def __init__(self, f: Callable[..., Any], interval: float):
        self.f = f
        self.interval = interval
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    def __call__(self, *args, **kwargs) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.interval, self.f, args, kwargs)
            self._timer.start()


def debounce(interval: float):
    """
    Wait `interval` seconds before calling `f`, and cancel if called again.
    The decorated function will return None immediately,
    ignoring the delayed return value of `f`.
    """

    def decorator(f: VoidFunction) -> VoidFunction:
        if interval <= 0:
            return f
        return cast(VoidFunction, Debouncer(f, interval))

    return decorator
