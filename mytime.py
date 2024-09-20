import time
from datetime import datetime, timedelta


def wait(seconds: float) -> None:
    time.sleep(seconds)


def now() -> datetime:
    time_now = datetime.now()
    return time_now + timedelta(microseconds=1_000_000 - time_now.microsecond)


def add_time(delta: timedelta) -> datetime:
    return now() + delta


def runtime(start: datetime) -> timedelta:
    return now() - start
