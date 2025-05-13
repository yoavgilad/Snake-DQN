from datetime import datetime, timedelta


def now() -> datetime:
    """

    :return: Current datetime rounded-up to the next full second
    """
    time_now = datetime.now()
    return time_now + timedelta(microseconds=1_000_000 - time_now.microsecond)


def add_time(delta: timedelta) -> datetime:
    """

    :param delta: A time period
    :return: A future datetime, which is the sum of now and the given time delta
    """
    return now() + delta


def runtime(start: datetime) -> timedelta:
    """

    :param start: A past timestamp
    :return: The duration since the given timestamp
    """
    return now() - start
