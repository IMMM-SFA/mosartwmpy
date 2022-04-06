from datetime import date, datetime


def get_epiweek_from_datetime(dt: datetime) -> int:
    """Calculate the epiweek number from a datetime

    Args:
        dt (datetime): the datetime object to calculate the epiweek for

    Returns:
        int: the epiweek number

    """
    year = dt.year
    dt_ordinal = dt.toordinal()
    year_start_ordinal = _get_year_start_ordinal(year)
    week = (dt_ordinal - year_start_ordinal) // 7
    if week < 0:
        year = year - 1
        year_start_ordinal = _get_year_start_ordinal(year)
        week = (dt_ordinal - year_start_ordinal) // 7
    elif week >= 52:
        year_start_ordinal = _get_year_start_ordinal(year + 1)
        if dt_ordinal > year_start_ordinal:
            year = year + 1
            week = 0
    week = week + 1
    return week


def _get_year_start_ordinal(y):
    year_start = date(y, 1, 1)
    year_start_weekday = year_start.weekday()
    year_start_ordinal = year_start.toordinal() - year_start_weekday - 1
    if year_start_weekday > 2:
        return year_start_ordinal + 7
    return year_start_ordinal