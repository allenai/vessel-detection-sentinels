
import math

import typing as t

R_EARTH_M = 6371000
LAT_RANGE = 180
LON_RANGE = 360
DEGREES_LAT_PER_METER = LAT_RANGE / (math.pi * R_EARTH_M)


def degrees_lon_per_meter(lat: float) -> float:
    """Approximate degrees of longitude per meter at specified latitude.

    Note: This is an infinitesimal approximation, and breaks down near poles.

    Parameters
    ----------
    lat: float
        Latitude.

    Returns
    -------
    dlpm: float
        Degrees longitude per meter.
    """
    lat_circumference = 2 * math.pi * R_EARTH_M * math.cos(lat)
    return LON_RANGE / lat_circumference


def extremal_bounds(
    center_lon: float, center_lat: float, width_m: float
) -> t.Tuple[float, float, float, float]:
    """Calculate minimal and maximal lon/lat coordinates given a center coordinate and a desired square width.

    Parameters
    ----------
    center_lon: float
        Longitude for center of object.

    center_lat: float
        Latitude for center of object.

    width_m: float
        Width of object in meters.

    Returns
    -------
    min_lon: float
        Minimum longitude of object square.

    min_lat: float
        Minimum latitude of object square.

    max_lon: flaot
        Maximum longitude of object square.

    max_lat:
        Minimum latitude of object square.
    """
    extend_by = width_m / 2
    dlonpm = degrees_lon_per_meter(center_lat)

    min_lat, min_lon = center_lat - (DEGREES_LAT_PER_METER * extend_by), center_lon - (
        dlonpm * extend_by
    )
    max_lat, max_lon = center_lat + (DEGREES_LAT_PER_METER * extend_by), center_lon + (
        dlonpm * extend_by
    )

    return min_lon, min_lat, max_lon, max_lat
