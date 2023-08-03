from functools import partial

import numpy as np
import pandas as pd

from src.utils.geom import extremal_bounds


def filter_detection(row: pd.Series, loc_df: str, default_width_m: float = 100) -> bool:
    """Return True if a row should be filtered due to overlap with loc_df.

    Parameters
    ----------
    row: pandas.Series
        Row of a pandas series with lat and lon columns, whose rows are detections.

    loc_df: pandas.DataFrame
        Dataframe with columns "lon", "lat" and "width_m" specifying
        locations of undesired detection centers, and the (assumed to be square)
        extent of the undesired object.

    default_width_m: float
        A default width to assign to locations in loc_df if the width_m field is empty.


    Returns
    -------
    : bool
        True if detection should be filtered due to overlap.
    """

    detect_lon = row.lon
    detect_lat = row.lat

    for _, r in loc_df.iterrows():
        lon = r.lon
        lat = r.lat
        width_m = r.width_m
        if np.isnan(width_m):
            width_m = default_width_m

        min_lon, min_lat, max_lon, max_lat = extremal_bounds(lon, lat, width_m)

        if (
            (detect_lon >= min_lon)
            and (detect_lon <= max_lon)
            and (detect_lat >= min_lat)
            and (detect_lat <= max_lat)
        ):
            return True

    return False


def filter_out_locs(
    df: pd.DataFrame, loc_path: str, default_width_m: float = 100
) -> pd.DataFrame:
    """Filter out rows in df overlapping locations specified in loc_path.

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe of predictions containing, at the least, columns
        named "lon" and "lat".

    loc_path: str
        Path to a csv with columns "lon", "lat" and "width_m" specifying
        locations of undesired detection centers, and the (assumed to be square)
        extent of the undesired object.

    Returns
    -------
    filtered_df: pandas.DatadFrame
        A sub-dataframe of df with the rows corresponding to detections
        overlapping the undesired object extents removed.
    """
    loc_df = pd.read_csv(loc_path)

    remove_detection = partial(
        filter_detection, loc_df=loc_df, default_width_m=default_width_m
    )

    filtered_df = df[~df.apply(remove_detection, axis=1)]

    return filtered_df
