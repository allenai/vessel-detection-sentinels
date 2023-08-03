import glob
import os
import typing as t
from datetime import datetime

import pandas as pd
from osgeo import gdal, osr

from src.data.image import InvalidDataError


def get_extent(ds: gdal.Dataset) -> t.Tuple[t.Tuple[float]]:
    """Return list of corner coordinates from a gdal Dataset.

    Return the corner coordinates of the minimal rectangle 
    circumscribing a gdal dataset. Corner coordinates
    are returned with the ordering
    (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin).

    Parameters
    ----------
    ds: gdal.Dataset
        Input gdal dataset

    Returns
    -------
    : tuple[tuple[float]]
        Corner coordinates of the minimal circumscribed
        rectangle surrounding the input dataset.
    """
    xmin, xpixel, _, ymax, _, ypixel = ds.GetGeoTransform()
    width, height = ds.RasterXSize, ds.RasterYSize
    xmax = xmin + width * xpixel
    ymin = ymax + height * ypixel

    return (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)


def reproject_coords(coords: t.List[t.List[float]],
                     src_srs: osr.SpatialReference, tgt_srs: osr.SpatialReference) -> t.List[t.List[float]]:
    """Reproject a list of coordinates from one coordinate system to another.

    Parameters
    ----------
    coords: list[list[float]]
        Original coordinates in source coordinate system.

    src_srs: osr.SpatialReference
        A source SpatialReference object specifying coordinate system.

    tgt_srs: osr.SpatialReference
        A target SpatialReference object specifying coordinate system. 

    Returns
    -------
    trans_coords: list[list[float]]
        Transformed coordinates in target coordinate system.
    """
    trans_coords = []
    transform = osr.CoordinateTransformation(src_srs, tgt_srs)
    for x, y in coords:
        x, y, z = transform.TransformPoint(x, y)
        trans_coords.append([x, y])
    return trans_coords


def get_frame_coords(raster_path: str, catalog: str) -> t.List[t.List[float]]:
    """Get list of (lon, lat) frame boundary coordinates.

    Returned list specifies a _closed_ polygon (i.e. starting
    coordinates are required to be the same as ending coordinates).

    Parameters
    ----------
    raster_path: str
        Path to a raw satellite image raster.

    catalog: str
        Specifies the satellite catalog from which the
        passed scene_dir is drawn. Only currently supported
        values are "sentinel1" and "sentinel2".

    Returns
    -------
    coordinates: list[list[float]]
        List of (lon, lat) frame boundary coordinates.
    """
    ds = gdal.Open(raster_path)

    if catalog == "sentinel2":
        ext = get_extent(ds)
        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt(ds.GetProjection())
        tgt_srs = osr.SpatialReference()
        tgt_srs.ImportFromEPSG(4326)
        geo_ext = reproject_coords(ext, src_srs, tgt_srs)
        geo_ext_lon_first = [[p[1], p[0]] for p in geo_ext]

    if catalog == "sentinel1":
        transformer = gdal.Transformer(ds, None, ["DST_SRS=WGS84"])
        _, p1 = transformer.TransformPoint(0, 0, 0, 0)
        _, p2 = transformer.TransformPoint(0, 0, ds.RasterYSize, 0)
        _, p3 = transformer.TransformPoint(0, ds.RasterXSize, 0, 0)
        _, p4 = transformer.TransformPoint(0, ds.RasterXSize, ds.RasterYSize, 0)
        geo_ext_lon_first = [p1, p2, p3, p4]
        geo_ext_lon_first = [p[0:2] for p in geo_ext_lon_first]

    # Close loop:
    geo_ext_lon_first.append(geo_ext_lon_first[0])
    coordinates = geo_ext_lon_first

    return coordinates


def get_raster_path_base(scene_dir_path: str, catalog: str) -> str:
    """Retreives path to one sample raster from a satellite scene dir.

    Parameters
    ----------
    scene_dir_path: str
        Path to a satellite scene dir (Sentinel-1 or Sentinel-2).

    catalog: str
        Specifies the satellite catalog from which the
        passed scene_dir is drawn. Only currently supported
        values are "sentinel1" and "sentinel2".

    Returns
    -------
    path: str
        Path to a raster file within the specified satellite scene
        dir.
    """
    if catalog == "sentinel2":
        coi_abbrev = "TCI"
        path_pattern = os.path.join(scene_dir_path, f"GRANULE/*/IMG_DATA/*_{coi_abbrev}.jp2")
        paths = glob.glob(path_pattern)
        if len(paths) == 1:
            path = paths[0]
        else:
            raise InvalidDataError(
                f"Raw Sentinel-2 data must be of L1C product type, and contain channel={coi_abbrev}.\n"
                f"Did not find a unique path using the pattern: {path_pattern}"
            )
    if catalog == "sentinel1":
        channel = "vh"
        measurement_path = os.path.join(scene_dir_path, "measurement")
        fnames = {
            fname.split("-")[3]: fname for fname in os.listdir(measurement_path)
        }
        path = os.path.join(measurement_path, fnames[channel])
        if not os.path.exists(path):
            raise InvalidDataError(
                f"Raw Sentinel-1 data does not contain required polarization channel={channel}.\n"
            )
    return path


def construct_scene(scene_dir_path: str, catalog: str) -> dict:
    """Constructs a diu-correlation-api compatible scene dict.

    Given a path to a Sentinel-1 or Sentinel-2 scene, this
    constructs a dictionary of scene metadata that can be
    used with the diu-correlation-api service.

    Parameters
    ----------
    scene_dir_path: str
        Path to an uncompressed raw satellite product dir.

        E.g. a Sentinel-1 SAFE dir.

    catalog: str
        Specification of satellite frame type.

        Currently supported catalogs are "sentinel1" and "sentinel2".
        For "sentinel1", only Level-1 GRD products are supported.
        For "sentinel2", only MSI L1C products are supported.


    Returns
    -------
    scene_obj: dict
        A diu-correlation-api compatible dictionary encoding
        the satellite scene metadata.
    """
    raster_path = get_raster_path_base(scene_dir_path, catalog)
    frame_coordinates = get_frame_coords(raster_path, catalog)
    if catalog == "sentinel2":
        time_loc = 2
    if catalog == "sentinel1":
        time_loc = 4
    frame_timestamp = os.path.basename(scene_dir_path).split("_")[time_loc]
    frame_timestamp = datetime.strptime(frame_timestamp, "%Y%m%dT%H%M%S")
    frame_timestamp = frame_timestamp.isoformat()

    scene_obj = {"ts": frame_timestamp, "polygon_points": frame_coordinates}

    return scene_obj


def format_detections(detections_path: str, scene_obj: dict) -> t.List[dict]:
    """Constructs diu-correlation-api compatible detection object from csv.

    Given a path to a Sentinel-1 or Sentinel-2 model output csv, this
    constructs a dictionary of detection metadata that can be
    used with the diu-correlation-api service.

    Parameters
    ----------
    detections_path: str
        Path to a a predictions csv.

    scene_obj: str
        A diu-correlation-api compatible dictionary encoding
        the satellite scene metadata.


    Returns
    -------
    detection_arr: list[dict]
        A diu-correlation-api compatible dictionary encoding detections
        from a scene.
    """
    detection_df = pd.read_csv(detections_path)

    detection_arr = detection_df.apply(
        lambda x: {"id": x.detect_id, "ts": scene_obj["ts"],
                   "lon": x.lon, "lat": x.lat}, axis=1).tolist()

    return detection_arr
