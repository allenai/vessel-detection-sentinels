import math
import subprocess
import typing as t

from osgeo import gdal

web_mercator_m = 2 * math.pi * 6378137


class ImageInfo(object):
    """Container for image metadata."""

    def __init__(self, width, height, bounds, column, row, zoom, projection=None):
        self.width = width
        self.height = height
        self.bounds = bounds
        self.column = column
        self.row = row
        self.zoom = zoom
        self.projection = projection


def warp(
    image, in_path: str, out_path: str, projection: t.Optional[str] = None
) -> ImageInfo:
    """Warp a raw image to the specified projection.

    Parameters
    ----------
    image: src.data.retrieve.RetrieveImage
        Image class to warp.

    in_path: str
        Path to input raster geotiff.

    out_path: str
        Path to output raster geotiff.

    projection: Optional[str]
        Desired output projection (e.g. 'epsg:3857' for pseudo-mercator).
        If None or '', the files are converted without warping.

    Returns
    -------
     : ImageInfo
        Class containing warped image information.
    """

    def get_pixel_size():
        """Returns pixel size provided in metadata, or if unavailable, computes
        pixel size from the input raster.
        """
        if image.pixel_size:
            return image.pixel_size

        raster = gdal.Open(in_path)
        geo_transform = raster.GetGeoTransform()
        pixel_size_x = geo_transform[1]
        pixel_size_y = -geo_transform[5]
        return min(pixel_size_x, pixel_size_y)

    if not projection:
        stdout = subprocess.check_output(["gdalwarp", in_path, out_path, "-overwrite"])
        return get_image_info(out_path)
    elif projection == "epsg:3857":
        # Determine desired output resolution.
        # We scale up to the zoom level just above the native resolution.
        # This takes up more space than needed, but ensures we don't "lose" any resolution.
        in_pixel_size = get_pixel_size()
        out_pixel_size = None
        out_zoom = None

        for zoom in range(20):
            zoom_pixel_size = web_mercator_m / 512 / (2**zoom)
            out_pixel_size = zoom_pixel_size
            out_zoom = zoom
            if out_pixel_size < in_pixel_size * 1.1:
                break

        # Warp the input image.
        stdout = subprocess.check_output(
            [
                "gdalwarp",
                "-r",
                "bilinear",
                "-t_srs",
                projection,
                "-tr",
                str(out_pixel_size),
                str(out_pixel_size),
                in_path,
                out_path,
                "-overwrite",
            ]
        )
        # logger.debug(stdout)

        raster = gdal.Open(out_path)
        geo_transform = raster.GetGeoTransform()
        offset_x = geo_transform[0] + web_mercator_m / 2
        offset_y = web_mercator_m - (geo_transform[3] + web_mercator_m / 2)
        offset_x /= out_pixel_size
        offset_y /= out_pixel_size

        return ImageInfo(
            width=raster.RasterXSize,
            height=raster.RasterYSize,
            bounds=get_wgs84_bounds(raster),
            column=int(offset_x),
            row=int(offset_y),
            zoom=out_zoom,
            projection=projection,
        )

    else:
        raise Exception("unknown projection {}".format(projection))


def get_image_info(fname: str) -> ImageInfo:
    """Get image information, assuming no projection.

    Parameters
    ----------
    fname: str
        Path to image.

    Returns
    -------
    : ImageInfo
        Image metadata.
    """
    raster = gdal.Open(fname)
    return ImageInfo(
        width=raster.RasterXSize,
        height=raster.RasterYSize,
        bounds=get_wgs84_bounds(raster),
        column=0,
        row=0,
        zoom=0,
    )


def get_wgs84_bounds(raster) -> dict:
    """Get WGS84 bounding box given gdal raster.

    Parameters
    ----------
    raster: gdal.raster
        Gdal image raster.

    Returns
    -------
    : dict
        Dictionary containing minimal and maximal lat/lon coords of raster.
    """
    transformer = gdal.Transformer(raster, None, ["DST_SRS=WGS84"])
    _, p1 = transformer.TransformPoint(0, 0, 0, 0)
    _, p2 = transformer.TransformPoint(0, 0, raster.RasterYSize, 0)
    _, p3 = transformer.TransformPoint(0, raster.RasterXSize, 0, 0)
    _, p4 = transformer.TransformPoint(0, raster.RasterXSize, raster.RasterYSize, 0)
    points = [p1, p2, p3, p4]
    return {
        "Min": {
            "Lon": min([p[0] for p in points]),
            "Lat": min([p[1] for p in points]),
        },
        "Max": {
            "Lon": max([p[0] for p in points]),
            "Lat": max([p[1] for p in points]),
        },
    }
