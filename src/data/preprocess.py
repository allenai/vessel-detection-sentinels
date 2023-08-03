import os
import os.path

import numpy as np
import skimage.exposure
import skimage.io

chip_size = 512


def pad_chip(image_data: dict, im: np.ndarray) -> np.ndarray:
    """Pad an image chip.

    Parameters
    ----------
    image_data: dict
        Image metadata.

    im: np.ndarray
        Raw image data as numpy array.

    Returns
    -------
     : np.ndarray
        Padded image.
    """
    start_tile = [
        image_data["Column"] // chip_size,
        image_data["Row"] // chip_size,
    ]
    end_tile = [
        (image_data["Column"] + im.shape[1]) // chip_size,
        (image_data["Row"] + im.shape[0]) // chip_size,
    ]
    col_padding = (
        image_data["Column"] - start_tile[0] * chip_size,
        (end_tile[0] + 1) * chip_size - (image_data["Column"] + im.shape[1]),
    )
    row_padding = (
        image_data["Row"] - start_tile[1] * chip_size,
        (end_tile[1] + 1) * chip_size - (image_data["Row"] + im.shape[0]),
    )
    padding = [row_padding, col_padding]
    while len(padding) < len(im.shape):
        padding += [(0, 0)]
    return np.pad(im, padding), start_tile


def preprocess(
    image_data: dict,
    fname: str,
    channel: dict,
    roi: dict = None,
    dest_dir: str = "../data",
) -> None:
    """Preprocess a Sentinel-1, Sentinel-2 or Landsat8 image.

    Performs the following operations on the raw imagery:

    Sentinel-1:
    (1) Cast to uint8.
    (2) Pad.
    (3) Crop out tiles.
    (4) Save all tiles associated with an image in preprocess output dir.

    Parameters
    ----------
    image_data: dict
        Metadata dict for raw image file to preprocess.

    fname: str
        Path to raw image file to preprocess.

    channel: dict

    roi: dict
        Dict representation of ROI object as in sqlite metadata database.

    dest_dir: str
        Path to directory in which preprocessed data will get stored.

    Returns
    -------
    : None
    """
    out_dir = os.path.join(dest_dir, "preprocess")
    im = skimage.io.imread(fname)
    out_path = os.path.join(out_dir, image_data["UUID"])
    channel_name = channel["Name"]
    os.makedirs(os.path.join(out_path, channel_name), exist_ok=True)

    # Normalize the image to 0-255.
    if roi and roi["CatalogName"] == "sentinel2":
        if channel_name[0] == "b" and len(channel_name) == 3:
            im = im // 4
    elif roi and roi["CatalogName"] == "landsat8":
        # Landsat-8 seems to need per-band normalization to get good outputs.
        im = skimage.exposure.equalize_hist(im)
        band_min, band_max = im.min(), im.max()
        im = 255 * (im - band_min) / (band_max - band_min)
    elif roi and roi["CatalogName"] == "naip":
        # Get rid of the NIR channel if it exists, we just want RGB images for now (for consistency).
        im = im[:, :, 0:3]

    im = np.clip(im, 0, 255).astype("uint8")

    im, start_tile = pad_chip(image_data, im)

    for i in range(0, im.shape[1] // chip_size):
        for j in range(0, im.shape[0] // chip_size):
            crop = im[
                j * chip_size : (j + 1) * chip_size, i * chip_size : (i + 1) * chip_size
            ]
            if crop.max() == 0:
                continue
            skimage.io.imsave(
                os.path.join(
                    out_path,
                    channel_name,
                    "{}_{}.png".format(start_tile[0] + i, start_tile[1] + j),
                ),
                crop,
                check_contrast=False,
            )
