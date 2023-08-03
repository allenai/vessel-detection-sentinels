import os.path

import numpy
import skimage.io


def load_window(
    image_uuid: str,
    channel: dict,
    column: int,
    row: int,
    width: int,
    height: int,
    chip_size: int = 512,
    preprocess_dir="../data/preprocess",
):
    """Stitch together an image from multiple tiles.

    Parameters
    ----------
    image_uuid: str
        UUID for image to load.

    channel: dict

    column: int

    row: int

    width: int

    height: int

    chip_size: int

    preprocess_dir: str
        Directory housing preprocessed images (warped images, broken into tiles).

    Returns
    -------
    im: numpy.ndarray
        Total image window.
    """
    # Initialize array for output.
    # We need different cases for single-channel image and multi-channel image.
    if channel["Count"] == 1:
        im = numpy.zeros((height, width), dtype=numpy.uint8)
    else:
        im = numpy.zeros((height, width, channel["Count"]), dtype=numpy.uint8)

    # Load tiles one at a time.
    start_tile = (column // chip_size, row // chip_size)
    end_tile = ((column + width - 1) // chip_size, (row + height - 1) // chip_size)
    for i in range(start_tile[0], end_tile[0] + 1):
        for j in range(start_tile[1], end_tile[1] + 1):
            fname = os.path.join(
                preprocess_dir,
                "{}/{}/{}_{}.png".format(image_uuid, channel["Name"], i, j),
            )
            if not os.path.exists(fname):
                continue

            cur_im = skimage.io.imread(fname)
            cur_col_off = chip_size * i
            cur_row_off = chip_size * j

            src_col_offset = max(column - cur_col_off, 0)
            src_row_offset = max(row - cur_row_off, 0)
            dst_col_offset = max(cur_col_off - column, 0)
            dst_row_offset = max(cur_row_off - row, 0)
            col_overlap = min(cur_im.shape[1] - src_col_offset, width - dst_col_offset)
            row_overlap = min(cur_im.shape[0] - src_row_offset, height - dst_row_offset)
            im[
                dst_row_offset: dst_row_offset + row_overlap,
                dst_col_offset: dst_col_offset + col_overlap,
            ] = cur_im[
                src_row_offset: src_row_offset + row_overlap,
                src_col_offset: src_col_offset + col_overlap,
            ]
    return im
