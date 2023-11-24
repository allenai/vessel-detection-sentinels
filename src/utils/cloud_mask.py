import os

import cv2
import numpy as np
import rasterio
from s2cloudless import S2PixelCloudDetector


class S2CloudMask:
    def __init__(self, directory_path):
        self.img_array = self.get_img_data_as_array(directory_path)

    def resize_image(self, input_image: np.array, output_size: tuple):
        """Resize a numpy array image to a specific size using nearest neighbor interpolation.
        Args:
            input_image (np.array): The input image represented as a numpy array.
            output_size (tuple): The desired output size as a tuple (width, height).
        Returns:
            np.array: The resized image as a numpy array.
        """
        resized_image = cv2.resize(
            input_image, output_size, interpolation=cv2.INTER_NEAREST
        )
        return resized_image

    def get_img_data_as_array(self, directory_path):
        """
        Convert Sentinel-2 image bands in the nested IMG_DATA directory of a SAFE folder to a stacked numpy array and a list of channel names.
        Parameters:
        - directory_path: string, path to the .SAFE directory of the Sentinel-2 image.
        Returns:
        - Stacked numpy array of shape (height, width, num_bands) and a list of channel names.
        """

        # Locate the nested IMG_DATA directory
        img_data_path = None
        print(directory_path)
        for subdir, _, _ in os.walk(directory_path):
            if os.path.basename(subdir) == "IMG_DATA":
                img_data_path = subdir

                break
        print(img_data_path)
        if not img_data_path:
            raise ValueError(
                f"IMG_DATA directory not found in the provided path: {directory_path}"
            )

            # List all .jp2 files in the found IMG_DATA directory and its subdirectories
        jp2_files = []
        for subdir, _, files in os.walk(img_data_path):
            for file in files:
                if file.endswith(".jp2"):
                    jp2_files.append(os.path.join(subdir, file))

        print(jp2_files)
        # Convert each .jp2 file to numpy array and store in a list
        channel_dict = {}
        channel_names = []
        for jp2_file in jp2_files:
            with rasterio.open(jp2_file) as src:
                band_name = (
                    os.path.basename(jp2_file).split("_")[2].split(".")[0]
                )  # Assuming the band name is in the filename like "Txxxxx_B0x.jp2"
                channel_names.append(band_name)
                if band_name.startswith("TCI"):
                    tci_dims = src.read(1).shape
                channel_dict[band_name] = src.read(1)
        resized_arrays = []
        for band in [
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B10",
            "B11",
            "B12",
        ]:
            resized_arrays.append(self.resize_image(channel_dict[band], tci_dims))
        del channel_dict
        # Stack the arrays along the third dimension (channels)
        stacked_array = np.stack(resized_arrays, axis=-1)
        # img is (10980, 10980, 13)
        return stacked_array

    def preprocess_data(self, raw_data):
        """_summary_
        Parameters
        ----------
        raw_data : _type_
            _description_
        Returns
        -------
        _type_
        """
        normalized_data = (raw_data - 1000) / 10000
        normalized_data = normalized_data.astype("float32")
        return normalized_data

    def get_cloud_mask(
        self,
        x_coord: int,
        y_coord: int,
        threshold: float = 0.4,
        average_over_pix: int = 22,
        dilation_size: int = 11,
        all_bands: bool = True,
    ):
        """gets cloud mask from sentinel-2 scene
        Parameters
        ----------
        scene_dir_path : str
        threshold : float, optional
        average_over_pix : int, optional
        dilation_size : int, optional
        all_bands : bool, optional
        Returns
        -------
        _type_
        """

        data = self.img_array[x_coord - 1 : x_coord + 1, y_coord - 1 : y_coord + 1, :]
        normalized_data = self.preprocess_data(data)

        cloud_detector = S2PixelCloudDetector(
            threshold=threshold,
            average_over=average_over_pix,
            dilation_size=dilation_size,
            all_bands=True,
        )
        cloud_prob = cloud_detector.get_cloud_probability_maps(
            normalized_data[np.newaxis, ...]
        )
        return cloud_prob
