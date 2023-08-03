import math
import random

import torch
import torchvision


class CropFlip(object):
    def __init__(self, model_cfg, options, my_cfg):
        self.my_cfg = my_cfg
        self.task = model_cfg["Data"]["task"]

    def __call__(self, data, targets):
        flip_horizontal = (
            self.my_cfg.get("HorizontalFlip", False) and random.random() < 0.5
        )
        flip_vertical = self.my_cfg.get("VerticalFlip", False) and random.random() < 0.5

        if "CropTo" in self.my_cfg:
            crop_width = self.my_cfg["CropTo"]
            crop_height = self.my_cfg["CropTo"]
        else:
            crop_width = data.shape[2]
            crop_height = data.shape[1]

        crop_left = random.randint(0, data.shape[2] - crop_width)
        crop_top = random.randint(0, data.shape[1] - crop_height)

        def crop_and_flip(im):
            if len(im.shape) == 3:
                im = im[
                    :,
                    crop_top: crop_top + crop_height,
                    crop_left: crop_left + crop_width,
                ]
                if flip_horizontal:
                    im = torch.flip(im, dims=[2])
                if flip_vertical:
                    im = torch.flip(im, dims=[1])
                return im
            elif len(im.shape) == 2:
                im = im[crop_top: crop_top + crop_height, crop_left: crop_left + crop_width]
                if flip_horizontal:
                    im = torch.flip(im, dims=[1])
                if flip_vertical:
                    im = torch.flip(im, dims=[0])
                return im

        data = crop_and_flip(data)

        if self.task in ["segmentation", "pixel_regression"]:
            targets["target"] = crop_and_flip(targets["target"])

        elif self.task in ["point", "box"]:
            valid_indices = (
                (targets["centers"][:, 0] > crop_left)
                & (targets["centers"][:, 0] < crop_left + crop_width)
                & (targets["centers"][:, 1] > crop_top)
                & (targets["centers"][:, 1] < crop_top + crop_height)
            )

            targets["centers"] = targets["centers"][valid_indices, :].contiguous()
            targets["boxes"] = targets["boxes"][valid_indices, :].contiguous()
            targets["labels"] = targets["labels"][valid_indices].contiguous()

            targets["centers"][:, 0] -= crop_left
            targets["centers"][:, 1] -= crop_top
            targets["boxes"][:, 0] -= crop_left
            targets["boxes"][:, 1] -= crop_top
            targets["boxes"][:, 2] -= crop_left
            targets["boxes"][:, 3] -= crop_top

            if flip_horizontal:
                targets["centers"][:, 0] = crop_width - targets["centers"][:, 0]
                targets["boxes"] = torch.stack(
                    [
                        crop_width - targets["boxes"][:, 2],
                        targets["boxes"][:, 1],
                        crop_width - targets["boxes"][:, 0],
                        targets["boxes"][:, 3],
                    ],
                    dim=1,
                )
            if flip_vertical:
                targets["centers"][:, 1] = data.shape[1] - targets["centers"][:, 1]
                targets["boxes"] = torch.stack(
                    [
                        targets["boxes"][:, 0],
                        crop_height - targets["boxes"][:, 3],
                        targets["boxes"][:, 2],
                        crop_height - targets["boxes"][:, 1],
                    ],
                    dim=1,
                )

        return data, targets


class Rotate(object):
    def __init__(self, info):
        pass

    def __call__(self, image, targets):
        angle_deg = random.randint(0, 359)
        angle_rad = angle_deg * math.pi / 180
        image = torchvision.transforms.functional.rotate(image, angle_deg)

        if len(targets["boxes"]) == 0:
            return image, targets

        im_center = (image.shape[2] // 2, image.shape[1] // 2)
        # Subtract center.
        centers = torch.stack(
            [
                targets["centers"][:, 0] - im_center[0],
                targets["centers"][:, 1] - im_center[1],
            ],
            dim=1,
        )
        # Rotate around origin.
        centers = torch.stack(
            [
                math.sin(angle_rad) * centers[:, 1]
                + math.cos(angle_rad) * centers[:, 0],
                math.cos(angle_rad) * centers[:, 1]
                - math.sin(angle_rad) * centers[:, 0],
            ],
            dim=1,
        )
        # Add back the center.
        centers = torch.stack(
            [
                centers[:, 0] + im_center[0],
                centers[:, 1] + im_center[1],
            ],
            dim=1,
        )
        # Prune ones outside image window.
        valid_indices = (
            (centers[:, 0] > 0)
            & (centers[:, 0] < image.shape[2])
            & (centers[:, 1] > 0)
            & (centers[:, 1] < image.shape[1])
        )
        centers = centers[valid_indices, :].contiguous()
        targets["centers"] = centers

        # Update boxes too, by keeping dimensions the same.
        boxes = targets["boxes"][valid_indices, :]
        x_sides = (boxes[:, 2] - boxes[:, 0]) / 2
        y_sides = (boxes[:, 3] - boxes[:, 1]) / 2
        targets["boxes"] = torch.stack(
            [
                centers[:, 0] - x_sides,
                centers[:, 1] - y_sides,
                centers[:, 0] + x_sides,
                centers[:, 1] + y_sides,
            ],
            dim=1,
        )

        targets["labels"] = targets["labels"][valid_indices].contiguous()

        return image, targets


class Noise(object):
    def __init__(self, info):
        pass

    def __call__(self, image, targets):
        image = image + 0.1 * torch.randn(image.shape)
        image = torch.clip(image, min=0, max=1)
        return image, targets


class Jitter(object):
    def __init__(self, info):
        pass

    def __call__(self, image, targets):
        jitter = 0.4 * (torch.rand(image.shape[0]) - 0.5)
        image = image + jitter[:, None, None]
        image = torch.clip(image, min=0, max=1)
        return image, targets


class Jitter2(object):
    def __init__(self, info):
        pass

    def __call__(self, image, targets):
        jitter = 0.1 * (torch.rand(image.shape[0]) - 0.5)
        image = image + jitter[:, None, None]
        image = torch.clip(image, min=0, max=1)
        return image, targets


class Bucket(object):
    def __init__(self, info):
        pass

    def __call__(self, image, targets):
        for channel_idx in [0, 1]:
            buckets = torch.tensor(
                [(i + 1) / 10 + (random.random() - 0.5) / 10 for i in range(9)],
                device=image.device,
            )
            image[channel_idx, :, :] = (
                torch.bucketize(image[channel_idx, :, :], buckets).float() / 10
            )
        return image, targets


class HideOverlapping(object):
    def __init__(self, info):
        self.overlap_indexes = []
        for i, channel in enumerate(info["Channels"].flatten()):
            if "overlap" in channel:
                self.overlap_indexes.append(i)

    def __call__(self, image, targets):
        hide = random.random() < 0.2
        if not hide:
            return image, targets

        if random.random() < 0.5:
            color = 0
        else:
            color = 255

        for i in self.overlap_indexes:
            image[i, :, :] = color

        return image, targets
