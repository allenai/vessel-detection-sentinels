from collections import OrderedDict

import torch
import torchvision
from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.ops.feature_pyramid_network import (
    FeaturePyramidNetwork,
    LastLevelMaxPool,
)

from .frcnn import NoopTransform

PRETRAINED_MODELS = {
    "tiny": {
        "constructor": torchvision.models.swin_v2_t,
        "weights": torchvision.models.Swin_V2_T_Weights.IMAGENET1K_V1,
    },
    "small": {
        "constructor": torchvision.models.swin_v2_s,
        "weights": torchvision.models.Swin_V2_S_Weights.IMAGENET1K_V1,
    },
    "base": {
        "constructor": torchvision.models.swin_v2_b,
        "weights": torchvision.models.Swin_V2_B_Weights.IMAGENET1K_V1,
    },
}


INTERMEDIATE_SPECIFICATIONS = {
    "tiny": {
        "four_features": {
            "feature_slices": [slice(0, 2), slice(2, 4), slice(4, 6), slice(6, 8)],
            "feature_dims": [96 * 2, 192 * 2, 384 * 2, 768 * 2],
        },
        "single_feature": {"feature_slices": [slice(0, 6)], "feature_dims": [384 * 2]},
    },
    "small": {
        "four_features": {
            "feature_slices": [slice(0, 2), slice(2, 4), slice(4, 6), slice(6, 8)],
            "feature_dims": [96 * 2, 192 * 2, 384 * 2, 768 * 2],
        },
        "single_feature": {"feature_slices": [slice(0, 6)], "feature_dims": [384 * 2]},
    },
    "base": {
        "four_features": {
            "feature_slices": [slice(0, 2), slice(2, 4), slice(4, 6), slice(6, 8)],
            "feature_dims": [128 * 2, 256 * 2, 512 * 2, 1024 * 2],
        },
        "single_feature": {"feature_slices": [slice(0, 6)], "feature_dims": [384 * 2]},
    },
}


class SimpleBackbone(torch.nn.Module):
    def __init__(self, num_channels):
        super(SimpleBackbone, self).__init__()

        def down_layer(in_channels, out_channels):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
                torch.nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03),
                torch.nn.ReLU(inplace=True),
            )

        self.down1 = down_layer(num_channels, 32)  # -> 400x400
        self.down2 = down_layer(32, 64)  # -> 200x200
        self.down3 = down_layer(64, 128)  # -> 100x100
        self.down4 = down_layer(128, 256)  # -> 50x50
        self.down5 = down_layer(256, 512)  # -> 25x25
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, 3, padding=1),
        )

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        features = self.features(down5)
        return {
            "0": down2,
            "1": down3,
            "2": down4,
            "3": features,
        }


class MyBackbone(torch.nn.Module):
    def __init__(self, aggregate_op="sum", encoder_backbone="simple", group_channels=3):
        super(MyBackbone, self).__init__()
        self.aggregate_op = aggregate_op
        self.group_channels = group_channels

        if encoder_backbone == "simple":
            self.backbone = SimpleBackbone(self.group_channels)
            encoder_channels = [128, 256, 512, 1024]
        elif encoder_backbone == "resnet50":
            returned_layers = [1, 2, 3, 4]
            return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}
            self.backbone = IntermediateLayerGetter(
                resnet.resnet50(
                    weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
                ),
                return_layers=return_layers,
            )
            if self.group_channels != 3:
                self.backbone.conv1 = torch.nn.Conv2d(
                    self.group_channels,
                    self.backbone.conv1.out_channels,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                )
            encoder_channels = [512, 1024, 2048, 4096]
        elif encoder_backbone == "resnet101":
            returned_layers = [1, 2, 3, 4]
            return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}
            self.backbone = IntermediateLayerGetter(
                resnet.resnet101(
                    weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1
                ),
                return_layers=return_layers,
            )
            if self.group_channels != 3:
                self.backbone.conv1 = torch.nn.Conv2d(
                    self.group_channels,
                    self.backbone.conv1.out_channels,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                )
            encoder_channels = [512, 1024, 2048, 4096]
        else:
            raise Exception("bad encoder backbone {}".format(encoder_backbone))

        self.out_channels = 256
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=encoder_channels,
            out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool(),
        )

    def forward(self, x):
        base_im_features = self.backbone(x[:, 0: self.group_channels, :, :])

        # Geat features for overlaps
        overlap_dict = {}
        for i in range(self.group_channels, x.shape[1], self.group_channels):
            overlap_features = self.backbone(x[:, i: i + self.group_channels, :, :])
            # Add feature contribution for current overlap
            for k, v in overlap_features.items():
                if k not in overlap_dict:
                    overlap_dict[k] = v
                else:
                    if self.aggregate_op == "sum":
                        overlap_dict[k] = overlap_dict[k] + v
                    elif self.aggregate_op == "max":
                        overlap_dict[k] = torch.maximum(overlap_dict[k], v)
                    else:
                        raise Exception(
                            "Unknown aggregate op: {}".format(self.aggregate_op)
                        )
        out_dict = {}
        if len(overlap_dict.keys()) == len(base_im_features.keys()):
            for k, v in base_im_features.items():
                out_dict[k] = torch.cat([base_im_features[k], overlap_dict[k]], dim=1)
        else:
            for k, v in base_im_features.items():
                blank_history = torch.zeros(
                    size=base_im_features[k].shape,
                    dtype=base_im_features[k].dtype,
                    device=base_im_features[k].device,
                )
                out_dict[k] = torch.cat([base_im_features[k], blank_history], dim=1)

        return self.fpn(out_dict)


class SwinTransformerIntermediateLayerModel(torch.nn.Module):
    def __init__(self, variant="tiny", feature_cfg="four_features", in_channels=3):
        super().__init__()

        if variant not in ["tiny", "small", "base"]:
            raise ValueError(
                "This class only supports 'tiny', 'small' and 'base' swin transformer v2 variants."
            )

        # Isolate models to produce feature maps
        self.pretrained_model_cfg = PRETRAINED_MODELS[variant]
        self.pretrained_model = self.pretrained_model_cfg["constructor"](
            weights=self.pretrained_model_cfg["weights"]
        )

        # Default torch impl. has 3-channel input conv
        if in_channels != 3:
            conv0 = self.pretrained_model.features[0][0]
            self.pretrained_model.features[0][0] = torch.nn.Conv2d(
                in_channels,
                conv0.out_channels,
                kernel_size=conv0.kernel_size,
                stride=conv0.stride,
                padding=conv0.padding
            )
        self.intermediate_feature_cfg = INTERMEDIATE_SPECIFICATIONS[variant][
            feature_cfg
        ]
        pretrained_model_slices = self.intermediate_feature_cfg["feature_slices"]
        self.intermediate_models = [
            self.pretrained_model.features[s] for s in pretrained_model_slices
        ]
        self.feature_dims = self.intermediate_feature_cfg["feature_dims"]

    def forward(self, x):
        """Return OrderedDict of intermediate features from specified Swin Transformer variant.

        Features returned with shape [n_samples, n_channels, height, width]
        """
        out_dict = OrderedDict()
        for idx, model in enumerate(self.intermediate_models):
            out_dict[f"{idx}"] = x = model(x)

        for k, v in out_dict.items():
            out_dict[k] = v.permute(0, 3, 1, 2)

        return out_dict


class SwinTransformerRCNNBackbone(torch.nn.Module):
    def __init__(
        self, aggregate_op="sum", use_fpn=False, variant="small", group_channels=3
    ):
        super().__init__()
        self.aggregate_op = aggregate_op
        self.group_channels = group_channels
        self.use_fpn = use_fpn

        if variant not in ["tiny", "small", "base"]:
            raise ValueError(
                "This class only supports 'tiny', 'small' and 'base' swin transformer v2 variants."
            )

        self.out_channels = 256

        # Prepare dict of intermediate layer outputs for consumption by FPN.
        if self.use_fpn:
            self.backbone = SwinTransformerIntermediateLayerModel(
                variant=variant, feature_cfg="four_features", in_channels=group_channels
            )
            feature_dims = self.backbone.feature_dims

            self.fpn = FeaturePyramidNetwork(
                in_channels_list=feature_dims,
                out_channels=self.out_channels,
                extra_blocks=LastLevelMaxPool(),
            )

        # Output a single feature vector for consumption downstream
        else:
            self.backbone = SwinTransformerIntermediateLayerModel(
                variant=variant, feature_cfg="single_feature", in_channels=group_channels
            )

    def forward(self, x):

        base_im_features = self.backbone(x[:, 0: self.group_channels, :, :])
        # Get features for overlaps
        overlap_dict = {}
        for i in range(self.group_channels, x.shape[1], self.group_channels):
            overlap_features = self.backbone(x[:, i: i + self.group_channels, :, :])
            # Add feature contribution for current overlap
            for k, v in overlap_features.items():
                if k not in overlap_dict:
                    overlap_dict[k] = v
                else:
                    if self.aggregate_op == "sum":
                        overlap_dict[k] = overlap_dict[k] + v
                    elif self.aggregate_op == "max":
                        overlap_dict[k] = torch.maximum(overlap_dict[k], v)
                    else:
                        raise Exception(
                            "Unknown aggregate op: {}".format(self.aggregate_op)
                        )
        out_dict = {}
        if len(overlap_dict.keys()) == len(base_im_features.keys()):
            for k, v in base_im_features.items():
                out_dict[k] = torch.cat([base_im_features[k], overlap_dict[k]], dim=1)
        else:
            for k, v in base_im_features.items():
                blank_history = torch.zeros(
                    size=base_im_features[k].shape,
                    dtype=base_im_features[k].dtype,
                    device=base_im_features[k].device,
                )
                out_dict[k] = torch.cat([base_im_features[k], blank_history], dim=1)
        if self.use_fpn:
            return self.fpn(out_dict)

        else:
            return out_dict


class FasterRCNNModel(torch.nn.Module):
    def __init__(self, info):
        super(FasterRCNNModel, self).__init__()

        options = info["Options"]
        image_size = info["Example"][0].shape[1]
        num_classes = len(info["Data"]["categories"])

        use_noop_transform = options.get("NoopTransform", True)
        aggregate_op = options.get("AggregateOp", "sum")
        encoder_backbone = options.get("EncoderBackbone", "simple")
        encoder_backbone_variant = options.get("EncoderBackboneVariant", "tiny")
        encoder_backbone_use_fpn = options.get("EncoderBackboneUseFPN", "true")
        # Number of channels per layer.
        # This is used to distinguish channels of current image from those of overlapping images.
        group_channels = options.get("GroupChannels", 2)

        # We have max 86 points per 800x800 chip.
        # So here, in case we're using larger image sizes, determine if we need to increase some parameters.
        box_detections_per_img = max(100, 100 * image_size * image_size // 800 // 800)
        rpn_pre_nms_top_n_train = max(
            2000, 2000 * image_size * image_size // 800 // 800
        )
        rpn_post_nms_top_n_train = max(
            2000, 2000 * image_size * image_size // 800 // 800
        )
        rpn_pre_nms_top_n_test = max(1000, 1000 * image_size * image_size // 800 // 800)
        rpn_post_nms_top_n_test = max(
            1000, 1000 * image_size * image_size // 800 // 800
        )

        if encoder_backbone == "swintransformer":
            use_fpn = encoder_backbone_use_fpn == "true"
            self.backbone = SwinTransformerRCNNBackbone(
                use_fpn=use_fpn,
                group_channels=group_channels,
                aggregate_op=aggregate_op,
                variant=encoder_backbone_variant,
            )

        else:
            self.backbone = MyBackbone(
                aggregate_op=aggregate_op,
                encoder_backbone=encoder_backbone,
                group_channels=group_channels,
            )
        self.faster_rcnn = FasterRCNN(
            self.backbone,
            num_classes + 1,
            min_size=image_size,
            max_size=image_size,
            box_detections_per_img=box_detections_per_img,
            rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
            rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
            rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
        )

        # replace the classifier with a new one for user-defined num_classes
        in_features = self.faster_rcnn.roi_heads.box_predictor.cls_score.in_features
        self.faster_rcnn.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes + 1
        )

        if use_noop_transform:
            self.faster_rcnn.transform = NoopTransform()

    def forward(self, images, raw_targets=None):
        device = images[0].device

        # Fix targets: if there are no labels, then for some reason we need to set one label.
        # If there are labels, we need to increment it by one, since 0 is reserved for background.
        targets = None
        if raw_targets:
            targets = []
            for target in raw_targets:
                target = dict(target)
                if len(target["boxes"]) == 0:
                    target["labels"] = torch.zeros(
                        (1,), device=device, dtype=torch.int64
                    )
                else:
                    target["labels"] = target["labels"] + 1
                targets.append(target)

        images, targets = self.faster_rcnn.transform(images, targets)
        features = self.faster_rcnn.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.faster_rcnn.rpn(images, features, targets)
        detections, detector_losses = self.faster_rcnn.roi_heads(
            features, proposals, images.image_sizes, targets
        )

        losses = {"base": torch.tensor(0, device=device, dtype=torch.float32)}
        losses.update(proposal_losses)
        losses.update(detector_losses)

        # Fix detections: need to decrement class label.
        for output in detections:
            output["labels"] = output["labels"] - 1

        loss = sum(x for x in losses.values())
        return detections, loss
