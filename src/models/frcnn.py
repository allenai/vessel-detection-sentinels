from collections import OrderedDict

import torch
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor


class FasterRCNNModel(torch.nn.Module):
    """
    Baseline model class for xView3 reference implementation.
    Wraps torchvision faster-rcnn, updates initial layer to handle
    man arbitrary number of input channels.
    """

    def __init__(self, info):
        super(FasterRCNNModel, self).__init__()

        options = info["Options"]
        image_size = info["Example"][0].shape[1]
        num_classes = len(info["Data"]["categories"])
        num_channels = info["Channels"].count()

        backbone = options.get("Backbone", "resnet50")
        pretrained = options.get("Pretrained", True)
        pretrained_backbone = options.get("Pretrained-Backbone", True)
        trainable_backbone_layers = options.get("Trainable-Backbone-Layers", 5)
        use_noop_transform = options.get("NoopTransform", True)

        # Load in a backbone, with capability to be pretrained on COCO
        if backbone == "resnet50":
            box_detections_per_img = max(
                100, 100 * image_size * image_size // 800 // 800
            )
            rpn_pre_nms_top_n_train = max(
                2000, 2000 * image_size * image_size // 800 // 800
            )
            rpn_post_nms_top_n_train = max(
                2000, 2000 * image_size * image_size // 800 // 800
            )
            rpn_pre_nms_top_n_test = max(
                1000, 1000 * image_size * image_size // 800 // 800
            )
            rpn_post_nms_top_n_test = max(
                1000, 1000 * image_size * image_size // 800 // 800
            )

            self.faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                pretrained=pretrained,
                min_size=image_size,
                max_size=image_size,
                pretrained_backbone=pretrained_backbone,
                trainable_backbone_layers=trainable_backbone_layers,
                box_detections_per_img=box_detections_per_img,
                rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
                rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
                rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
                rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
            )
        else:
            backbone = resnet_fpn_backbone(
                backbone_name=backbone,
                pretrained=pretrained_backbone,
                trainable_layers=trainable_backbone_layers,
            )
            model = FasterRCNN(
                backbone,
                num_classes + 1,
                min_size=image_size,
                max_size=image_size,
            )

        # replace the classifier with a new one for user-defined num_classes
        in_features = self.faster_rcnn.roi_heads.box_predictor.cls_score.in_features
        self.faster_rcnn.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes + 1
        )

        if use_noop_transform:
            self.faster_rcnn.transform = NoopTransform()

        print(f"Using {num_channels} channels for input layer...")
        self.num_channels = num_channels
        if num_channels != 3:
            # Adjusting initial layer to handle arbitrary number of inputchannels
            self.faster_rcnn.backbone.body.conv1 = torch.nn.Conv2d(
                num_channels,
                self.faster_rcnn.backbone.body.conv1.out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

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


class NoopTransform(torch.nn.Module):
    def __init__(self):
        super(NoopTransform, self).__init__()

        self.transform = (
            torchvision.models.detection.transform.GeneralizedRCNNTransform(
                min_size=800,
                max_size=800,
                image_mean=[],
                image_std=[],
            )
        )

    def forward(self, images, targets):
        images = self.transform.batch_images(images, size_divisible=32)
        image_sizes = [(image.shape[1], image.shape[2]) for image in images]
        image_list = torchvision.models.detection.image_list.ImageList(
            images, image_sizes
        )
        return image_list, targets

    def postprocess(self, detections, image_sizes, orig_sizes):
        return detections
