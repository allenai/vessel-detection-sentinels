import torch
import torchvision

from src.data.dataset import NEGATIVE_VESSEL_TYPE_FACTOR


class Model(torch.nn.Module):
    def __init__(self, info):
        super(Model, self).__init__()

        num_channels = info["Channels"].count()
        self.num_channels = num_channels
        self.resnet = torchvision.models.resnet.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        )
        self.resnet.conv1 = torch.nn.Conv2d(
            num_channels,
            self.resnet.conv1.out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.resnet.fc = torch.nn.Sequential(
            torch.nn.Linear(self.resnet.fc.in_features, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(
                256, 1 + 1 + 16 + 1 + 2
            ),  # loss, width, heading classes, speed, is_fishing_vessel classes
        )

    def forward(self, images, targets=None):
        images = torch.stack(images, dim=0)
        output = self.resnet(images)  # batch x self.num_classes

        loss = None
        if targets:
            targets = torch.stack([target["target"] for target in targets], dim=0)
            loss_length = torch.mean(torch.abs(output[:, 0] - targets[:, 0]))
            loss_width = torch.mean(torch.abs(output[:, 1] - targets[:, 1]))
            loss_heading = torch.nn.functional.cross_entropy(
                input=output[:, 2:18], target=targets[:, 2:18]
            )
            loss_speed = torch.mean(torch.abs(output[:, 18] - targets[:, 18]))
            loss_type = torch.nn.functional.cross_entropy(
                input=output[:, 19:21],
                target=targets[:, 19:21],
                weight=torch.tensor([1, NEGATIVE_VESSEL_TYPE_FACTOR]).to(
                    targets.device
                ),
            )
            loss = loss_length + loss_width + loss_speed + loss_type + loss_heading

        return output, loss


class SeparateHeadAttrModel(torch.nn.Module):
    def __init__(self, info):
        super().__init__()

        num_channels = info["Channels"].count()
        self.num_channels = num_channels

        def down_layer(in_channels, out_channels, stride=2):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 4, stride=stride, padding=1),
                torch.nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03),
                torch.nn.ReLU(inplace=True),
            )

        self.layer1 = down_layer(num_channels, 32, stride=1)
        self.layer2 = down_layer(32, 64)
        self.layer3 = down_layer(64, 128)
        self.layer4 = down_layer(128, 256)
        self.layer5 = down_layer(256, 512)
        self.layer6 = down_layer(512, 512)

        self.pred_length = torch.nn.Conv2d(512, 1, 4, stride=2, padding=1)
        self.pred_width = torch.nn.Conv2d(512, 1, 4, stride=2, padding=1)
        self.pred_heading = torch.nn.Conv2d(512, 16, 4, stride=2, padding=1)
        self.pred_speed = torch.nn.Conv2d(512, 1, 4, stride=2, padding=1)
        self.pred_activity_type = torch.nn.Conv2d(512, 2, 4, stride=2, padding=1)

    def forward(self, images, targets=None):
        images = torch.stack(images, dim=0)
        device = images.device
        layer1 = self.layer1(images)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)
        layer6 = self.layer6(layer5)

        lengths = self.pred_length(layer6)[:, :1, 0, 0]
        widths = self.pred_width(layer6)[:, :1, 0, 0]
        heading_bucket_preds = self.pred_heading(layer6)[:, :, 0, 0]
        speeds = self.pred_speed(layer6)[:, :1, 0, 0]
        activity_type = self.pred_activity_type(layer6)[:, :, 0, 0]

        output = torch.cat([lengths, widths, heading_bucket_preds, speeds, activity_type], dim=-1)

        loss = None
        if targets:
            targets = torch.stack([target["target"] for target in targets], dim=0)

            def get_normalized_pe(labels, preds):
                """Get a normalized percent error for labels which ought to be > 0.
                """
                valid_indices = labels >= 0
                valid_labels = labels[valid_indices]
                valid_preds = preds[valid_indices]
                if len(valid_labels) > 0:
                    loss = torch.div(
                        torch.abs(valid_labels - valid_preds),
                        valid_labels).mean()
                else:
                    loss = torch.zeros((1,), dtype=torch.float32, device=device)
                return loss

            # Length
            length_labels = targets[:, 0]
            loss_length = get_normalized_pe(length_labels, lengths)

            # Width
            width_labels = targets[:, 1]
            loss_width = get_normalized_pe(width_labels, widths)

            # Speed
            speed_labels = targets[:, 18]
            loss_speed = torch.mean(torch.abs(speeds - speed_labels))

            # Heading
            heading_labels = targets[:, 2:18]
            loss_heading = torch.nn.functional.cross_entropy(heading_bucket_preds, heading_labels)

            # Activity type
            activity_labels = targets[:, 19:21]
            loss_activity_type = torch.nn.functional.cross_entropy(activity_type, activity_labels, weight=torch.tensor([
                                                                   1, NEGATIVE_VESSEL_TYPE_FACTOR]).to(targets.device))

            loss = loss_length + loss_width + loss_speed + loss_heading + loss_activity_type

        return output, loss
