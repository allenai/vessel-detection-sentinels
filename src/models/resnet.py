import torch
import torchvision


class Model(torch.nn.Module):
    def __init__(self, info):
        super(Model, self).__init__()

        self.task = info["Data"]["task"]
        num_channels = info["Channels"].count()

        options = info["Options"]
        resnet_mode = options.get("Mode", "resnet50")

        if self.task == "regression":
            self.num_classes = 1
        else:
            self.num_classes = len(info["Data"]["categories"])

        resnet_fn = None
        if resnet_mode == "resnet18":
            resnet_fn = torchvision.models.resnet.resnet18
        elif resnet_mode == "resnet34":
            resnet_fn = torchvision.models.resnet.resnet34
        elif resnet_mode == "resnet50":
            resnet_fn = torchvision.models.resnet.resnet50
        elif resnet_mode == "resnet101":
            resnet_fn = torchvision.models.resnet.resnet101
        elif resnet_mode == "resnet152":
            resnet_fn = torchvision.models.resnet.resnet152

        self.resnet = resnet_fn(
            pretrained=True,
        )
        self.resnet.conv1 = torch.nn.Conv2d(
            num_channels,
            self.resnet.conv1.out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # We could pass num_classes to resnet_fn, but then it doesn't work with loading pre-trained model.
        # So instead we override the fully-connected layer here.
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, self.num_classes)

    def forward(self, images, targets=None):
        images = torch.stack(images, dim=0)
        output = self.resnet(images)  # batch x self.num_classes

        if self.task == "regression":
            loss = None
            if targets:
                targets = torch.stack([target["target"] for target in targets], dim=0)
                loss = torch.mean(torch.square(output[:, 0] - targets))

            return output[:, 0], loss
        elif self.task == "classification":
            loss = None
            if targets:
                targets = torch.stack([target["target"] for target in targets], dim=0)
                loss = torch.nn.functional.cross_entropy(input=output, target=targets)

            return output, loss
