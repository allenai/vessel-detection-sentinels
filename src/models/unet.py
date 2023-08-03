import torch


class Model(torch.nn.Module):
    def __init__(self, info):
        super(Model, self).__init__()

        num_channels = info["Channels"].count()

        def down_layer(in_channels, out_channels):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
                torch.nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03),
                torch.nn.ReLU(inplace=True),
            )

        def up_layer(in_channels, out_channels):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.BatchNorm2d(in_channels // 2, eps=1e-3, momentum=0.03),
                torch.nn.ConvTranspose2d(
                    in_channels // 2, out_channels, 4, stride=2, padding=1
                ),
                torch.nn.ReLU(inplace=True),
            )

        self.layers = torch.nn.Sequential(
            down_layer(num_channels, 32),  # 1/2
            down_layer(32, 64),  # 1/4
            down_layer(64, 128),  # 1/8
            down_layer(128, 256),  # 1/16
            up_layer(256, 128),  # 1/8
            up_layer(128, 64),  # 1/4
            up_layer(64, 32),  # 1/2
            up_layer(32, 32),  # 1,
            torch.nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, images, targets=None):
        images = torch.stack(images, dim=0)
        output = self.layers(images)[:, 0, :, :]

        loss = None
        if targets:
            targets = torch.stack([target["target"] for target in targets], dim=0)
            loss = torch.mean(torch.square(output - targets))

        return output, loss
