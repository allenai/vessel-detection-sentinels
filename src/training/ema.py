from collections import OrderedDict
from copy import deepcopy
from sys import stderr

import torch

from torch import nn


class EMA(nn.Module):
    def __init__(self, model: nn.Module, decay: float):
        super().__init__()
        self.decay = decay

        self.model = model
        print("preparing EMA with decay={}".format(decay))
        self.shadow = deepcopy(self.model)

        for param in self.shadow.parameters():
            param.detach_()

    @torch.no_grad()
    def update(self, epoch):
        if not self.training:
            print(
                "EMA update should only be called during training",
                file=stderr,
                flush=True,
            )
            return

        model_params = OrderedDict(self.model.named_parameters())
        shadow_params = OrderedDict(self.shadow.named_parameters())

        # check if both model contains the same set of keys
        assert model_params.keys() == shadow_params.keys()

        if epoch <= 5:
            decay = 0.99
        else:
            decay = self.decay

        for name, param in model_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            shadow_params[name].sub_((1.0 - decay) * (shadow_params[name] - param))

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.shadow.named_buffers())

        # check if both model contains the same set of keys
        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            # buffers are copied
            shadow_buffers[name].copy_(buffer)

    def forward(self, *input, **kwargs):
        if self.training:
            return self.model(*input, **kwargs)
        else:
            return self.shadow(*input, **kwargs)
