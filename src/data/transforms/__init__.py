class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, targets):
        for transform in self.transforms:
            image, targets = transform(image, targets)
        return image, targets


def get_transform(model_cfg, options, transforms_cfg):
    from . import augment as siv_transforms

    if len(transforms_cfg) == 0:
        return None

    transforms = []
    for transform_cfg in transforms_cfg:
        transform_cls = getattr(siv_transforms, transform_cfg["Name"])
        transform = transform_cls(model_cfg, options, transform_cfg)
        transforms.append(transform)
    return Compose(transforms)
