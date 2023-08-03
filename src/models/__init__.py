from . import custom as custom
from . import resnet as resnet
from . import unet as unet
from . import frcnn_cmp2 as frcnn_cmp2
from . import frcnn as frcnn
models = {}

models["frcnn"] = frcnn.FasterRCNNModel

models["frcnn_cmp2"] = frcnn_cmp2.FasterRCNNModel

models["unet"] = unet.Model

models["resnet"] = resnet.Model

models["custom"] = custom.Model
models["custom_separate_heads"] = custom.SeparateHeadAttrModel
