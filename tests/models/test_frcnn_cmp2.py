import pytest
import torch

from src.models.frcnn_cmp2 import INTERMEDIATE_SPECIFICATIONS, SwinTransformerIntermediateLayerModel, SwinTransformerRCNNBackbone


@pytest.mark.parametrize(
    "in_channels",
    [1, 3, 5, 8]
)
@pytest.mark.parametrize("feature_cfg,", ["single_feature", "four_features"])
def test_swin_transformer_intermediate_layer_model_forward(
        in_channels, feature_cfg, variant="small", batch_size=1, img_size=64):
    """Test expected output of the intermediate feature specifications."""
    test_img_batch = torch.rand(batch_size, in_channels, img_size, img_size)
    intermediate_layer_model = SwinTransformerIntermediateLayerModel(
        variant=variant, feature_cfg=feature_cfg, in_channels=in_channels
    )
    output = intermediate_layer_model(test_img_batch)

    feature_metadata = INTERMEDIATE_SPECIFICATIONS[variant][feature_cfg]
    num_features = len(feature_metadata["feature_dims"])

    # Expected number of features output
    assert len(output.keys()) == num_features

    # Expected feature shapes
    feature_dims = [x // 2 for x in feature_metadata["feature_dims"]]
    for idx, (k, v) in enumerate(output.items()):
        assert v.shape[0:2] == torch.Size([batch_size, feature_dims[idx]])


@pytest.mark.parametrize(
    "group_in_channels",
    [1, 3, 5, 8]
)
@pytest.mark.parametrize("use_fpn", [True, False])
@pytest.mark.parametrize(
    "num_groups",
    [1, 2, 3]
)
def test_swin_transformer_rcnn_backbone(
        group_in_channels, use_fpn, num_groups, variant="small", aggregate_op="sum", batch_size=1, img_size=64):
    """Test expected output of the swin transformer RCNN backbone."""
    test_img_batch = torch.rand(batch_size, group_in_channels*num_groups, img_size, img_size)

    swinbackbone = SwinTransformerRCNNBackbone(
        use_fpn=use_fpn,
        group_channels=group_in_channels,
        aggregate_op=aggregate_op,
        variant=variant,
    )

    features = swinbackbone(test_img_batch)

    # Expected output dict sizes
    if use_fpn:
        assert len(features.keys()) == 5
        output_feature_dim = swinbackbone.out_channels

    else:
        assert len(features.keys()) == 1
        output_feature_dim = swinbackbone.backbone.feature_dims[0]

    # Expected output dict item shapes
    for k, v in features.items():
        assert v.shape[0:2] == torch.Size([batch_size, output_feature_dim])
