import torch.nn as nn

from sgad.config.model_variants import resolve_model_variant

from ..backbone import build_backbone
from .inputs import (
    pad_area_descriptor_batch,
    prepare_area_inputs,
)
from .features import (
    extract_bbox_features,
    extract_features_from_image,
)
from .matching import DualSoftmaxMatcher, match_area_descriptors_bidirectionally
from .padding import pad_overlap_matrix
from .transformers import apply_descriptor_transformer, build_descriptor_transformer

SUPPORTED_BACKBONE_TYPES = ("dinov2", "dinov3")


def resolve_backbone_input_dim(config) -> int:
    backbone_type = config["backbone_type"]
    if backbone_type not in SUPPORTED_BACKBONE_TYPES:
        supported = "', '".join(SUPPORTED_BACKBONE_TYPES)
        raise ValueError(
            f"Unknown backbone_type: {backbone_type}. "
            f"Supported types: '{supported}'"
        )
    return config[backbone_type]["initial_dim"]


def _write_outputs(
    data,
    *,
    bboxes_img_centers0,
    bboxes_img_centers1,
    mask_features0,
    mask_features1,
    conf_matrix_0to1,
    mask_0to1,
    conf_matrix_1to0,
    mask_1to0,
):
    data.update(
        {
            "bboxes_img_centers0": bboxes_img_centers0,
            "bboxes_img_centers1": bboxes_img_centers1,
            "mask_features0": mask_features0,
            "mask_features1": mask_features1,
        }
    )
    pad_overlap_matrix(data)
    data["conf_matrix_0to1"] = conf_matrix_0to1
    data["mask_0to1"] = mask_0to1
    data["conf_matrix_1to0"] = conf_matrix_1to0
    data["mask_1to0"] = mask_1to0


class AreaDescriptorMatcher(nn.Module):
    def __init__(self, config, img_resize=224):
        super().__init__()
        self.model_variant = resolve_model_variant(config)
        self.img_resize = img_resize

        self.backbone = build_backbone(config)
        input_dim = resolve_backbone_input_dim(config)

        transformer_modules = build_descriptor_transformer(config, input_dim)
        self.attention_type = transformer_modules.attention_type
        self.position_encoding = transformer_modules.position_encoding
        self.pos_scale = transformer_modules.pos_scale
        self.region_transformer = transformer_modules.region_transformer

        self.bbox_matching = DualSoftmaxMatcher()

        self.from_img = config['bbox']['from_img']

    def forward(self, data):
        bbox_inputs = prepare_area_inputs(
            data,
            from_img=self.from_img,
            img_resize=self.img_resize,
        )

        if self.from_img:
            bbox_features0 = extract_features_from_image(
                self.backbone,
                data['image0'],
                data['bboxes0'],
            )
            bbox_features1 = extract_features_from_image(
                self.backbone,
                data['image1'],
                data['bboxes1'],
            )
        else:
            bbox_features0 = extract_bbox_features(
                self.backbone,
                bbox_inputs.bbox_images0,
                batch_size=1500,
            )
            bbox_features1 = extract_bbox_features(
                self.backbone,
                bbox_inputs.bbox_images1,
                batch_size=1500,
            )

        bbox_batch = pad_area_descriptor_batch(
            bbox_features0,
            bbox_features1,
            bbox_inputs.centers0,
            bbox_inputs.centers1,
        )
        bbox_features0 = bbox_batch.features0
        bbox_features1 = bbox_batch.features1
        bboxes_img_centers0 = bbox_batch.centers0
        bboxes_img_centers1 = bbox_batch.centers1
        mask_features0 = bbox_batch.mask0
        mask_features1 = bbox_batch.mask1

        feat_c0, feat_c1 = apply_descriptor_transformer(
            attention_type=self.attention_type,
            position_encoding=self.position_encoding,
            pos_scale=self.pos_scale,
            region_transformer=self.region_transformer,
            bbox_features0=bbox_features0,
            bbox_features1=bbox_features1,
            bboxes_img_centers0=bboxes_img_centers0,
            bboxes_img_centers1=bboxes_img_centers1,
            mask_features0=mask_features0,
            mask_features1=mask_features1,
        )

        (
            conf_matrix_0to1,
            mask_0to1,
            conf_matrix_1to0,
            mask_1to0,
        ) = match_area_descriptors_bidirectionally(
            self.bbox_matching,
            feat_c0=feat_c0,
            feat_c1=feat_c1,
            data=data,
            mask_features0=mask_features0,
            mask_features1=mask_features1,
        )

        _write_outputs(
            data,
            bboxes_img_centers0=bboxes_img_centers0,
            bboxes_img_centers1=bboxes_img_centers1,
            mask_features0=mask_features0,
            mask_features1=mask_features1,
            conf_matrix_0to1=conf_matrix_0to1,
            mask_0to1=mask_0to1,
            conf_matrix_1to0=conf_matrix_1to0,
            mask_1to0=mask_1to0,
        )
