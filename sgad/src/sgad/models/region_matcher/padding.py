import torch


def pad_overlap_matrix(data):
    """Pad per-sample overlap matrices to the matcher confidence shape."""
    batch_size, max_rows, max_cols = data["conf_matrix_bbox"].shape
    padded = torch.zeros(
        (batch_size, max_rows, max_cols),
        device=data["conf_matrix_bbox"].device,
        dtype=torch.float32,
    )

    for batch_idx, overlap in enumerate(data["overlap_matrix"]):
        rows, cols = overlap.shape
        padded[batch_idx, :rows, :cols] = overlap

    data["overlap_matrix"] = padded


def pad_aligned_features(*feature_lists):
    """Pad aligned variable-length feature lists and return a shared mask."""
    if not feature_lists:
        raise ValueError("At least one feature list is required")

    batch_size = len(feature_lists[0])
    if any(len(features) != batch_size for features in feature_lists[1:]):
        raise ValueError("Aligned feature lists must have the same batch size")

    lengths = [features.shape[0] for features in feature_lists[0]]
    for features in feature_lists[1:]:
        if [item.shape[0] for item in features] != lengths:
            raise ValueError("Aligned feature lists must have matching lengths")

    max_length = max(lengths)
    reference = feature_lists[0][0]
    mask = torch.zeros(
        batch_size,
        max_length,
        dtype=torch.bool,
        device=reference.device,
    )
    padded_lists = []

    for features in feature_lists:
        sample = features[0]
        padded = torch.zeros(
            batch_size,
            max_length,
            *sample.shape[1:],
            dtype=sample.dtype,
            device=sample.device,
        )
        for batch_idx, (item, length) in enumerate(zip(features, lengths)):
            padded[batch_idx, :length] = item
        padded_lists.append(padded)

    for batch_idx, length in enumerate(lengths):
        mask[batch_idx, :length] = True

    return (*padded_lists, mask)
