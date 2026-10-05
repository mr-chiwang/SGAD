"""Hierarchical containment filtering of matched bbox pairs.

All boxes use [x_min, x_max, y_min, y_max]. Filtering runs on CPU and
returns shared indices; it never modifies coordinates or reassigns matches.
"""

from functools import lru_cache

import numpy as np


def rectangle_union_area(boxes):
    """Return exact continuous rectangle-union area without image-sized masks."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    boxes = boxes[(boxes[:, 1] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 2])]
    if not len(boxes):
        return 0.0
    xs = np.unique(boxes[:, :2])
    total = 0.0
    for left, right in zip(xs[:-1], xs[1:]):
        intervals = boxes[(boxes[:, 0] < right) & (boxes[:, 1] > left), 2:]
        if not len(intervals):
            continue
        intervals = intervals[np.argsort(intervals[:, 0], kind="stable")]
        bottom, top = intervals[0]
        height = 0.0
        for lo, hi in intervals[1:]:
            if lo > top:
                height += top - bottom
                bottom, top = lo, hi
            else:
                top = max(top, hi)
        total += (right - left) * (height + top - bottom)
    return float(total)


def _boxes_array(boxes):
    array = np.asarray(boxes, dtype=np.float64)
    if array.size == 0 and array.shape in {(0,), (0, 4)}:
        return array.reshape(0, 4)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError("Matched boxes must have shape [N, 4].")
    if not np.isfinite(array).all():
        raise ValueError("Matched boxes must have finite coordinates.")
    if np.any(array[:, 1] <= array[:, 0]) or np.any(array[:, 3] <= array[:, 2]):
        raise ValueError("Matched boxes must have positive width and height.")
    return array


def _indices(mask):
    """Iterate set bits without limiting region counts to a machine word."""
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def _minimum_cover(near, finer, covered, preference):
    """Solve the monotone final-set constraints by reduced subset-state DP."""
    n = len(near)
    full = (1 << n) - 1
    near_masks = [sum(1 << int(j) for j in np.flatnonzero(row)) for row in near]
    fine_masks = [sum(1 << int(j) for j in np.flatnonzero(row)) for row in finer]

    @lru_cache(None)
    def fine_covered(i, mask):
        return covered(i, list(_indices(mask)))

    def satisfied(i, selected):
        return bool(selected & ((1 << i) | near_masks[i])) or fine_covered(i, selected & fine_masks[i])

    mandatory = sum(1 << i for i in range(n) if not satisfied(i, full ^ (1 << i)))
    optional = full ^ mandatory
    constraints = {i: ((1 << i) | near_masks[i] | fine_masks[i]) & optional
                   for i in _indices(optional) if not satisfied(i, mandatory)}

    # Factor components share optional variables, not merely geometric edges.
    # Constraints already satisfied by mandatory boxes cannot couple choices.
    components = []
    for support in constraints.values():
        merged, rest = support, []
        for component in components:
            if component & support:
                merged |= component
            else:
                rest.append(component)
        components = rest + [merged]

    tie_bits = {i: 1 << (n - position - 1) for position, i in enumerate(preference)}

    def objective(mask):
        return mask.bit_count(), -sum(tie_bits[i] for i in _indices(mask))

    selected = mandatory
    for component in components:
        factors = [i for i, support in constraints.items() if support & component]

        @lru_cache(None)
        def remaining_choices(mask):
            # Branch on one unsatisfied factor. Every feasible extension must
            # add at least one of its remaining variables; no solution is lost.
            choices = None
            for i in factors:
                if not satisfied(i, mandatory | mask):
                    support = constraints[i] & ~mask
                    if choices is None or support.bit_count() < choices.bit_count():
                        choices = support
            return choices

        @lru_cache(None)
        def best(mask):
            choices = remaining_choices(mask)
            if choices is None:
                return mask
            children = [mask | (1 << j) for j in preference if choices & (1 << j)]
            # An unsatisfied state needs at least one addition. A feasible
            # one-step child attains that bound; the preference-ordered first
            # such child also wins the tie. No deeper state can improve it.
            for child in children:
                if remaining_choices(child) is None:
                    return child
            return min((best(child) for child in children), key=objective)

        selected |= best(0)
        best.cache_clear()
        remaining_choices.cache_clear()
    return np.asarray(list(_indices(selected)), dtype=np.int64)


def filter_matched_regions(
    boxes0,
    boxes1,
    *,
    overlap_threshold=0.9,
    coverage_threshold=0.4,
    image_shape0=None,
    image_shape1=None,
):
    """Return a minimum-cardinality feasible set of matched region pairs.

    A parent suppresses a child only when they mutually cover at least
    overlap_threshold of each other in BOTH images (near duplicates).
    Children can replace a parent only when their union covers
    coverage_threshold of the parent in each image, regardless of its size.
    Image shapes are (height, width[, channels]) and only normalize the
    coarse-to-fine ordering across views.

    Every omitted ORIGINAL pair needs a retained near duplicate or sufficient
    union coverage by eligible finer pairs in the FINAL set. Ancestors cannot
    substitute for finer pairs. Mandatory pairs are fixed first; remaining
    independent constraint components use memoized subset-state dynamic
    programming, not greedy deletion or a tree approximation.

    Among equally small feasible sets, prefer finer pairs in ascending summed
    normalized area, then symmetric coordinate order. No confidence or extra
    coefficient is used. Exact duplicate input indices are interchangeable.

    Fine-replacement eligibility is ordered; near replacement is symmetric.
    The subset-state DP handles these shared and cyclic dependencies directly.
    Opposite containment directions between images are not a hierarchy,
    unless the pairs are mutually near-contained in both images.

    Returns:
        A 1D integer NumPy array of retained indices, in input order.
        Coverage refers to bbox area, not masks or point-matching quality.
    """
    boxes = (_boxes_array(boxes0), _boxes_array(boxes1))
    n = len(boxes[0])
    if len(boxes[1]) != n:
        raise ValueError("Both images must have the same number of matched boxes.")
    for name, value in (("overlap_threshold", overlap_threshold),
                        ("coverage_threshold", coverage_threshold)):
        if not np.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} must be in (0, 1].")

    areas = np.array([(b[:, 1] - b[:, 0]) * (b[:, 3] - b[:, 2]) for b in boxes])
    scales = []
    for view, shape in enumerate((image_shape0, image_shape1)):
        if shape is None:
            scales.append(1.0)
            continue
        if len(shape) not in (2, 3) or not np.isfinite(shape[:2]).all() or min(shape[:2]) <= 0:
            raise ValueError("Image shapes must contain positive height and width.")
        image_area = float(shape[0]) * float(shape[1])
        scales.append(image_area)
    if not n:
        return np.empty(0, dtype=np.int64)

    # overlaps[v, i, j] is the fraction of box j covered by box i.
    intersections = []
    for b in boxes:
        clipped = np.stack((
            np.maximum(b[:, None, 0], b[None, :, 0]),
            np.minimum(b[:, None, 1], b[None, :, 1]),
            np.maximum(b[:, None, 2], b[None, :, 2]),
            np.minimum(b[:, None, 3], b[None, :, 3]),
        ), axis=-1)
        intersections.append(clipped)
    intersections = np.asarray(intersections)
    intersection_areas = (
        np.maximum(0, intersections[..., 1] - intersections[..., 0])
        * np.maximum(0, intersections[..., 3] - intersections[..., 2])
    )
    overlaps = intersection_areas / areas[:, None, :]
    contains = np.all(overlaps >= overlap_threshold - 1e-12, axis=0)
    dominates = np.all(areas[:, :, None] >= areas[:, None, :], axis=0)
    related = contains & (dominates | contains.T)
    np.fill_diagonal(related, False)

    def priority(i):
        # Symmetric under swapping the images; independent of input order
        # except for indistinguishable duplicate pairs.
        size = areas[0, i] / scales[0] + areas[1, i] / scales[1]
        coordinates = tuple(sorted((tuple(boxes[0][i]), tuple(boxes[1][i]))))
        return (-size, coordinates, i)

    # Distinct pairs can be exact mirror images of one another across views.
    # Exclude direct replacement between distinct mirror-tied pairs. Other
    # eligible regions may still replace either pair in the final solution.
    keys = [priority(i) for i in range(n)]
    for i in range(n):
        for j in range(i):
            if keys[i][:-1] == keys[j][:-1] and not (
                np.array_equal(boxes[0][i], boxes[0][j])
                and np.array_equal(boxes[1][i], boxes[1][j])
            ):
                related[i, j] = related[j, i] = False
    order = sorted(range(n), key=lambda i: keys[i])
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)
    refines = related & (rank[None, :] > rank[:, None])

    def covered(view, index, retained, threshold):
        fractions = overlaps[view, retained, index]
        if len(fractions) and fractions.max() >= threshold - 1e-12:
            return True
        # The sum is only an upper bound, never a union estimate.
        if fractions.sum() < threshold - 1e-12:
            return False
        pieces = intersections[view, retained, index]
        return rectangle_union_area(pieces) / areas[view, index] >= threshold - 1e-12

    near = related & related.T
    preference = sorted(range(n), key=lambda i: (-keys[i][0], keys[i][1], i))
    return _minimum_cover(near, refines,
                          lambda i, retained: all(covered(v, i, retained, coverage_threshold) for v in range(2)),
                          preference)
