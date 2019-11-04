# -*- coding: utf-8 -*-
import time
from math import pi
from typing import List, Tuple

import cv2 as cv
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from .utils import rotate_image


def segment_weeds(
    image: np.ndarray,
    veg_mask: np.ndarray,
    crop_rows: np.ndarray,
    model: DecisionTreeClassifier = None
):
    """Segment weed pixels of a crop field image.

    Parameters
    ----------
    image : array-like of shape = [m, n, 3] and ``dtype=np.uint8``
        The crop field image to be segmented.
    veg_mask : array-like of shape = [m, n] and ``dtype=np.uint8``
        Vegetation mask image, where non zero and zero elements represents
        vegetation and background pixels, respectively.
    crop_rows : array-like of shape = [r, c, 2] and ``dtype=np.int32``
        Crop rows lines, where each element is crop row line represented by an
        array of `c` pairs of indexes in the form `(row, column)`. Each crop
        row line is a continuous polygonal line.
    model : sklearn.tree.DecisionTreeClassifier or None, optional (default=None)
        Decision tree-based model to classify vegetation pixels into "crop" and
        "weed" classes. Crop and weed pixels must be labeled as `0` and `1`,
        respectively.

    Returns
    -------
    weed_mask : array-like of shape = [m, n] and ``dtype=np.uint8``
        Weed mask image, where non zero and zero elements represents weed and
        background pixels, respectively.

    See also
    --------
    detect_rows, classification_model, segment_vegetation

    """
    veg_mask_bool = veg_mask > 0

    if model is None:
        dist_map = _row_distance_map(veg_mask, crop_rows)
        dist_thr = np.mean(dist_map[veg_mask_bool])
        pixels_crop = image[(dist_map < dist_thr) & (dist_map > 0)]
        pixels_weed = image[dist_map > 2 * dist_thr]
        model = classification_model(pixels_crop, pixels_weed)

    predictions = model.predict(image[veg_mask_bool])
    values = np.zeros((predictions.size, 1), np.uint8)
    values[predictions == 1] = 255
    weed_mask = np.zeros_like(veg_mask)
    weed_mask[veg_mask_bool] = values.reshape(-1)

    return weed_mask


def _row_distance_map(veg_mask: np.ndarray, rows: np.ndarray):
    """Compute distances between vegetation pixels and crop row lines.

    Parameters
    ----------
    veg_mask : array-like of shape = [m, n] and ``dtype=np.uint8``
        Vegetation mask image, where non zero and zero elements represents
        vegetation and background pixels, respectively.
    rows : array-like of shape = [r, c, 2] and ``dtype=np.int``
        Crop rows lines, where each element is crop row line represented by
        an array of `c` pairs of indexes in the form `(row, column)`. Each crop
        row line is a continuous polygonal line.

    Returns
    -------
    distance_map : array-like of shape = [m, n] and ``dtype=np.int32``
        An array of the same shape as the input `veg_mask`, where each
        element that correspond to a non zero element in `veg_mask` is non zero,
        represents the distance in pixels from that pixel to the closest crop
        row line.

    """
    distance_map = np.zeros(veg_mask.shape, np.int32)
    h, w = veg_mask.shape[:2]
    n_rows, n_cols = rows.shape[:2]

    for k in range(n_rows):
        for n in range(n_cols):
            if k == 0:
                r1 = 0
            elif n == n_cols - 1:
                r1 = int((rows[k, n, 0] + rows[k - 1, n, 0]) / 2)
            else:
                r1 = int((
                    rows[k, n, 0] + rows[k, n + 1, 0] +
                    rows[k - 1, n, 0] + rows[k - 1, n + 1, 0]
                ) / 4)

            if k == n_rows - 1:
                r2 = veg_mask.shape[0]
            elif n == n_cols - 1:
                r2 = int((rows[k, n, 0] + rows[k + 1, n, 0]) / 2)
            else:
                r2 = int((
                    rows[k, n, 0] + rows[k, n + 1, 0] +
                    rows[k + 1, n, 0] + rows[k + 1, n + 1, 0]
                ) / 4)

            c1 = rows[k, n, 1]

            if n == n_cols - 1:
                c2 = veg_mask.shape[1]
                value = rows[k, n, 0]
            else:
                c2 = rows[k, n + 1, 1]
                value = int((rows[k, n, 0] + rows[k, n + 1, 0]) / 2)

            distance_map[r1:r2, c1:c2] = value

    distance_map = np.abs(
        distance_map - np.arange(0, h).reshape((-1, 1)).repeat(w, axis=1)
    )
    # cv.namedWindow('test image', cv.WINDOW_NORMAL)
    # cv.resizeWindow('test image', 900, 576)
    # ci = array_image(distance_map)
    # ci = cv.applyColorMap(ci, cv.COLORMAP_HSV)
    # cv.imshow('test image', ci)
    # ret = cv.waitKey(0)
    return distance_map


def classification_model(
    pixels_crop: np.ndarray,
    pixels_weed: np.ndarray
):
    """Build a pixel classification model based on decision tree.

    Crop pixels are labeled as `0`, while weed pixels are labeled as `1`.

    Parameters
    ----------
    pixels_crop : array-like of shape = [p, 3] and ``dtype=np.uint8``
        Crop pixels data set, where each element represents a pixel of crop
        image regions.
    pixels_weed : array-like of shape = [q, 3] and ``dtype=np.uint8``
        Weeds pixels data set, where each element represents a pixel of weed
        image regions.

    Returns
    -------
    model : DecisionTreeClassifier object
        The DecisionTreeClassifier object. Please refer to
        ``help(sklearn.tree.DecisionTreeClassifier)`` for attributes of
        DecisionTreeClassifier object and basic usage of these attributes.

    """
    n_crops, n_weeds = pixels_crop.shape[0], pixels_weed.shape[0]
    inputs = np.vstack((pixels_crop, pixels_weed))
    outputs = np.ones((n_crops + n_weeds, 1), np.uint8)
    outputs[0:n_crops] = 0
    model = DecisionTreeClassifier()
    model.fit(inputs, outputs)
    return model


def fake_weeds(
    image: np.ndarray,
    veg_mask: np.ndarray,
    weed_images: List[np.ndarray],
    n_patches: int = 100,
    patch_density: float = 0.2,
    patch_size: Tuple[int, int] = (100, 100),
    resolution: float = 20
):
    """Add weed patches to a crop field image.

    Parameters
    ----------
    image : array-like of shape = [m, n, 3] and ``dtype=np.uint8``
        The crop field image.
    veg_mask : array-like of shape = [m, n] and ``dtype=np.uint8``
        Vegetation mask image, where non zero and zero elements represents
        vegetation and background pixels, respectively.
    weed_images : list of array-like of shape = [m, n, 4] and ``dtype=np.uint8``
        List of the four-channel weed image samples.
    n_patches : int, optional, (default=100)
        Number of weed patches to add.
    patch_density : float, optional, (default=0.2)
        Weed density inside each added weed patch.
    patch_size : tuple of ints with length = 2, optional, (default=(2, 2))
        Mean size (width x height) in meters of the weed patches
    resolution : float, optional, (default=20)
        Resolution in pixels/meter of the input mask image.


    Returns
    -------
    image_weeded : array-like of shape = [m, n, 3] and ``dtype=np.uint8``
        Crop field image with added weed patches.
    weed_mask : array-like of shape = [m, n] and ``dtype=np.uint8``
        Weed mask image, where non zero and zero elements represents weed and
        background pixels, respectively.

    """

    veg_density_thr = 0.1
    max_iter = 100000
    patch_size_var = 0.9
    rotation_aug = 5

    patch_size = tuple([int(size * resolution) for size in patch_size])

    weed_images_aug = []
    for weed_image in weed_images:
        for n in range(rotation_aug):
            weed_image, _ = rotate_image(
                weed_image,
                np.random.uniform(0, 2 * pi),
                interp=cv.INTER_LINEAR,
                border_mode=cv.BORDER_CONSTANT
            )
            weed_images_aug.append(weed_image)

    weed_images = weed_images_aug

    weed_areas = []
    weed_masks = []
    for weed_image in weed_images:
        alpha = weed_image[:, :, 3]
        mask = np.zeros(weed_image.shape[:2], np.uint8)
        mask[alpha > 250] = 255
        weed_masks.append(mask)
        weed_areas.append(np.count_nonzero(mask))

    n_weeds = len(weed_images)

    weed_mask = np.zeros_like(veg_mask)
    image_weeded = np.copy(image)

    x_max, y_max = image.shape[1] - 1, image.shape[0] - 1
    xw_max, yw_max = int(0.5 * patch_size[0]), int(0.5 * patch_size[1])
    xw_min, yw_min = patch_size_var * xw_max, patch_size_var * yw_max

    np.random.seed(int(time.time()))
    iter_count = 0
    n = 0
    while n < n_patches:
        x = np.random.randint(0, x_max)
        y = np.random.randint(0, y_max)
        wr = np.random.randint(xw_min, xw_max)
        hr = np.random.randint(yw_min, yw_max)
        x1, x2 = max(0, x - wr), min(x + wr + 1, x_max + 1)
        y1, y2 = max(0, y - hr), min(y + hr + 1, y_max + 1)
        if (x1 < 0) or (y1 < 0) or (x2 > x_max) or (y2 > y_max):
            continue
        patch_area = (2 * wr + 1) * (2 * hr + 1)
        veg_density = np.count_nonzero(veg_mask[y1:y2, x1:x2]) / patch_area
        if veg_density > veg_density_thr:
            continue

        max_weed_area = patch_area * patch_density
        weed_area = 0
        while weed_area < max_weed_area:
            weed_ind = np.random.randint(0, n_weeds)
            weed_image = weed_images[weed_ind]
            xw1, yw1 = np.random.randint(x1, x2), np.random.randint(y1, y2)
            xw2, yw2 = xw1 + weed_image.shape[1], yw1 + weed_image.shape[0]
            if (xw2 > x_max) or (yw2 > y_max):
                continue
            alpha = weed_image[:, :, 3] / 255.0
            alpha = np.dstack((alpha, alpha, alpha))
            image_weeded[yw1:yw2, xw1:xw2] = cv.add(
                (1 - alpha) * image[yw1:yw2, xw1:xw2],
                alpha * weed_image[:, :, 0:3]
            )
            weed_mask[yw1:yw2, xw1:xw2] = weed_masks[weed_ind][:]
            weed_area += np.count_nonzero(weed_mask[y1:y2, x1:x2])
            iter_count += 1

        n += 1
        iter_count += 1
        if iter_count > max_iter:
            break

    return image_weeded, weed_mask


