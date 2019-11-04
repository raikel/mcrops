# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np

from .utils import trim_poly, rotate_image, trim_image, transform_matrix


def segment_vegetation(image: np.ndarray, threshold: float = 1):
    """Segment vegetation pixels of a crop field image.

    Parameters
    ----------
    image : array-like of shape = [m, n, 3] and ``dtype=np.uint8``
        The crop field image to be segmented.
    threshold : float, optional, (default=1)
        Threshold segmentation value.

    Returns
    -------
    mask : array-like of shape = [m, n] and ``dtype=np.int32``
        A mask image where non zero and zero elements represents
        vegetation and background pixels in the input image, respectively.

    """

    if image.dtype != np.float:
        image = np.float32(image)
    b, g, r = cv.split(image)
    index = (2 * g - r - b)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[index > threshold] = 255
    return mask


def norm_image(
    image: np.ndarray,
    roi_poly: np.ndarray = None,
    roi_trim: bool = True,
    rows_direction: float = 0,
    is_mask: bool = False
):
    """Normalize a crop field image.

    A crop field image is normalized by rotating it by and angle equal in
    magnitude to the crop rows direction, but in opposite direction, such that
    in the normalized image the crop rows are horizontally oriented. Optionally,
    the image can be cropped to an specified region-of-interest (ROI).

    Parameters
    ----------
    image : array-like of shape = [m, n, 3] or [m, n] and ``dtype=np.uint8``
        Crop field image. Can be a multi-channel or single-channel image.
    roi_poly : array-like of shape = [r, 1, 2] and ``dtype=np.int32``
        Convex polygon that encloses the ROI of the input image, in the format
        returned by ``help(cv2.contourArea)``.
    roi_trim : bool, optional, (default=True)
        Whether to trim the image to the roi area.
    rows_direction : float, optional, (default=0.0)
        The crop rows mean direction, clock wise, in radians.
    is_mask: bool, optional, (default=False)
        Whether `image` is a binary mask image. This controls the interpolation
        method used internally to transform images. For binary mask images, the
        method used is `cv.INTER_NEAREST`, otherwise the method used is
        `cv.INTER_LINEAR`.

    Returns
    -------
    image : array-like of shape = [m, n, 3] or [m, n] and ``dtype=np.int32``
        An array of the same shape as the input `image`, representing the
        normalized image.
    roi_poly: array-like of shape = [r, 1, 2] and ``dtype=np.int32``
        The modified input `roi_poly`, after normalization
    t_matrix: array-like of shape = [3, 3] and ``dtype=np.float``
        The transformation matrix that make the normalization.

    """

    (h, w) = image.shape[0:2]
    interp = cv.INTER_NEAREST if is_mask else cv.INTER_LINEAR

    if roi_poly is not None:
        roi_poly = np.array(roi_poly, np.int32)
        roi_poly = np.reshape(roi_poly, (-1, 1, 2))
        roi_poly = trim_poly(roi_poly, (0, 0, w, h))

    image, t_matrix = rotate_image(image, rows_direction, interp=interp)
    (h, w) = image.shape[0:2]

    if roi_poly is not None:
        roi_poly = cv.transform(roi_poly, t_matrix[0:2, :])
        roi_poly = trim_poly(roi_poly, (0, 0, w, h))

    if roi_trim and roi_poly is not None:
        roi_box = cv.boundingRect(roi_poly)
        image = trim_image(image, roi_box)
        box_corner = (roi_box[0], roi_box[1])
        roi_poly -= [box_corner]
        t_matrix = np.dot(
            np.linalg.inv(t_matrix),
            transform_matrix(translate=box_corner)
        )

    roi_mask = np.zeros(image.shape[0:2], np.uint8)
    cv.fillConvexPoly(roi_mask, roi_poly, (255,))

    image[roi_mask == 0] = 0

    return image, roi_poly, t_matrix


def mask_density(
    mask: np.ndarray,
    roi_mask: np.ndarray = None,
    cell_size: tuple = (5, 5),
    resolution: float = 20
):
    """Compute a density map from a mask image.

    Parameters
    ----------
    mask : array-like of shape = [m, n] and ``dtype=np.uint8``
        Mask image formed by non zero and zero elements.
    roi_mask : array-like of shape = [m, n] and ``dtype=np.uint8``
        Region-of-interest mask image.
    cell_size : tuple, optional, (default=(10, 10))
        Size (width x height) in meters of each rectangular cell of the
        density map grid.
    resolution : float, optional, (default=20)
        Resolution in pixels/meter of the input mask image.

    Returns
    -------
    density_map : array-like of shape = [m, n] and ``dtype=np.float32``
        An array of the same shape as the input `image`, representing the
        density map. To compute the density map, the input image is divided by
        an uniform grid with cell size `cell_size`. Then, inside each cell, the
        density is computed as the ratio between the number non zero pixels
        that belongs to `mask` and the number of non zero pixels from
        `roi_mask`.

    """
    roi_ratio_thr = 0.1
    cell_size = tuple([int(s * resolution) for s in cell_size])

    density_map = np.zeros(mask.shape[0:2], np.float32)

    def set_density(cell: tuple):
        cell_area = roi_area = (cell[1] - cell[0]) * (cell[3] - cell[2])
        if cell_area <= 0:
            return

        if roi_mask is not None:
            roi_area = np.count_nonzero(
                roi_mask[cell[0]:cell[1], cell[2]:cell[3]]
            )

        density = 0
        if (roi_area / cell_area) > roi_ratio_thr:
            density = np.count_nonzero(
                mask[cell[0]:cell[1], cell[2]:cell[3]]
            ) / roi_area
        density_map[cell[0]:cell[1], cell[2]:cell[3]] = density

    h, w = mask.shape[0:2]
    r1 = 0
    while r1 < h:
        c1 = 0
        r2 = min(r1 + cell_size[1], h)
        while c1 < w:
            c2 = min(c1 + cell_size[0], w)
            set_density((r1, r2, c1, c2))
            c1 += cell_size[0]
        r1 += cell_size[1]

    return density_map


def detect_roi(
    veg_mask,
    row_sep: float = 0.7,
    resolution: float = 20,
    min_ratio: float = 0.1,
    max_ratio: float = 0.9
):
    """Detect the region-of-interest in the vegetation mask of a crop field.

    Parameters
    ----------
    veg_mask : array-like of shape = [m, n] and ``dtype=np.uint8``
        Vegetation mask image, where non zero and zero elements represents
        vegetation and background pixels, respectively.
    row_sep : float, optional, (default=0.7)
        Rough estimated of the mean crop row separation.
    resolution : float, optional, (default=20)
        Resolution in pixels/meter of the input mask image.
    min_ratio : float, optional, (default=0.1)
        If the ratio between the area of the detected ROI and the full image
        area if higher than `min_ratio` or lower `max_ratio`, the detected ROI
        is discarded and it is assumed as the whole image.
    max_ratio : float, optional, (default=0.1)
        If the ratio between the area of the detected ROI and the full image
        area if higher than `min_ratio` or lower `max_ratio`, the detected ROI
        is discarded and it is assumed as the whole image.

    Returns
    -------
    roi_poly : array-like of shape = [r, 1, 2] and ``dtype=np.int32``
        Convex polygon that encloses the detected ROI, in the format
        returned by ``help(cv2.contourArea)``.

    """
    dilation = int(2 * row_sep * resolution)
    if dilation > 0:
        se_rect = cv.getStructuringElement(
            cv.MORPH_RECT, (dilation, dilation)
        )
        veg_mask = cv.morphologyEx(veg_mask, cv.MORPH_DILATE, se_rect)

    contours, _ = cv.findContours(
        veg_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )
    contours_area = list(map(cv.contourArea, contours))
    max_ind = int(np.argmax(contours_area))
    area_ratio = contours_area[max_ind] / veg_mask.size

    if min_ratio < area_ratio < max_ratio:
        roi_poly = cv.convexHull(contours[max_ind])
    else:
        (h, w) = veg_mask.shape
        roi_poly = np.int32(
            [[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]]
        )
    return roi_poly
