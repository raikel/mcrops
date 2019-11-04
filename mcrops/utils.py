# -*- coding: utf-8 -*-
from typing import Tuple
import numpy as np
import cv2 as cv


def draw_rows(
    image: np.ndarray,
    rows: np.ndarray,
    line_color: Tuple[int, int, int] = (255, 0, 0),
    line_width: int = 2
):
    """Draw crop rows lines in a crop field image.

    Parameters
    ----------
    image : array-like of shape = {[m, n, 3], [m, n]} and ``dtype=np.uint8``
        Color or grayscale crop field image.
    rows : array-like of shape = [r, c, 2] and ``dtype=np.int32``
        Crop rows lines, where each element is crop row line represented by an
        array of `c` pairs of indexes in the form `(row, column)`. Each crop
        row line is a continuous polygonal line.
    line_color : tuple of ints with length = 3, optional, (default=(255, 0, 0))
        Color of drawn lines, in BGR format.
    line_width : int, optional, (default=2)
        Width of drawn lines.

    Returns
    -------
    image : array-like of shape = {[m, n, 3], [m, n]} and ``dtype=np.uint8``
        Image with drawn crop rows lines. Input `image` is not modified, drawing
        is done in a copy of it.

    """
    if image.ndim == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    else:
        image = image.copy()

    for k in range(rows.shape[0]):
        for n in range(rows.shape[1] - 1):
            p1 = (rows[k, n, 1], rows[k, n, 0])
            p2 = (rows[k, n + 1, 1], rows[k, n + 1, 0])
            cv.line(image, p1, p2, line_color, line_width, lineType=cv.LINE_AA)

    return image


def poly_mask(poly: np.ndarray, shape: tuple):
    roi_mask = np.zeros(shape, dtype=np.uint8)
    cv.fillConvexPoly(roi_mask, poly, 255)
    return roi_mask


def array_image(
    values: np.ndarray,
    colormap: int = None,
    full_scale: bool = True
):
    if values.dtype != np.uint8 or full_scale:
        value_max = float(np.max(values))
        value_min = float(np.min(values))
        if value_max <= 255 and value_min >= 0 and not full_scale:
            values = np.uint8(values)
        elif not full_scale or value_min == value_max:
            values = np.clip(values, 0, 255, dtype=np.uint8, casting='unsafe')
        else:
            m = 255.0 / (value_max - value_min)
            values = np.uint8(m * (values - value_min))

    if colormap is not None:
        values = cv.applyColorMap(values, colormap)

    return values


# Rotation is CCW with rotation center in upper left corner
def rotate_image(
    image: np.ndarray,
    angle: float,
    interp: int = cv.INTER_NEAREST,
    border_mode: int = cv.BORDER_TRANSPARENT,
    border_value: int = 0
):
    (h, w) = image.shape[0:2]
    matrix = transform_matrix(angle)
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]])
    corners = np.int32(np.dot(matrix, corners.T)).T[:, 0:-1]
    x, y, w, h = cv.boundingRect(corners.reshape(1, -1, 2))
    matrix[0:2, -1] = [-x, -y]
    image = cv.warpAffine(
        image,
        matrix[0:2, :],
        (w, h),
        flags=interp,
        borderMode=border_mode,
        borderValue=border_value
    )
    return image, matrix


def trim_image(img, box):
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    return img[y1:y2, x1:x2]


def trim_poly(poly, box):
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    return np.minimum(np.maximum(poly, [x1, y1]), [x2, y2])


# Rotation is CCW
def transform_matrix(
    angle: float = 0,
    translate: tuple = (0, 0),
    scale: tuple = (1.0, 1.0)
):
    s, c = np.sin(angle), np.cos(angle)
    sx, sy = scale[0], scale[1]
    dx, dy = translate[0], translate[1]
    matrix = np.array([[sx*c, sy*s, dx], [-sx*s, sy*c, dy], [0, 0, 1]])
    return matrix