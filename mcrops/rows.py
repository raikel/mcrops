# -*- coding: utf-8 -*-
from math import pi
from typing import Tuple

import cv2 as cv
import numpy as np
from scipy.signal import convolve

from .utils import transform_matrix


def detect_direction(
    veg_mask: np.ndarray,
    window_shape: Tuple[float, float] = (10, 10),
    n_steps: int = 360,
    resolution: float = 20
) -> float:
    """Detect the crop rows direction in a vegetation mask of a crop field.

    The algorithm works as follows. A window is placed in the center of the
    vegetation mask image. This window is rotated in clock wise direction by
    uniformly spaced angles in the interval `[0, pi]`. Inside each window,
    a vegetation profile is computed. A vegetation profile is array where each
    element represents the number of non zero pixels along the corresponding
    column of pixels inside the region of the vegetation mask selected by the
    rotated window. For each vegetation profile, the peak-to-peak variation is
    computed and associated to the rotation angle of the corresponding window.
    The rotated window with the highest peak-to-peak vegetation profile is
    assumed to be perpendicular to the crop rows direction.

    Parameters
    ----------
    veg_mask : array-like of shape = [m, n] and ``dtype=np.uint8``
        Vegetation mask image, where non zero and zero elements represents
        vegetation and background pixels, respectively.
    window_shape : tuple, optional, (default=(200, 200))
        The shape of the window inside which vegetation profiles are computed.
    n_steps : int, optional, (default=360)
        Number of sample angles to rotate the profile window.
    resolution : float, optional, (default=20)
        Resolution in pixels/meter of the input mask image.

    Returns
    -------
    direction : float
        The detected mean crop rows direction.

    """
    max_resolution = 10

    scale = max_resolution / resolution
    if scale < 1:
        veg_mask = cv.resize(
            veg_mask,
            (0, 0),
            fx=scale,
            fy=scale,
            interpolation=cv.INTER_NEAREST
        )
        resolution = 10

    window_shape = (
        int(window_shape[0] * resolution),
        int(window_shape[1] * resolution)
    )

    image_shape = veg_mask.shape[0:2]

    center_point = np.array([
        image_shape[1] / 2,
        image_shape[0] / 2
    ])

    angles = np.linspace(0, pi, n_steps)

    y_grid, x_grid = np.mgrid[
                     0:int(window_shape[0]),
                     0:int(window_shape[1])
                     ]

    window_shape = y_grid.shape

    points = np.zeros((2, y_grid.size), dtype=y_grid.dtype)
    points[0, :] = x_grid.flat
    points[1, :] = y_grid.flat

    scores = np.zeros((angles.size,))
    for angle_ind, angle in enumerate(angles):
        tm = transform_matrix(angle=angle)[:2, :2]
        t_points = np.dot(tm, points)
        t_points = np.add(
            t_points,
            center_point.reshape((-1, 1)),
            dtype=np.int32,
            casting='unsafe'
        )

        t_points[0] = np.clip(t_points[0], a_min=0, a_max=image_shape[1] - 1)
        t_points[1] = np.clip(t_points[1], a_min=0, a_max=image_shape[0] - 1)

        outside = (
                (t_points[0] < 0) |
                (t_points[0] >= image_shape[1]) |
                (t_points[1] < 0) |
                (t_points[1] >= image_shape[0])
        ).reshape(window_shape)

        window = veg_mask[t_points[1], t_points[0]].reshape(window_shape)

        window[outside] = 0
        profile = window.sum(axis=0)
        scores[angle_ind] = profile.ptp()

    best_ind = np.argmax(scores)
    best_angle = angles[best_ind]
    return pi / 2 - best_angle


def detect_rows(
    veg_mask: np.ndarray,
    roi_mask: np.ndarray = None,
    row_sep: float = 0.7,
    extent_max: float = 5,
    extent_thr: float = 0.5,
    fusion_thr: float = 0.4,
    link_thr: int = 3,
    resolution: float = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect the crop rows in a vegetation mask of a crop field.

    The algorithm works as follows. A window is placed in the center of the
    vegetation mask image. This window is rotated in clock wise direction by
    uniformly spaced angles in the interval `[0, pi]`. Inside each window,
    a vegetation profile is computed. A vegetation profile is array where each
    element represents the number of non zero pixels along the corresponding
    column of pixels inside the region of the vegetation mask selected by the
    rotated window. For each vegetation profile, the peak-to-peak variation is
    computed and associated to the rotation angle of the corresponding window.
    The rotated window with the highest peak-to-peak vegetation profile is
    assumed to be perpendicular to the crop rows direction.

    Parameters
    ----------
    veg_mask : array-like of shape = [m, n] and ``dtype=np.uint8``
        Vegetation mask image, where non zero and zero elements represents
        vegetation and background pixels, respectively.
    roi_mask : array-like of shape = [m, n] and ``dtype=np.uint8``
        Region-of-interest mask image.
    row_sep : float, optional, (default=0.7)
        Rough estimated of the mean crop row separation.
    extent_max : float, optional, (default=5)
        The maximum length, in meters, of each line of pixels added to eval an
        element of a normalized vegetation profile.
    extent_thr : float, optional, (default=0.5)
        The minimum length, in meters, of each line of pixels added to eval an
        element of a normalized vegetation profile.
    fusion_thr : float, optional, (default=0.4)
        The maximum standard deviation of the row indices of a set of anchor
        points for these points to be fused into a single crop row line.
    link_thr : int, optional, (default=3)
        The minimum number of anchor points required to be fused in a single
        crop row line.
    resolution : float, optional, (default=20)
        Resolution in pixels/meter of the input mask image.

    Returns
    -------
    rows_ridges : array-like of shape = [r, c, 2] and ``dtype=np.int32``
        Detected crop rows lines, ridged aligned, where each element is crop
        row line represented by an array of `c` pairs of indexes in the form
        `(row, column)`. Each crop row line is a continuous polygonal line.
    rows_furrows : array-like of shape = [r, c, 2] and ``dtype=np.int32``
        Detected crop rows lines, furrow aligned, where each element is crop
        row line represented by an array of `c` pairs of indexes in the form
        `(row, column)`. Each crop row line is a continuous polygonal line.

    """
    extent_max = int(extent_max * resolution)
    extent_thr = int(extent_thr * resolution)
    fusion_thr *= resolution
    max_deviation = int(0.5 * row_sep * resolution)
    filter_size = int(0.5 * row_sep * resolution)
    peaks_mpd = int(0.5 * row_sep * resolution)

    (height, width) = veg_mask.shape
    n_profiles = int(width / extent_max)
    profile_ind = np.linspace(0, width - 1, n_profiles, False, dtype=np.int32)
    profiles_peaks = []

    extent_max *= 255
    extent_thr *= 255

    for ind in profile_ind:
        next_ind = min(width, ind + extent_max)
        distances = extent_max
        if roi_mask is not None:
            distances = roi_mask[:, ind:next_ind].sum(1, np.float32)
            distances[distances < extent_thr] = np.inf
        profile = veg_mask[:, ind:next_ind].sum(1, np.float32) / distances
        profile_peaks = _find_profile_peaks(
            profile,
            filter_size=filter_size,
            mpd=peaks_mpd,
        )
        profiles_peaks.append(profile_peaks)

    linked_peaks = []
    for k in range(n_profiles):
        n_peaks = profiles_peaks[k].size
        for n in range(n_peaks):
            if profiles_peaks[k][n] >= 0:
                rows = _link_peaks(
                    profiles_peaks, k, n, max_deviation, link_thr
                )
                if len(rows) > 0:
                    linked_peaks.append(rows)

    if len(linked_peaks) == 0:
        return np.array([], np.int32), np.array([], np.int32)

    linked_peaks = np.array(linked_peaks)
    linked_peaks = linked_peaks[linked_peaks.mean(1).argsort(), :]
    n_rows = linked_peaks.shape[0]

    linked_peaks_fused = []
    k = 0
    while k < n_rows:
        rows = [linked_peaks[k]]
        n = k + 1
        while (n + 1) < n_rows:
            if linked_peaks[k:(n + 1)].std(0).max() <= fusion_thr:
                rows.append(linked_peaks[n])
            else:
                break
            n += 1
        k = n
        linked_peaks_fused.append(np.array(rows).mean(0))

    linked_peaks = np.array(linked_peaks_fused)
    n_rows = linked_peaks.shape[0]

    ridges = np.empty((n_rows, n_profiles + 1, 2), linked_peaks.dtype)
    ridges[:, :-1, 0] = linked_peaks
    ridges[:, -1, 0] = linked_peaks[:, -1]
    ridges[:, :-1, 1] = profile_ind
    ridges[:, -1, 1] = width - 1

    furrows = np.empty((n_rows + 1, n_profiles + 1, 2), linked_peaks.dtype)
    furrows[1:-1, :, 0] = np.round((ridges[0:-1, :, 0] + ridges[1:, :, 0]) / 2.0)
    furrows[0, :, 0] = np.maximum(0, 2 * furrows[1, :, 0] - furrows[2, :, 0])
    furrows[-1, :, 0] = np.minimum(height - 1, 2 * furrows[-2, :, 0] - furrows[-3, :, 0])
    furrows[:, :, 1] = ridges[0, :, 1]

    return np.int32(ridges), np.int32(furrows)


def _find_profile_peaks(
        profile,
        filter_size: int = 100,
        mpd: int = 50,
        mph: float = 0.2,
        rpl: float = 0.95
) -> np.ndarray:
    mph = mph * profile.max()
    if filter_size > 0:
        profile = convolve(profile / filter_size, np.ones(filter_size), 'same')

    raw_peaks = _detect_peaks(profile, mph=mph, mpd=mpd)

    # Refine peaks locations
    n_peaks = raw_peaks.size
    peaks = np.zeros_like(raw_peaks)

    for k in range(n_peaks):
        this_peak = raw_peaks[k]

        if k != (n_peaks - 1):
            next_peak = raw_peaks[k + 1] + 1
        else:
            next_peak = profile.size

        if k != 0:
            prev_peak = raw_peaks[k - 1]
        else:
            prev_peak = 0

        dx = np.nonzero(
            profile[this_peak:next_peak] < rpl * profile[this_peak]
        )[0]

        if dx.size != 0:
            right_peak = this_peak + dx[0]
        else:
            right_peak = this_peak

        dx = np.nonzero(
            profile[this_peak:prev_peak:-1] < rpl * profile[this_peak]
        )[0]

        if dx.size != 0:
            left_pos = this_peak - dx[0]
        else:
            left_pos = this_peak

        peaks[k] = round((left_pos + right_peak) / 2)

    return peaks


def _detect_peaks(
        x: np.ndarray,
        mph: float = None,
        mpd: float = 1,
        threshold: float = 0,
        edge: str = 'rising',
        kpsh: bool = False
) -> np.ndarray:
    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks that are greater than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.

    Returns
    -------
    ind : 1D array_like
        indexes of the peaks in `x`.

    The function can handle NaN's
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)

    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where(
                (np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0)
            )[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where(
                (np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0)
            )[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(
            np.hstack((indnan, indnan - 1, indnan + 1))
        ), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(
            np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]),
            axis=0
        )
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind


def _link_peaks(profile_peaks, profile_ind, peak_ind, max_dev, link_thr=1):
    n_profiles = len(profile_peaks)
    linked_peaks = np.empty(n_profiles, np.int32)
    linked_peaks[0:(profile_ind + 1)] = profile_peaks[profile_ind][peak_ind]
    profile_peaks[profile_ind][peak_ind] = -(max_dev + 1)
    n_linked = 1
    for k in range(profile_ind + 1, n_profiles):
        if profile_peaks[k].size == 0:
            linked_peaks[k] = linked_peaks[k - 1]
            continue
        dev = np.abs(profile_peaks[k] - linked_peaks[k - 1])
        min_ind = dev.argmin()
        if dev[min_ind] < max_dev:
            linked_peaks[k] = profile_peaks[k][min_ind]
            profile_peaks[k][min_ind] = -(max_dev + 1)
            n_linked += 1
        else:
            linked_peaks[k] = linked_peaks[k - 1]

    if n_linked >= link_thr:
        return linked_peaks
    return []