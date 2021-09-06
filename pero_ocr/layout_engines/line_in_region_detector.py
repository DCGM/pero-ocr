import numpy as np
import cv2
from scipy import signal

from pero_ocr.layout_engines import layout_helpers as helpers


def detect_lines_in_region(region, detection_maps, downsample, line_detection_threshold=0.2):
    """
    Detects straight textlines inside a single region.

    :param region: numpy array of polygon points
    :param detection_maps: channel 0: ascender heights, channel 1: descender heights, channel 2: baseline detections,
    channel 3: baseline endpoints, channel 4: region detections
    :return: list of baselines, list of heights, list of textline polygons
    """

    region_polygon = np.stack([
        np.clip(region[:, 0] / downsample, 1, detection_maps.shape[1] - 2),
        np.clip(region[:, 1] / downsample, 1, detection_maps.shape[0] - 2)],
        axis=1
    )
    region_bb_lt = np.round(np.amin(region_polygon, axis=0) - 1).astype(np.int32)
    region_bb_rb = np.round(np.amax(region_polygon, axis=0) + 1).astype(np.int32)
    region_maps = detection_maps[region_bb_lt[1]:region_bb_rb[1], region_bb_lt[0]:region_bb_rb[0]]

    region_polygon -= region_bb_lt[np.newaxis]

    polygon_mask = np.zeros(region_maps.shape[0:2], dtype=np.float32)

    cv2.fillPoly(polygon_mask, [np.round(region_polygon).astype(np.int32)], 1.0)
    region_maps = region_maps * polygon_mask[:, :, np.newaxis]

    contours, hierarchy = cv2.findContours((region_maps[:, :, 2] > line_detection_threshold).astype(np.uint8),
                                          cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cov_mat = np.zeros([2, 2])
    for contour in contours:
        contour = contour[:, 0]
        centralized = contour - contour.mean(axis=0)
        cov_mat += centralized.T.dot(centralized)
    eig_val, eig_vec = np.linalg.eig(cov_mat)
    direction = eig_vec[np.argmax(eig_val)]
    if direction[0] < 0:
        direction *= -1
    rad_angle = np.arctan2(direction[1], direction[0])

    T = cv2.getRotationMatrix2D(tuple(np.asarray(region_maps.shape[0:2]) * 0.5), -rad_angle / np.pi * 180, 1)
    T = np.concatenate((T, np.array([[0, 0, 1]])), axis=0)

    transformed_polygon = cv2.transform(region_polygon[np.newaxis], T[:2, :])
    transformed_polygon = transformed_polygon[0]

    polygon_lt = np.amin(transformed_polygon, axis=0)
    polygon_rb = np.amax(transformed_polygon, axis=0)

    M_trans = np.array([
        [1, 0, -polygon_lt[0]],
        [0, 1, -polygon_lt[1]],
        [0, 0, 1]
    ])
    T = np.dot(T, M_trans)
    output_size = tuple((polygon_rb - polygon_lt + 1).astype(int))

    region_map = cv2.warpAffine(region_maps[:, :, :3], T[:2, :], output_size)
    polygon_mask = cv2.warpAffine(polygon_mask, T[:2, :], output_size)

    region_map[:, :, 2][region_map[:, :, 2] < line_detection_threshold] = 0
    detection_projections = np.sum(region_map[:, :, 2], axis=1) / output_size[0]

    mean_height = np.average((region_map[:, :, 0] + region_map[:, :, 1])[polygon_mask > 0])
    baselines_y, baselines_y_float = find_peaks(detection_projections, min_distance=np.maximum(0.7*mean_height, 1))

    if baselines_y.shape[0] == 0:
        return [], [], []

    baselines_x0 = np.argmax(polygon_mask, axis=1)[baselines_y]  # first x of polygon mask
    baselines_x1 = (polygon_mask.shape[1] - np.argmax(polygon_mask[:, ::-1], axis=1))[baselines_y]  # last x of polygon mask

    baselines = np.stack((
        np.stack((baselines_x0, baselines_x1), axis=1),
        np.stack((baselines_y_float, baselines_y_float), axis=1)),
        axis=2
    )
    baselines = cv2.transform(baselines.astype(np.float32), np.linalg.inv(T)[:2, :])
    baselines = (baselines + region_bb_lt[np.newaxis] + 1) * downsample

    b_list = [b for b in baselines]

    h_list = []
    for by in baselines_y:
        asc_line = region_map[by, :, 0]
        asc = np.percentile(asc_line[region_map[by, :, 2] > line_detection_threshold], 70)
        des_line = region_map[by, :, 1]
        des = np.percentile(des_line[region_map[by, :, 2] > line_detection_threshold], 70)
        h_list.append([asc * downsample, des * downsample])

    t_list = [helpers.baseline_to_textline(b, h) for b, h in zip(b_list, h_list)]

    return b_list, h_list, t_list


def find_peaks(array, min_distance=1, min_height=0.05):
    """
    Detects peaks in 1D array with subpixel precision.

    :param array: 1D numpy array of values
    :param min_distance: Minimum distance of individual peaks
    :param min_height: Minimum height of peaks to avoid noise
    :return: 1D array of integer peak positions, 1D array of float peak positions
    """
    # array = np.concatenate((array, [0]), axis=0)
    peaks, _ = signal.find_peaks(array, distance=min_distance, height=min_height)

    peaks_float = peaks.copy().astype(np.float)
    for i, x in enumerate(peaks):
        xs = np.clip(np.array(range(x - 2, x + 3)), 0, array.shape[0]-1)
        ys = array[xs]
        p = np.polyfit(xs, ys, 2)
        peaks_float[i] = -p[1] / (2 * p[0])

    return peaks, peaks_float












