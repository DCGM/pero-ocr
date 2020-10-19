import numpy as np
import cv2
import shapely
import sys
from scipy import ndimage
from scipy import signal
from scipy.ndimage.morphology import binary_erosion
from skimage.draw import polygon2mask

from pero_ocr.layout_engines import layout_helpers as helpers


class EngineLineDetectorSimple(object):
    def __init__(self, adaptive_threshold=91, block_size=21,
                 minimum_length=6, ignored_border_pixels=10):
        self.adaptive_threshold = adaptive_threshold
        self.block_size = block_size
        self.minimum_length = minimum_length
        self.ignored_border_pixels = ignored_border_pixels

    def detect_lines(self, img, region):
        """Performs simple line extraction in single text region using thresholding,
        correlation and connected component analysis.
        :param img: input image array
        :param region: target region polygon
        """

        baselines_list = []
        heights_list = []

        x1 = np.clip(np.amin(region[:, 0].astype(np.int32)), 0, img.shape[1])
        x2 = np.clip(np.amax(region[:, 0].astype(np.int32)), 0, img.shape[1])
        y1 = np.clip(np.amin(region[:, 1].astype(np.int32)), 0, img.shape[0])
        y2 = np.clip(np.amax(region[:, 1].astype(np.int32)), 0, img.shape[0])

        if x1 == x2 or y1 == y2:
            return [], [], []

        column_width = x2 - x1
        column_height = y2 - y1

        img_mask = polygon2mask(img.shape[0:2], np.flip(region, axis=1))
        img_mask = img_mask[y1:y2, x1:x2]
        img_mask = binary_erosion(img_mask, structure=np.ones((1, 2 * self.ignored_border_pixels + 1)))

        img_crop = img[y1:y2, x1:x2, :]
        img_crop = img_crop.mean(axis=2).astype(np.uint8)
        img_crop = cv2.adaptiveThreshold(img_crop, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, self.block_size, self.adaptive_threshold) == 0

        img_crop = img_crop * img_mask

        img_crop_labeled, num_features = ndimage.measurements.label(img_crop)
        proj = np.sum(img_crop, axis=1)
        corr = np.correlate(proj, proj, mode='full')[proj.shape[0]:]
        corr_peaks = signal.find_peaks(corr, prominence=0, distance=1)[0]
        if len(corr_peaks) > 0:
            line_period = float(signal.find_peaks(corr, prominence=0, distance=1)[0][0])
        else:
            line_period = 1
        target_signal = - np.diff(proj)
        target_signal[target_signal < 0] = 0

        baseline_coords = signal.find_peaks(target_signal, distance=int(round(0.85*line_period)))[0]
        region = shapely.geometry.polygon.Polygon(region)
        used_inds = []

        for baseline_coord in baseline_coords[::-1]:
            valid_baseline = True
            matching_objects = np.unique(img_crop_labeled[baseline_coord-10, :])[1:]
            if len(matching_objects) > 0:
                for ind in matching_objects:
                    if ind in used_inds:
                        valid_baseline = False
                    used_inds.append(ind)

                for yb1 in range(baseline_coord, 0, -3):
                    line_inds_to_check = img_crop_labeled[yb1, :]
                    if not np.any(np.intersect1d(matching_objects, line_inds_to_check)):
                        break

                for yb2 in range(baseline_coord, column_height, 3):
                    line_inds_to_check = img_crop_labeled[yb2, :]
                    if not np.any(np.intersect1d(matching_objects, line_inds_to_check)):
                        break

                xb1, xb2 = 0, column_width

                if xb2 - xb1 < self.minimum_length:
                    valid_baseline = False

                line = shapely.geometry.LineString([[x1+xb1, y1+baseline_coord],
                                                    [x1+xb2, y1+baseline_coord]])
                intersection = region.intersection(line)
                if intersection.geom_type == 'LineString':
                    if valid_baseline:
                        baselines_list.append(np.round(np.asarray(list(region.intersection(line).coords[:]))).astype(np.int16))
                        heights_list.append([baseline_coord-yb1, yb2-baseline_coord])

        textlines_list = [helpers.baseline_to_textline(baseline, heights) for baseline, heights in zip(baselines_list, heights_list)]

        return baselines_list, heights_list, textlines_list
