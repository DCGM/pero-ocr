import numpy as np
from copy import deepcopy
import time

import cv2
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import skimage.draw
import shapely.geometry as sg

from pero_ocr.layout_engines import layout_helpers as helpers
from pero_ocr.layout_engines.torch_parsenet import TorchParseNet, TorchOrientationNet


class LineFilterEngine(object):

    def __init__(self, model_path, device, downsample=4, max_mp=5):
        self.tiltnet = TorchOrientationNet(
            model_path,
            device=device,
            max_mp=max_mp
        )
        self.downsample = downsample

    @staticmethod
    def get_angle_diff(angle_1, angle_2):
        smaller = np.minimum(angle_1, angle_2)
        larger = np.maximum(angle_1, angle_2)
        diff = np.minimum(
            np.abs(larger - smaller),
            np.abs(larger - (smaller + 2 * np.pi)))
        return diff

    def predict_directions(self, image):
        self.predictions = self.tiltnet.get_maps(image, self.downsample)

    def check_line_rotation(self, polygon, baseline):
        line_mask = skimage.draw.polygon2mask(
            self.predictions.shape[:2], np.flip(polygon, axis=1) / self.downsample)

        target_angle = np.arctan2(
            baseline[0, 1] - baseline[-1, 1], baseline[-1, 0] - baseline[0, 0])

        predicted_x = np.median(self.predictions[:, :, 0][line_mask > 0])
        predicted_y = np.median(self.predictions[:, :, 1][line_mask > 0])
        predicted_angle = np.arctan2(predicted_y, predicted_x)

        # If line is horizontal, keep it anyway because its safe to assume we want to keep it
        if target_angle < np.pi/4 and target_angle > -np.pi/4:
            return True
        # If line is not horizontal, check it against the OrientationNet output and keep it if they differ by less than pi/4
        else:
            return self.get_angle_diff(predicted_angle, target_angle) < np.pi/4


class LayoutEngine(object):
    def __init__(self, model_path, device, downsample=4, max_mp=5, detection_threshold=0.2, adaptive_downsample=True,
                 line_end_weight=1.0, vertical_line_connection_range=5, smooth_line_predictions=True,
                 paragraph_line_threshold=0.3):
        self.parsenet = TorchParseNet(
            model_path,
            downsample=downsample,
            adaptive_downsample=adaptive_downsample,
            device=device,
            max_mp=max_mp,
            detection_threshold=detection_threshold
        )

        self.line_end_weight = line_end_weight
        self.vertical_line_connection_range = vertical_line_connection_range
        self.smooth_line_predictions = smooth_line_predictions
        self.line_detection_threshold = detection_threshold
        self.adaptive_downsample = adaptive_downsample

        self.paragraph_line_threshold = paragraph_line_threshold

        params = ' '.join([f'{name}:{str(getattr(self, name))}'
                  for name in ['line_end_weight', 'vertical_line_connection_range', 'smooth_line_predictions', 'line_detection_threshold', 'adaptive_downsample']])
        print(f'LayoutEngine params are {params}')

    def get_heights(self, heights_map, ds, inds):

        inds /= ds
        y_inds = np.clip(
            np.round(inds[:, 1]).astype(np.int), 0, heights_map.shape[0]-1)
        x_inds = np.clip(
            np.round(inds[:, 0]).astype(np.int), 0, heights_map.shape[1]-1)

        heights_pred = heights_map[(y_inds, x_inds)]

        heights_pred = np.maximum(heights_pred, 0)
        heights_pred = np.asarray([
            np.percentile(heights_pred[:, 0], 70),
            np.percentile(heights_pred[:, 1], 70)
        ])
        return heights_pred * ds

    def detect(self, image, rot=0):
        """Uses parsenet to find lines and region separators, clusters vertically
        close lines by computing penalties and postprocesses the resulting
        regions.
        :param image: input image
        :param rot: number of counter-clockwise 90degree rotations (0 <= n <= 3)
        """
        if rot > 0:
            image = np.rot90(image, k=rot)

        tic = time.time()
        maps, ds = self.parsenet.get_maps_with_optimal_resolution(image)
        print(f'GET MAPS TIME: {time.time() - tic}')

        b_list, h_list, t_list = self.parse(maps, ds)

        if not b_list:
            return [], [], [], []

        clusters_array = self.make_clusters(b_list, h_list, t_list, maps[:, :, 4], ds)
        p_list = self.clustered_lines_to_polygons(t_list, clusters_array)

        b_list, h_list, t_list = helpers.order_lines_vertical(b_list, h_list, t_list)
        p_list, b_list, t_list = self.rotate_layout(p_list, b_list, t_list, rot, image.shape)

        return p_list, b_list, h_list, t_list

    def parse(self, out_map, downsample):
        """Parse input baseline, height and region map into list of baselines
        coords, list of heights and region map
        :param out_map: array of baseline and endpoint probabilities with
        channels: ascender height, descender height, baselines, baseline
        endpoints, region boundaries
        :param downsample: downsample factor to apply to layout coords
        """
        b_list = []
        h_list = []

        print('MAP RES:', out_map.shape)
        out_map[:, :, 4][out_map[:, :, 4] < 0] = 0

        # expand line heights verticaly
        heights_map = ndimage.morphology.grey_dilation(
            out_map[:, :, :2], size=(5, 1, 1))

        baselines_map = out_map[:, :, 2]
        if self.smooth_line_predictions:
            baselines_map = ndimage.convolve(baselines_map, np.ones((3, 3))/9)
        baselines_map = nonmaxima_suppression(baselines_map, element_size=(5, 1))
        baselines_map = (baselines_map - self.line_end_weight * out_map[:, :, 3]) > self.line_detection_threshold

        # connect vertically disconnected lines - any effect? Parameter is vertical connection distance in pixels.
        baselines_map_dilated = ndimage.morphology.binary_dilation(
            baselines_map, structure=np.asarray([[1, 1, 1] for i in range(self.vertical_line_connection_range)]))
        baselines_img, num_detections = ndimage.measurements.label(baselines_map_dilated, structure=np.ones([3, 3]))
        baselines_img *= baselines_map
        inds = np.where(baselines_img > 0)
        labels = baselines_img[inds[0], inds[1]]

        for i in range(1, num_detections+1):
            bl_inds, = np.where(labels == i)
            if len(bl_inds) > 5:
                # go from matrix indexing to image indexing
                pos_all = np.stack([inds[1][bl_inds], inds[0][bl_inds]], axis=1)

                _, indices = np.unique(pos_all[:, 0], return_index=True)
                pos = pos_all[indices]
                x_index = np.argsort(pos[:, 0])
                pos = pos[x_index]

                target_point_count = min(10, pos.shape[0] // 10)
                target_point_count = max(target_point_count, 2)
                selected_pos = np.linspace(
                    0, (pos.shape[0]) - 1, target_point_count).astype(np.int32)

                pos = pos[selected_pos, :]
                pos[0, 0] -= 2  # compensate for endpoint detection overlaps
                pos[-1, 0] += 2

                heights_pred = heights_map[inds[0][bl_inds], inds[1][bl_inds], :]

                heights_pred = np.maximum(heights_pred, 0)
                heights_pred = np.asarray([
                    np.percentile(heights_pred[:, 0], 50),
                    np.percentile(heights_pred[:, 1], 50)
                ])

                b_list.append(downsample * pos.astype(np.float))
                h_list.append([downsample * heights_pred[0], downsample * heights_pred[1]])

        # sort lines from LEFT to RIGHT
        x_inds = [np.amin(baseline[:, 0]) + 0.0001 * np.random.rand() for baseline in b_list]
        b_list = [b for _, b in sorted(zip(x_inds, b_list))]
        h_list = [h for _, h in sorted(zip(x_inds, h_list))]

        t_list = [helpers.baseline_to_textline(b, h) for b, h in zip(b_list, h_list)]

        return b_list, h_list, t_list

    def rotate_layout(self, p_list, b_list, t_list, rot, shape):
        if rot == 1:
            b_list = [np.flip(b, axis=1) for b in b_list]
            t_list = [np.flip(t, axis=1) for t in t_list]
            p_list = [np.flip(p, axis=1) for p in p_list]
            for b in b_list:
                b[:, 0] = shape[0] - b[:, 0]
            for t in t_list:
                t[:, 0] = shape[0] - t[:, 0]
            for p in p_list:
                p[:, 0] = shape[0] - p[:, 0]
        elif rot == 2:
            shape_array = np.asarray(shape[:2][::-1])
            b_list = [shape_array - b for b in b_list]
            t_list = [shape_array - t for t in t_list]
            p_list = [shape_array - p for p in p_list]
        elif rot == 3:
            b_list = [np.flip(b, axis=1) for b in b_list]
            t_list = [np.flip(t, axis=1) for t in t_list]
            p_list = [np.flip(p, axis=1) for p in p_list]
            for b in b_list:
                b[:, 1] = shape[1] - b[:, 1]
            for t in t_list:
                t[:, 1] = shape[1] - t[:, 1]
            for p in p_list:
                p[:, 1] = shape[1] - p[:, 1]
        return p_list, b_list, t_list

    def filter_polygons(self, polygons, region_textlines):
        polygons = [helpers.check_polygon(polygon) for polygon in polygons]
        inds_to_remove = []
        for i in range(len(polygons)):
            for j in range(i+1, len(polygons)):
                # first check if a polygon is completely inside another, remove the smaller in that case
                if polygons[i].contains(polygons[j]):
                    inds_to_remove.append(j)
                elif polygons[j].contains(polygons[i]):
                    inds_to_remove.append(i)
                elif polygons[i].intersects(polygons[j]):
                    poly_intersection = polygons[i].intersection(polygons[j])
                    # remove the overlap from both regions
                    poly_tmp = deepcopy(polygons[i])
                    polygons[i] = polygons[i].difference(polygons[j])
                    polygons[j] = polygons[j].difference(poly_tmp)
                    # append the overlap to the one with more textlines in the overlap area
                    score_i = 0
                    for line in region_textlines[i]:
                        line_poly = helpers.check_polygon(sg.Polygon(line))
                        score_i += line_poly.intersection(poly_intersection).area
                    score_j = 0
                    for line in region_textlines[j]:
                        line_poly = helpers.check_polygon(sg.Polygon(line))
                        score_j += line_poly.intersection(poly_intersection).area
                    if score_i > score_j:
                        polygons[i] = polygons[i].union(poly_intersection)
                    else:
                        polygons[j] = polygons[j].union(poly_intersection)
        return [polygon for i, polygon in enumerate(polygons) if i not in inds_to_remove]


    def get_penalty(self, b, shift, x_1, x_2, map, t=1):
        b_shifted = np.round(b).astype(np.int32)
        b_shifted[:, 1] += int(round(shift))
        x_1_shifted = int(round(x_1)) - np.amin(b_shifted[:, 0])
        x_2_shifted = int(round(x_2)) - np.amin(b_shifted[:, 0])
        map_crop = map[
                   np.clip(np.amin(b_shifted[:, 1]-t), 0, map.shape[0]-1): np.clip(np.amax(b_shifted[:, 1]+t+1), 0, map.shape[0]-1),
                   np.amin(b_shifted[:, 0]): np.amax(b_shifted[:, 0])
                   ]

        b_shifted[:, 1] -= (np.amin(b_shifted[:, 1]) - t)
        b_shifted[:, 0] -= np.amin(b_shifted[:, 0])

        penalty_mask = np.zeros_like(map_crop)
        for b_ind in range(b_shifted.shape[0]-1):
            cv2.line(penalty_mask, tuple(b_shifted[b_ind, :]), tuple(b_shifted[b_ind+1, :]), color=1, thickness=(2*t)+1)

        penalty_area = penalty_mask * map_crop

        return np.sum(penalty_area[:, x_1_shifted:x_2_shifted]) / (x_2 - x_1)


    def get_pair_penalty(self, b1, b2, h1, h2, map, ds):
        x_overlap = max(0, min(np.amax(b1[:, 0]), np.amax(b2[:, 0])) - max(np.amin(b1[:, 0]), np.amin(b2[:, 0])))
        if x_overlap > 5:
            x_1 = int(max(np.amin(b1[:, 0]), np.amin(b2[:, 0])))
            x_2 = int(min(np.amax(b1[:, 0]), np.amax(b2[:, 0])))
            if np.average(b1[:, 1]) > np.average(b2[:, 1]):
                penalty_1 = self.get_penalty(b1/ds, -h1[0]/ds, x_1/ds, x_2/ds, map)
                penalty_2 = self.get_penalty(b2/ds, h2[1]/ds, x_1/ds, x_2/ds, map)
            else:
                penalty_1 = self.get_penalty(b1/ds, h1[1]/ds, x_1/ds, x_2/ds, map)
                penalty_2 = self.get_penalty(b2/ds, -h2[0]/ds, x_1/ds, x_2/ds, map)
            penalty = np.abs(max(penalty_1, penalty_2))
        else:
            penalty = 1
        return penalty


    def clustered_lines_to_polygons(self, t_list, clusters_array):
        regions_textlines_tmp = []
        polygons_tmp = []
        for i in range(np.amax(clusters_array) + 1):
            region_textlines = []
            for textline, cluster in zip(t_list, clusters_array):
                if cluster == i:
                    region_textlines.append(textline)

            region_poly = helpers.region_from_textlines(region_textlines)
            regions_textlines_tmp.append(region_textlines)
            polygons_tmp.append(region_poly)

        # remove overlaps while minimizing textline modifications
        polygons_tmp = self.filter_polygons(
            polygons_tmp, regions_textlines_tmp)
        # up to this point, polygons can be any geometry that comes from alpha_shape
        p_list = []
        for region_poly in polygons_tmp:
            if region_poly.is_empty:
                continue
            if region_poly.geom_type == 'MultiPolygon':
                for poly in region_poly:
                    if not poly.is_empty:
                        p_list.append(poly.simplify(5))
            if region_poly.geom_type == 'Polygon':
                p_list.append(region_poly.simplify(5))
        return [np.array(poly.exterior.coords) for poly in p_list]

    def make_clusters(self, b_list, h_list, t_list, layout_separator_map, ds):
        if len(t_list) > 1:

            min_pos = np.zeros([len(t_list), 2], dtype=np.float32)
            max_pos = np.zeros([len(t_list), 2], dtype=np.float32)

            t_list_dilated = []
            for textline, min_, max_ in zip(t_list, min_pos, max_pos):
                textline_poly = sg.Polygon(textline)
                tot_height = np.abs(textline[0, 1] - textline[-1, 1])
                t_list_dilated.append(textline_poly.buffer(3*tot_height/4))
                min_[:] = textline.min(axis=0) - tot_height
                max_[:] = textline.max(axis=0) + tot_height

            candidates = np.logical_and(
                np.logical_or(
                    max_pos[:, np.newaxis, 1] <= min_pos[np.newaxis, :, 1],
                    min_pos[:, np.newaxis, 1] >= max_pos[np.newaxis, :, 1]),
                np.logical_or(
                    max_pos[:, np.newaxis, 0] <= min_pos[np.newaxis, :, 0],
                    min_pos[:, np.newaxis, 0] >= max_pos[np.newaxis, :, 0]),
            )
            candidates = np.logical_not(candidates)

            candidates = np.triu(candidates, k=1)
            distances = np.ones((len(t_list), len(t_list)))
            for i, j in zip(*candidates.nonzero()):
                if t_list_dilated[i].intersects(t_list_dilated[j]):
                    penalty = self.get_pair_penalty(
                        b_list[i], b_list[j], h_list[i], h_list[j], layout_separator_map, ds)
                    distances[i, j] = penalty
                    distances[j, i] = penalty

            adjacency = (distances < self.paragraph_line_threshold).astype(np.int)
            adjacency = adjacency * (1 - np.eye(adjacency.shape[0]))  # put zeros on diagonal
            graph = csr_matrix(adjacency > 0)
            _, clusters_array = connected_components(
                csgraph=graph, directed=False, return_labels=True)

            return clusters_array

        else:
            return [0]

def nonmaxima_suppression(input, element_size=(7, 1)):
    """Vertical non-maxima suppression.
    :param input: input array
    :param element_size: structure element for greyscale dilations
    """
    if len(input.shape) == 3:
        dilated = np.zeros_like(input)
        for i in range(input.shape[0]):
            dilated[i, :, :] = ndimage.morphology.grey_dilation(
                input[i, :, :], size=element_size)
    else:
        dilated = ndimage.morphology.grey_dilation(input, size=element_size)

    return input * (input == dilated)
