import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import cv2
from scipy import ndimage
from scipy.spatial import Delaunay, distance
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from skimage.measure import block_reduce
from sklearn.metrics import pairwise_distances
import shapely.geometry as sg
from shapely.ops import cascaded_union, polygonize

from pero_ocr.layout_engines import layout_helpers as helpers
from pero_ocr.layout_engines import parsenet


class LayoutEngine(object):
    def __init__(self, model_path, downsample=4, use_cpu=False, pad=52,
                 max_mp=5, gpu_fraction=None, detection_threshold=0.2):
        self.parsenet = parsenet.ParseNet(
            model_path,
            downsample=downsample,
            use_cpu=use_cpu,
            pad=pad,
            max_mp=max_mp,
            gpu_fraction=gpu_fraction,
            detection_threshold=detection_threshold)
        self.line_detection_threshold = detection_threshold

    def get_maps(self, image):
        return self.parsenet.get_maps_with_optimal_resolution(image), self.parsenet.tmp_downsample

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

    def detect(self, image):
        """Uses parsenet to find lines and region separators, clusters vertically
        close lines by computing penalties and postprocesses the resulting
        regions.
        """
        maps, ds = self.get_maps(image)
        b_list, h_list, layout_separator_map = self.parse(
            maps, ds)
        if not b_list:
            return [], [], [], []
        t_list = [
            helpers.baseline_to_textline(b, h) for b, h in zip(b_list, h_list)]

        # cluster the lines into regions
        clusters_array = self.cluster_lines(t_list, layout_separator_map, ds)
        regions_textlines_tmp = []
        polygons_tmp = []
        for i in range(np.amax(clusters_array)+1):
            region_baselines = []
            region_heights = []
            region_textlines = []
            for baseline, heights, textline, cluster in zip(b_list, h_list, t_list, clusters_array):
                if cluster == i:
                    region_baselines.append(baseline)
                    region_heights.append(heights)
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
            if region_poly.geom_type == 'MultiPolygon':
                for poly in region_poly:
                    p_list.append(poly.simplify(5))
            if region_poly.geom_type == 'Polygon':
                p_list.append(region_poly.simplify(5))

        b_list, h_list, t_list = helpers.order_lines_vertical(
            b_list, h_list, t_list)
        p_list = [np.array(poly.exterior.coords) for poly in p_list]

        return p_list, b_list, h_list, t_list

    def parse(self, out_map, downsample):
        """Parse input baseline, height and region map into list of baselines
        coords, list of heights and region map
        :param out_map: array of baseline and endpoint probabilities with
        channels: ascender height, descender height, baselines, baseline
        endpoints, region boundaries
        """
        baselines_list = []
        heights_list = []
        structure = np.asarray(
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ])

        out_map[:, :, 4][out_map[:, :, 4] < 0] = 0
        baselines_map = ndimage.convolve(out_map[:, :, 2], np.ones((3, 3))/9)
        baselines_map = nonmaxima_suppression(
            baselines_map, element_size=(7, 1))
        baselines_map = (baselines_map - out_map[:, :, 3]) > self.line_detection_threshold
        heights_map = ndimage.morphology.grey_dilation(
            out_map[:, :, :2], size=(7, 1, 1))

        baselines_map_dilated = ndimage.morphology.binary_dilation(
            baselines_map, structure=structure)
        baselines_img, num_detections = ndimage.measurements.label(
            baselines_map_dilated, structure=np.ones([3, 3]))
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
                    np.percentile(heights_pred[:, 0], 70),
                    np.percentile(heights_pred[:, 1], 70)
                ])

                baselines_list.append(downsample * pos.astype(np.float))
                heights_list.append([downsample * heights_pred[0],
                                     downsample * heights_pred[1]])

        return baselines_list, heights_list, out_map[:, :, 4]

    def filter_polygons(self, polygons, region_textlines):
        polygons = [helpers.check_polygon(polygon) for polygon in polygons]
        # TBD first, replace any invalid polygons with convex hulls
        inds_to_remove = []
        for i in range(len(polygons)):
            for j in range(i+1, len(polygons)):
                if polygons[i].contains(polygons[j]):
                    inds_to_remove.append(j)
                elif polygons[j].contains(polygons[i]):
                    inds_to_remove.append(i)
                elif polygons[i].intersects(polygons[j]):
                    # first check if a polygon is completely inside another, remove the smaller in that case
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

    def get_penalty(self, textline1, textline2, map):
        x_overlap = max(0, min(np.amax(textline1[:,0]), np.amax(textline2[:,0])) - max(np.amin(textline1[:,0]), np.amin(textline2[:,0])))
        smaller_len = min(np.amax(textline1[:,0])-np.amin(textline1[:,0]), np.amax(textline2[:,0])-np.amin(textline2[:,0]))
        if x_overlap > smaller_len / 4:
            x_1 = int(max(np.amin(textline1[:,0]), np.amin(textline2[:,0])))
            x_2 = int(min(np.amax(textline1[:,0]), np.amax(textline2[:,0])))
            if np.average(textline1[:,1]) > np.average(textline2[:,1]):
                y_pos_1 = np.average(textline1[:textline1.shape[0]//2,1]).astype(np.int)
                penalty_1 = np.sum(map[
                    np.clip(y_pos_1-3, 0, map.shape[0]):np.clip(y_pos_1+3, 0, map.shape[0]),
                    np.clip(x_1, 0, map.shape[1]):np.clip(x_2, 0, map.shape[1])
                    ])
                penalty_1 /= x_overlap

                y_pos_2 = np.average(textline2[textline1.shape[0]//2:,1]).astype(np.int)
                penalty_2 = np.sum(map[
                    np.clip(y_pos_2-3, 0, map.shape[0]):np.clip(y_pos_2+3, 0, map.shape[0]),
                    np.clip(x_1, 0, map.shape[1]):np.clip(x_2, 0, map.shape[1])
                    ])
                penalty_2 /= x_overlap
            else:
                y_pos_1 = np.average(textline1[textline1.shape[0]//2:,1]).astype(np.int)
                penalty_1 = np.sum(map[
                    np.clip(y_pos_1-3, 0, map.shape[0]):np.clip(y_pos_1+3, 0, map.shape[0]),
                    np.clip(x_1, 0, map.shape[1]):np.clip(x_2, 0, map.shape[1])
                    ])
                penalty_1 /= x_overlap

                y_pos_2 = np.average(textline2[:textline1.shape[0]//2,1]).astype(np.int)
                penalty_2 = np.sum(map[
                    np.clip(y_pos_2-3, 0, map.shape[0]):np.clip(y_pos_2+3, 0, map.shape[0]),
                    np.clip(x_1, 0, map.shape[1]):np.clip(x_2, 0, map.shape[1])
                    ])
                penalty_2 /= x_overlap
            penalty = np.abs(max(penalty_1, penalty_2))
        else:
            penalty = 999
        return penalty

    def cluster_lines(self, textlines, layout_separator_map, downsample, threshold=0.3):
        if len(textlines) > 1:

            textlines_dilated = []
            for textline in textlines:
                textline_poly = sg.Polygon(textline)
                tot_height = np.abs(textline[0, 1] - textline[-1, 1])
                textlines_dilated.append(textline_poly.buffer(tot_height))

            distances = np.ones((len(textlines), len(textlines)))
            for i in range(len(textlines)):
                for j in range(i+1, len(textlines)):
                    if textlines_dilated[i].intersects(textlines_dilated[j]):
                        penalty = self.get_penalty(
                            textlines[i]/downsample, textlines[j]/downsample, layout_separator_map)
                        distances[i, j] = penalty
                        distances[j, i] = penalty

            adjacency = distances < threshold
            graph = csr_matrix(adjacency)
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
