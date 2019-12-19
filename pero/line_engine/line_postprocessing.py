import math
import random

import tensorflow as tf
import numpy as np
import cv2
import shapely.geometry
from scipy import ndimage
from sklearn import cluster


def merge_lines(baselines, heights):
    """Merge lines on similar vertical offsets. Useful as postprocessing with known regions.
    :param baselines: list of baselines to merge
    :param heights: list of respective textline heights
    """

    rotation = get_rotation(baselines)
    baselines = [rotate_coords(baseline, rotation, (0, 0)) for baseline in baselines]
    baselines = [baseline.tolist() for baseline in baselines]

    merged_lines = list()
    lines_to_merge = list()
    for i in range(len(baselines)):
        lines_to_merge_i = list()
        for j in range(len(baselines)):
            if i != j:
                avg_hpos_1 = np.average(np.asarray(baselines[i])[:, 0]).astype(np.int32)
                avg_hpos_2 = np.average(np.asarray(baselines[j])[:, 0]).astype(np.int32)
                min_i = np.amin(np.asarray(baselines[i])[:, 1]).astype(np.int32)
                max_i = np.amax(np.asarray(baselines[i])[:, 1]).astype(np.int32)
                min_j = np.amin(np.asarray(baselines[j])[:, 1]).astype(np.int32)
                max_j = np.amax(np.asarray(baselines[j])[:, 1]).astype(np.int32)
                v_overlay = (min_i > min_j and max_i < max_j) or (min_j > min_i and max_j < max_i)
                h_overlay = np.minimum(avg_hpos_1 + heights[i][1], avg_hpos_2 + heights[j][1]) - np.maximum(avg_hpos_1 - heights[i][0], avg_hpos_2 - heights[j][0])
                if h_overlay > (0.7 * np.minimum(heights[i][0] + heights[i][1], heights[j][0] + heights[j][1])) and not v_overlay:
                    if i not in merged_lines:
                        lines_to_merge_i.append(i)
                        merged_lines.append(i)
                    if j not in merged_lines:
                        lines_to_merge_i.append(j)
                        merged_lines.append(j)
        lines_to_merge.append(lines_to_merge_i)

    for line_group in lines_to_merge:
        if len(line_group) > 0:
            new_line = list()
            new_height = np.zeros(2)
            for l_num in line_group:
                new_line += baselines[l_num]
                if heights[l_num][0] > new_height[0]:
                    new_height[0] = heights[l_num][0]
                if heights[l_num][1] > new_height[1]:
                    new_height[1] = heights[l_num][1]
            new_line_inds = np.argsort(np.asarray(new_line)[:, 1])
            baselines.append([new_line[x] for x in new_line_inds.tolist()])
            heights.append(new_height.astype(np.int32).tolist())

    baselines = filter_list(baselines, merged_lines)
    heights = filter_list(heights, merged_lines)

    baselines = [np.asarray(baseline) for baseline in baselines]
    baselines = [rotate_coords(baseline, -rotation, (0, 0)) for baseline in baselines]
    return baselines, heights


def cluster_baselines(baselines, heights):
    """Cluster baselines according to their vertical and horrizontal offset.
    Resulting clusters should match text regions.
    :param baselines: list of baselines to cluster
    :param heights: list of respective textline heights
    """
    baseline_features = np.asarray([[baseline[0][1], (baseline[0][0]+baseline[-1][0])/2] for baseline in baselines])
    alpha = 4  # how much to reduce vertical difference penalty
    feature_normalizers = np.asarray([[baseline[-1][1] - baseline[0][1], alpha*height[0]] for baseline, height in zip(baselines, heights)])
    feature_normalizers = np.median(feature_normalizers, axis=0)
    baseline_labels = cluster.DBSCAN(eps=0.5, min_samples=1).fit(baseline_features/feature_normalizers).labels_

    return baseline_labels


def order_lines_general(baselines, heights, textlines):
    """Attempt to order lines according to their position, both inside regions
    and across regions. This should be the correct reading order.
    :param baselines: list of baselines to order
    :param heights: list of respective textline heights
    :param textlines: list of respective textline polygons
    """
    baseline_labels = cluster_baselines(baselines, heights)
    clusters = np.arange(np.amax(baseline_labels)+1).tolist()
    clusters_v_inds = list()
    clusters_h_inds = list()
    for c in clusters:
        bs_inds = np.where(baseline_labels == c)[0]
        v_min = np.amin(np.asarray([baselines[bs_ind][0][0] for bs_ind in bs_inds]))
        v_max = np.amax(np.asarray([baselines[bs_ind][0][0] for bs_ind in bs_inds]))
        h_min = np.median(np.asarray([baselines[bs_ind][0][1] for bs_ind in bs_inds]))
        h_max = np.median(np.asarray([baselines[bs_ind][-1][1] for bs_ind in bs_inds]))
        clusters_v_inds.append([v_min, v_max])
        clusters_h_inds.append([h_min, h_max])
    clusters_rank = np.zeros_like(clusters)
    for cluster_ind, (cluster_v_inds, cluster_h_inds) in enumerate(zip(clusters_v_inds, clusters_h_inds)):
        for cluster_v_inds_c, cluster_h_inds_c in zip(clusters_v_inds, clusters_h_inds):
            if cluster_h_inds[0] - cluster_h_inds_c[1] > -20:  # how much can regions overlap to be considered next to each other
                clusters_rank[cluster_ind] += 1
            overlay_threshold = np.minimum(cluster_h_inds_c[1]-cluster_h_inds_c[0], cluster_h_inds[1]-cluster_h_inds[0])/2
            overlay = np.minimum(cluster_h_inds_c[1], cluster_h_inds[1]) - np.maximum(cluster_h_inds_c[0], cluster_h_inds[0])
            if cluster_v_inds_c[0] < cluster_v_inds[0] and overlay > overlay_threshold:
                clusters_rank[cluster_ind] += 1
    K = len(baselines)
    baselines_order = np.arange(K).tolist()
    for b, (_, label) in enumerate(zip(baselines_order, baseline_labels)):
        baselines_order[b] = b + K * clusters_rank[label]
    baselines = [baseline for _, baseline in sorted(zip(baselines_order, baselines))]
    heights = [height for _, height in sorted(zip(baselines_order, heights))]
    textlines = [textlines for _, textlines in sorted(zip(baselines_order, textlines))]

    return baselines, heights, textlines


def order_lines_vertical(baselines, heights, textlines):
    """Order lines according to their vertical position.
    :param baselines: list of baselines to order
    :param heights: list of respective textline heights
    :param textlines: list of respective textline polygons
    """
    baselines_order = [baseline[0][0]+random.uniform(0.001, 0.999) for baseline in baselines]  # adding random number to order to prevent swapping when two lines are on same y-coord
    baselines = [baseline for _, baseline in sorted(zip(baselines_order, baselines))]
    heights = [height for _, height in sorted(zip(baselines_order, heights))]
    textlines = [textline for _, textline in sorted(zip(baselines_order, textlines))]

    return baselines, heights, textlines


def stretch_baselines(baselines, stretch):
    baselines_stretched = []
    for baseline in baselines:
        last_point = baseline[-1:, :].copy()
        last_point[0,1] += stretch
        first_point = baseline[:1, :].copy()
        first_point[0,1] -= stretch
        baselines_stretched.append(np.concatenate((first_point, baseline, last_point), axis=0))
    return baselines_stretched


def nonmaxima_suppression(input, element_size=(7,1)):
    """Vertical non-maxima suppression.
    :param input: input array
    :param element_size: structure element for greyscale dilations
    """
    if len(input.shape) == 3:
        dilated = np.zeros_like(input)
        for i in range(input.shape[0]):
            dilated[i,:,:] = ndimage.morphology.grey_dilation(input[i,:,:], size=element_size)
    else:
        dilated = ndimage.morphology.grey_dilation(input, size=element_size)

    return input * (input == dilated)


def filter_list(items_list, indices_to_remove):
    """Remove list items by their indices.
    :param items_list: target list
    :param indices_to_remove: indices of items to be removed from target list
    """
    items_to_remove = list()
    [items_to_remove.append(items_list[index_to_remove]) for index_to_remove in indices_to_remove]
    [items_list.remove(item_to_remove) for item_to_remove in items_to_remove]

    return items_list


def mask_textline_by_region(baseline, textline, region):
    baseline_shpl = shapely.geometry.LineString(baseline)
    textline_shpl = shapely.geometry.Polygon(textline)
    region_shpl = shapely.geometry.Polygon(region)
    if not textline_shpl.is_valid: # this can happen after merging two lines
        textline_shpl = textline_shpl.convex_hull
    baseline_is = region_shpl.intersection(baseline_shpl)
    textline_is = region_shpl.intersection(textline_shpl)

    if baseline_is.length > 1 and isinstance(baseline_is, shapely.geometry.LineString) and isinstance(textline_is, shapely.geometry.Polygon):
        return np.asarray(baseline_is.coords), np.asarray(textline_is.exterior.coords)
    else:
        return None, None


def baseline_to_textline(baseline, heights):
    """Convert lists of baselines coords and their respective heights to textline polygons.
    :param baselines: list of baselines
    :param heights: list of respective textline heights
    """

    pos_up = np.asarray(baseline.copy()).astype(int)
    pos_up[:,0] -= heights[0]
    pos_down = np.asarray(baseline.copy()).astype(int)
    pos_down[:,0] += heights[1]
    pos_t = np.concatenate([pos_up, pos_down[::-1,:]], axis=0)

    return np.clip(pos_t, 0, None)


def get_rotation(lines):
    """Get mean baseline tilt as angle.
    :param baselines: list of baselines
    """
    lines_info = list()

    for line in lines:
        first_line_point = line[0].astype(np.float64)
        last_line_point = line[-1].astype(np.float64)

        if last_line_point[0] != first_line_point[0]:
            rotation = math.degrees(
                math.atan((last_line_point[0] - first_line_point[0]) / (last_line_point[1] - first_line_point[1])))
            length = math.sqrt(
                math.pow(last_line_point[1] - first_line_point[1], 2)
                + math.pow(last_line_point[0] - first_line_point[0], 2))
            lines_info.append((length, rotation))
        else:
            lines_info.append((0,0))

    lines_info = sorted(lines_info, key = lambda x: x[0], reverse = True)
    lines_info = lines_info[0:int(len(lines_info)/2)]
    rotation_sum = sum(item[1] for item in lines_info)
    rotation = 0

    if len(lines_info) > 0:
        rotation = rotation_sum/len(lines_info)

    return rotation


def rotate_coords(coords, rotation, center):
    """Rotate coords around given center point
    :param coords: points to rotate
    :param rotation: rotation angle
    :param center: center of rotation
    """
    M = cv2.getRotationMatrix2D((center), rotation, 1)
    change_coords = [[item[1], item[0]] for item in coords]
    coords = np.array([change_coords])
    rotated_coords = cv2.transform(coords, M)[0]
    out_coords = [[item[1], item[0]] for item in rotated_coords]

    return np.asarray(out_coords)
