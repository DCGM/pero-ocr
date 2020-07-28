import math
import random
import warnings

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
                avg_hpos_1 = np.average(np.asarray(baselines[i])[:, 1]).astype(np.int32)
                avg_hpos_2 = np.average(np.asarray(baselines[j])[:, 1]).astype(np.int32)
                min_i = np.amin(np.asarray(baselines[i])[:, 0]).astype(np.int32)
                max_i = np.amax(np.asarray(baselines[i])[:, 0]).astype(np.int32)
                min_j = np.amin(np.asarray(baselines[j])[:, 0]).astype(np.int32)
                max_j = np.amax(np.asarray(baselines[j])[:, 0]).astype(np.int32)
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
            new_line_inds = np.argsort(np.asarray(new_line)[:, 0])
            baselines.append([new_line[x] for x in new_line_inds.tolist()])
            heights.append(new_height.tolist())

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
    baseline_features = np.asarray([[baseline[0][0], (baseline[0][1]+baseline[-1][1])/2] for baseline in baselines])
    alpha = 4  # how much to reduce vertical difference penalty
    feature_normalizers = np.asarray([[baseline[-1][0] - baseline[0][0], alpha*height[0]] for baseline, height in zip(baselines, heights)])
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
        v_min = np.amin(np.asarray([baselines[bs_ind][0][1] for bs_ind in bs_inds]))
        v_max = np.amax(np.asarray([baselines[bs_ind][0][1] for bs_ind in bs_inds]))
        h_min = np.median(np.asarray([baselines[bs_ind][0][0] for bs_ind in bs_inds]))
        h_max = np.median(np.asarray([baselines[bs_ind][-1][0] for bs_ind in bs_inds]))
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
    baselines_order = [baseline[0][1]+random.uniform(0.001, 0.999) for baseline in baselines]  # adding random number to order to prevent swapping when two lines are on same y-coord
    baselines = [baseline for _, baseline in sorted(zip(baselines_order, baselines))]
    heights = [height for _, height in sorted(zip(baselines_order, heights))]
    textlines = [textline for _, textline in sorted(zip(baselines_order, textlines))]

    return baselines, heights, textlines


def stretch_baselines(baselines, stretch):
    baselines_stretched = []
    for baseline in baselines:
        last_point = baseline[-1:, :].copy()
        last_point[0,0] += stretch
        first_point = baseline[:1, :].copy()
        first_point[0,0] -= stretch
        baselines_stretched.append(np.concatenate((first_point, baseline, last_point), axis=0))
    return baselines_stretched


def stretch_baselines_to_region(baselines, region):
    baselines_stretched = []
    region = np.concatenate((region, region[:1, :]), axis=0)
    for baseline in baselines:
        # print(shapely.geometry.LineString(baseline).intersects(shapely.geometry.Polygon(region)))
        line_interpf = np.poly1d(np.polyfit(baseline[:, 0], baseline[:, 1], 1))
        y_1 = line_interpf(np.amin(region[:, 0]))
        y_2 = line_interpf(np.amax(region[:, 0]))
        baseline_ls = shapely.geometry.LineString([(np.amin(region[:, 0]), y_1), (np.amax(region[:, 0]), y_2)])
        region_ls = shapely.geometry.LineString(region)

        intersections_ls = region_ls.intersection(baseline_ls)
        #intersection can be empty due to borderline baselines and integer coordinate rotations
        if isinstance(intersections_ls, shapely.geometry.MultiPoint):
            intersections = np.squeeze(np.asarray([intersection.coords.xy for intersection in intersections_ls]))
            intersection_left = intersections[np.argmin(intersections[:, 0]), :]
            intersection_right = intersections[np.argmax(intersections[:, 0]), :]

            baselines_stretched.append(np.concatenate((intersection_left[np.newaxis, :], baseline, intersection_right[np.newaxis, :]), axis=0))
    return baselines_stretched


def resample_baselines(baselines, num_points=10):
    baselines_resampled = []

    for baseline in baselines:
        vertical = np.abs(baseline[0,0]-baseline[-1,0]) < np.abs(baseline[0,1]-baseline[-1,1])
        if vertical:
            baseline = np.stack((baseline[:,-1], baseline[:,0]), axis=1)
        if baseline.shape[0] == 2:
            line_interpf = np.poly1d(np.polyfit(baseline[:,0], baseline[:,1], 1))
        else:
            line_interpf = np.poly1d(np.polyfit(baseline[:,0], baseline[:,1], 2))
        new_xs = np.linspace(baseline[0,0], baseline[-1,0], num_points)
        new_ys = line_interpf(new_xs)
        baseline_resampled = np.stack((new_xs, new_ys), axis=-1)
        if vertical:
            baseline_resampled = np.stack((baseline_resampled[:,-1], baseline_resampled[:,0]), axis=1)
        baselines_resampled.append(baseline_resampled)
    return baselines_resampled


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
    region_shpl = shapely.geometry.Polygon(region)
    baseline_shpl = shapely.geometry.LineString(baseline)

    if not baseline_shpl.intersects(region_shpl):
        return None, None

    textline_shpl = shapely.geometry.Polygon(textline)
    if not textline_shpl.is_valid: # this can happen after merging two lines
        textline_shpl = textline_shpl.convex_hull
    if not region_shpl.is_valid:
        warnings.warn("Input region contains self-intersections, replacing it with convex hull...")
        region_shpl = region_shpl.convex_hull
    baseline_is = region_shpl.intersection(baseline_shpl)
    textline_is = region_shpl.intersection(textline_shpl)
    if isinstance(baseline_is, shapely.geometry.LineString) and isinstance(textline_is, shapely.geometry.Polygon) and baseline_is.length>2:
        return np.asarray(baseline_is.coords), np.asarray(textline_is.exterior.coords)
    else:
        return None, None


def baseline_to_textline(baseline, heights):
    """Convert baseline coords and its respective heights to a textline polygon.
    :param baseline: baseline coords
    :param heights: textline heights
    """

    heights = np.array([max(1, heights[0]), max(1, heights[1])]).astype(np.float32)

    x_diffs = np.diff(baseline[:,0])
    x_diffs = np.concatenate((x_diffs, x_diffs[-1:]), axis=0)
    y_diffs = np.diff(baseline[:,1])
    y_diffs = np.concatenate((y_diffs, y_diffs[-1:]), axis=0)

    alfas = np.pi/2 + np.arctan2(y_diffs, x_diffs)
    y_up_diffs = np.sin(alfas) * heights[0]
    x_up_diffs = np.cos(alfas) * heights[0]
    y_down_diffs = np.sin(alfas) * heights[1]
    x_down_diffs = np.cos(alfas) * heights[1]

    pos_up = baseline.copy().astype(np.float32)
    pos_up[:, 1] -= y_up_diffs
    pos_up[:, 0] -= x_up_diffs
    pos_down = baseline.copy().astype(np.float32)
    pos_down[:, 1] += y_down_diffs
    pos_down[:, 0] += x_down_diffs
    pos_t = np.concatenate([pos_up, pos_down[::-1, :]], axis=0)

    return pos_t


def get_rotation(lines):
    """Get mean baseline tilt as angle.
    :param baselines: list of baselines
    """
    lines_info = list()

    for line in lines:
        first_line_point = line[0].astype(np.float64)
        last_line_point = line[-1].astype(np.float64)

        if last_line_point[1] != first_line_point[1]:
            rotation = math.degrees(
                math.atan((last_line_point[1] - first_line_point[1]) / (last_line_point[0] - first_line_point[0])))
            length = math.sqrt(
                math.pow(last_line_point[0] - first_line_point[0], 2)
                + math.pow(last_line_point[1] - first_line_point[1], 2))
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
    change_coords = [[item[0], item[1]] for item in coords]
    coords = np.array([change_coords])
    rotated_coords = cv2.transform(coords, M)[0]
    out_coords = [[item[0], item[1]] for item in rotated_coords]

    return np.asarray(out_coords)

def adjust_baselines_to_intensity(baselines, img, tolerance=5):
    grad_img = np.gradient(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float))[0]
    grad_img = ndimage.gaussian_filter(grad_img, 3)
    new_baselines = []
    for baseline in baselines:
        num_points = baseline[-1][0] - baseline[0][0]
        baseline_pts = np.round(resample_baselines([baseline], num_points=num_points)[0]).astype(np.int)
        best_score = -np.inf
        for offset in range(-tolerance, tolerance):
            score = np.sum(grad_img[
                np.clip(baseline_pts[:,1]+offset, 0, grad_img.shape[0]-1),
                np.clip(baseline_pts[:,0], 0, grad_img.shape[1]-1)])
            if score > best_score:
                best_score = score
                best_offset = offset
        baseline_pts[:,1] += best_offset
        new_baselines.append(resample_baselines([baseline_pts], num_points=len(baseline))[0])
    return new_baselines
