import math
import random
import warnings

import numpy as np
import cv2
from scipy import ndimage
from scipy.spatial import Delaunay, distance
from sklearn import cluster
import shapely
import shapely.geometry as sg
from shapely.ops import cascaded_union, polygonize

from pero_ocr.document_ocr.layout import PageLayout, RegionLayout, TextLine


def check_line_position(baseline, page_size, margin=20, min_ratio=0.125):
    """Checks if line is short and very close to the page edge, which may indicate that the region actually belongs to
    a second, partially scanned page of the document.
    """
    x_coords = np.array(baseline)[:, 0]
    if np.any(x_coords < margin) and not np.any(x_coords > page_size[1] * min_ratio):
        return False
    elif np.any(x_coords > (page_size[1] - margin)) and not np.any(x_coords < page_size[1] * min_ratio):
        return False
    else:
        return True


def get_max_line_length(baseline_list):
    if not baseline_list:
        return 0
    x0s = np.array([b[0, 0] for b in baseline_list])
    x1s = np.array([b[-1, 0] for b in baseline_list])
    return np.abs(x1s - x0s).max()


def assign_lines_to_regions(baseline_list, heights_list, textline_list, regions):
    min_line = np.zeros([len(textline_list), 2], dtype=np.float32)
    max_line = np.zeros([len(textline_list), 2], dtype=np.float32)
    for textline, min_, max_ in zip(baseline_list, min_line, max_line):
        min_[:] = textline.min(axis=0)
        max_[:] = textline.max(axis=0)

    min_region = np.zeros([len(regions), 2], dtype=np.float32)
    max_region = np.zeros([len(regions), 2], dtype=np.float32)
    for region, min_, max_ in zip(regions, min_region, max_region):
        min_[:] = region.polygon.min(axis=0)
        max_[:] = region.polygon.max(axis=0)

    candidates = np.logical_and(
            np.logical_or(
                max_line[:, np.newaxis, 1] <= min_region[np.newaxis, :, 1],
                min_line[:, np.newaxis, 1] >= max_region[np.newaxis, :, 1]),
            np.logical_or(
                max_line[:, np.newaxis, 0] <= min_region[np.newaxis, :, 0],
                min_line[:, np.newaxis, 0] >= max_region[np.newaxis, :, 0]),
    )
    candidates = np.logical_not(candidates)
    for line_id, region_id in zip(*candidates.nonzero()):
        baseline = baseline_list[line_id]
        heights = heights_list[line_id]
        textline = textline_list[line_id]
        region = regions[region_id]
        baseline_intersection, textline_intersection = mask_textline_by_region(
            baseline, textline, region.polygon)
        if baseline_intersection is not None and textline_intersection is not None:
            new_textline = TextLine(
                id='{}-l{:03d}'.format(region.id, line_id+1),
                baseline=baseline_intersection,
                polygon=textline_intersection,
                heights=heights
                )
            region.lines.append(new_textline)

    return regions


def retrace_region(region):
    """ Discards existing region coords and makes new ones from alpha shape
    around text lines.
    """
    region_textlines = [line.polygon for line in region.lines]
    new_polygon = region_from_textlines(region_textlines)

    if new_polygon.geom_type == 'MultiPolygon':
        new_polygon = new_polygon.convex_hull.simplify(5)
    elif new_polygon.geom_type == 'Polygon':
        new_polygon = new_polygon.simplify(5)
    else:
        print('WARNING: polygon coordinates discarded during retrace.')

    region.polygon = np.array(new_polygon.exterior.coords)

    return


def baseline_to_textline(baseline, heights):
    """Convert baseline coords and its respective heights to a textline polygon.
    :param baseline: baseline coords
    :param heights: textline heights
    """

    heights = np.array(
        [max(1, heights[0]), max(1, heights[1])]).astype(np.float32)

    x_diffs = np.diff(baseline[:, 0])
    x_diffs = np.concatenate((x_diffs, x_diffs[-1:]), axis=0)
    y_diffs = np.diff(baseline[:, 1])
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


def region_from_textlines(region_textlines):
    '''
    Convert textline list to Shapely polygon using alpha shape
    '''
    max_spacings = []
    for textline in region_textlines:
        pts_1 = textline[1:]
        pts_2 = textline[:-1]
        spacings = np.linalg.norm(
            np.asarray(pts_1) - np.asarray(pts_2), axis=1)
        max_spacings.append(spacings.max())
    max_spacing = np.asarray(max_spacings).max()
    region_poly_points = np.concatenate(region_textlines, axis=0)

    region_poly = alpha_shape(region_poly_points, max_spacing)

    for textline in region_textlines:
        textline_poly = check_polygon(sg.Polygon(textline))
        if not region_poly.contains(textline_poly):
            region_poly = region_poly.union(textline_poly)

    return region_poly


def get_circumradius(a, b, c):
    '''
    Compute circumradius of a triangle
    '''
    s = (a + b + c) / 2.0
    areas = (s*(s-a)*(s-b)*(s-c)) ** 0.5
    circums = a * b * c / (4.0 * (areas + 0.0001))
    return circums


def alpha_shape(points, alpha):
    '''
    Get shapely polygon around a point cloud using alpha shape algorithm
    '''
    if len(points) < 4:
        return sg.MultiPoint(list(points)).convex_hull

    tri = Delaunay(points)
    triangles = points[tri.vertices]
    a = ((triangles[:, 0, 0] - triangles[:, 1, 0]) ** 2 + (triangles[:, 0, 1] - triangles[:, 1, 1]) ** 2) ** 0.5
    b = ((triangles[:, 1, 0] - triangles[:, 2, 0]) ** 2 + (triangles[:, 1, 1] - triangles[:, 2, 1]) ** 2) ** 0.5
    c = ((triangles[:, 2, 0] - triangles[:, 0, 0]) ** 2 + (triangles[:, 2, 1] - triangles[:, 0, 1]) ** 2) ** 0.5
    circums = get_circumradius(a, b, c)
    filtered = triangles[circums <= alpha]
    edge1 = filtered[:, (0, 1)]
    edge2 = filtered[:, (1, 2)]
    edge3 = filtered[:, (2, 0)]
    edge_points = np.unique(
        np.concatenate((edge1, edge2, edge3)), axis=0).tolist()
    m = sg.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles)


def check_polygon(polygon):
    '''
    Check that polygon does not contain any self-intersections. If it does,
    return the respective convex hull.
    '''
    if not polygon.is_valid:
        polygon = polygon.convex_hull
    return polygon


def merge_lines(baselines, heights):
    """Merge lines on similar vertical offsets. Useful as postprocessing with
    known regions.
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
                v_gap = np.maximum(min_i - max_j, min_j - max_i)
                h_overlay = np.minimum(avg_hpos_1 + heights[i][1], avg_hpos_2 + heights[j][1]) - np.maximum(avg_hpos_1 - heights[i][0], avg_hpos_2 - heights[j][0])
                if (h_overlay > (0.7 * np.minimum(heights[i][0] + heights[i][1], heights[j][0] + heights[j][1]))
                        and not v_overlay
                        and v_gap < 2 * np.minimum(heights[i][0] + heights[i][1], heights[j][0] + heights[j][1])):
                    # print(v_gap)
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
            baselines.append(resample_baselines([np.asarray([new_line[x] for x in new_line_inds.tolist()])])[0])
            heights.append(new_height.tolist())

    baselines = filter_list(baselines, merged_lines)
    heights = filter_list(heights, merged_lines)

    baselines = [np.asarray(baseline) for baseline in baselines]

    baselines_order = [baseline[0][1] + random.uniform(0.001, 0.999) for baseline in
                       baselines]  # adding random number to order to prevent swapping when two lines are on same y-coord
    baselines = [baseline for _, baseline in sorted(zip(baselines_order, baselines))]
    heights = [height for _, height in sorted(zip(baselines_order, heights))]

    baselines = [rotate_coords(baseline, -rotation, (0, 0)) for baseline in baselines]

    return baselines, heights


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
    region_shpl = sg.Polygon(region)
    baseline_shpl = sg.LineString(baseline)

    try:
        if not baseline_shpl.intersects(region_shpl):
            return None, None
    except shapely.errors.TopologicalError:
        return None, None

    textline_shpl = sg.Polygon(textline)
    if not textline_shpl.is_valid: # this can happen after merging two lines
        print('Invalid textline encountered, replacing it with convex hull...')
        textline_shpl = textline_shpl.convex_hull
    if not region_shpl.is_valid:
        warnings.warn("Input region contains self-intersections, replacing it with convex hull...")
        region_shpl = region_shpl.convex_hull
    baseline_is = region_shpl.intersection(baseline_shpl)
    textline_is = region_shpl.intersection(textline_shpl)

    if isinstance(textline_is, sg.MultiPolygon): # this can happen generally with some combinations of layout and line detection
        areas = np.array([poly.area for poly in textline_is])
        textline_is = textline_is[np.argmax(areas)]
    if isinstance(baseline_is, sg.MultiLineString):  # this can happen generally with some combinations of layout and line detection
        lengths = np.array([line.length for line in baseline_is])
        baseline_is = baseline_is[np.argmax(lengths)]

    if isinstance(baseline_is, sg.LineString) and isinstance(textline_is, sg.Polygon) and baseline_is.length>2:
        return np.asarray(baseline_is.coords), np.asarray(textline_is.exterior.coords)
    else:
        return None, None


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
                np.arctan2((last_line_point[1] - first_line_point[1]), (last_line_point[0] - first_line_point[0])))
            length = math.sqrt(
                math.pow(last_line_point[0] - first_line_point[0], 2)
                + math.pow(last_line_point[1] - first_line_point[1], 2))
            lines_info.append((length, rotation))
        else:
            lines_info.append((0,0))

    lines_info = sorted(lines_info, key=lambda x: x[0], reverse=True)
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
    coords = coords.copy()
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
