import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import cv2
import tensorflow as tf
from scipy import ndimage
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from skimage.measure import block_reduce
from sklearn.metrics import pairwise_distances
import shapely.geometry as sg
from shapely.ops import cascaded_union, polygonize

from pero_ocr.line_engine import line_postprocessing as pp
from pero_ocr.region_engine import spectral_clustering as sc

class EngineRegionSimple(object):

    def __init__(self, model_path, downsample=4, use_cpu=False, pad=52,
                 max_mp=5, gpu_fraction=None):

        self.downsample = downsample # downsample factor before CNN inference
        self.pad = pad # CNN training pad
        self.max_megapixels = max_mp if max_mp is not None else 5 # maximum megapixels when processing image to avoid OOM
        self.gpu_fraction = gpu_fraction

        if model_path is not None:
            saver = tf.train.import_meta_graph(model_path + '.meta')
            if use_cpu:
                tf_config = tf.ConfigProto(device_count={'GPU': 0})
            else:
                tf_config = tf.ConfigProto(device_count={'GPU': 1})
                if self.gpu_fraction is None:
                    tf_config.gpu_options.allow_growth = True
                else:
                    tf_config.gpu_options.per_process_gpu_memory_fraction = self.gpu_fraction
            self.session = tf.Session(config=tf_config)
            saver.restore(self.session, model_path)

    def detect(self, image):

        # check that big images are rescaled before first CNN run
        downsample = self.downsample
        if (image.shape[0]/downsample) * (image.shape[1]/downsample) > self.max_megapixels * 10e5:
            downsample = np.sqrt((image.shape[0] * image.shape[1]) / (self.max_megapixels * 10e5))
        out_map = self.get_maps(image, downsample)
        # adapt second CNN run so that text height is between 10 and 14 downscaled pixels
        med_height = self.get_med_height(out_map)
        if med_height > 14 or med_height < 10:
            downsample = max(
                    np.sqrt((image.shape[0] * image.shape[1]) / (self.max_megapixels * 10e5)),
                    downsample * (med_height / 12)
                    )
            out_map = self.get_maps(image, downsample)

        baselines_list, heights_list, layout_separator_map = self.parse_maps(out_map, downsample)
        if not baselines_list:
            return [], [], [], []

        textlines_list = [pp.baseline_to_textline(baseline, heights) for baseline, heights in zip(baselines_list, heights_list)]

        clusters_array = self.cluster_lines(textlines_list, layout_separator_map, downsample)

        regions_textlines_tmp = []
        polygons_tmp = []
        for i in range(np.amax(clusters_array)+1):
            region_baselines = []
            region_heights = []
            region_textlines = []
            for baseline, heights, textline, cluster in zip(baselines_list, heights_list, textlines_list, clusters_array):
                if cluster == i:
                    region_baselines.append(baseline)
                    region_heights.append(heights)
                    region_textlines.append(textline)

            region_poly_points = np.concatenate(region_textlines, axis=0)

            max_poly_line = np.amax(np.array([np.amax(np.diff(baseline[:,0])) for baseline in region_baselines]))
            max_height = np.amax(np.array(region_heights))
            max_alpha =  1.5 * np.maximum(max_poly_line, max_height)
            region_poly = alpha_shape(region_poly_points, max_alpha)

            regions_textlines_tmp.append(region_textlines)
            polygons_tmp.append(region_poly)

        #remove overlaps
        polygons_tmp = self.filter_polygons(polygons_tmp, regions_textlines_tmp)

        #up to this point, polygons can be any geometry that comes from alpha_shape
        polygons_list = []
        for region_poly in polygons_tmp:
            if region_poly.geom_type == 'MultiPolygon':
                for poly in region_poly:
                    polygons_list.append(poly.simplify(5))
            if region_poly.geom_type == 'Polygon':
                polygons_list.append(region_poly.simplify(5))

        baselines_list, heights_list, textlines_list = pp.order_lines_vertical(baselines_list, heights_list, textlines_list)
        polygons_list = [np.array(poly.exterior.coords) for poly in polygons_list]

        return polygons_list, baselines_list, heights_list, textlines_list

    def get_maps(self, img, downsample):

        img = cv2.resize(img, (0,0), fx=1/downsample, fy=1/downsample, interpolation=cv2.INTER_AREA)
        img = np.pad(img, [(self.pad, self.pad), (self.pad, self.pad), (0,0)], 'constant')

        new_shape_x = int(np.ceil(img.shape[0] / 64) * 64)
        new_shape_y = int(np.ceil(img.shape[1] / 64) * 64)
        test_img_canvas = np.zeros((1, new_shape_x, new_shape_y, 3))
        test_img_canvas[0, :img.shape[0], :img.shape[1], :] = img

        out_map = self.session.run('test_probs:0', feed_dict={'test_dataset:0': test_img_canvas[:,:,:]/256.})
        out_map = out_map[0, self.pad:img.shape[0]-self.pad, self.pad:img.shape[1]-self.pad, :]

        return out_map

    def get_med_height(self, out_map):
        heights = (out_map[:,:,2] > 0.2).astype(np.float) * out_map[:,:,0]
        med_height = np.median(heights[heights>0])

        return med_height


    def parse_maps(self, out_map, downsample):
        """Parse input baseline, height and region map into list of baselines coords, heights and embd
        :param baseline_map: array of baseline and endpoint probabilities
        :param heights_map: array of estimated heights
        """
        baselines_list = []
        heights_list = []
        l_embd_list = []
        m_embd_list = []
        r_embd_list = []

        out_map[:,:,4][out_map[:,:,4]<0] = 0
        baselines_map = ndimage.convolve(out_map[:,:,2], np.ones((3,3)))
        baselines_map = pp.nonmaxima_suppression(baselines_map, element_size=(7,1))
        baselines_map /= 9 # revert signal amplification from convolution
        baselines_map = (baselines_map - out_map[:,:,3]) > 0.2
        heights_map = ndimage.morphology.grey_dilation(out_map[:,:,:2], size=(7,1,1))

        baselines_img, num_detections = ndimage.measurements.label(baselines_map, structure=np.ones((3, 3)))
        inds = np.where(baselines_img > 0)
        labels = baselines_img[inds[0], inds[1]]

        for i in range(1, num_detections+1):
            bl_inds, = np.where(labels == i)
            if len(bl_inds) > 5:
                pos_all = np.stack([inds[1][bl_inds], inds[0][bl_inds]], axis=1) # go from matrix indexing to image indexing

                _, indices = np.unique(pos_all[:, 0], return_index=True)
                pos = pos_all[indices]
                x_index = np.argsort(pos[:, 0])
                pos = pos[x_index]

                target_point_count = min(10, pos.shape[0] // 10)
                target_point_count = max(target_point_count, 2)
                selected_pos = np.linspace(0, (pos.shape[0]) - 1, target_point_count).astype(np.int32)

                pos = pos[selected_pos, :]
                pos[0,0] -= 2 # region edge detection bites out of baseline pixels, stretch to compensate
                pos[-1,0] += 2

                heights_pred = heights_map[inds[0][bl_inds], inds[1][bl_inds], :]

                heights_pred = np.maximum(heights_pred, 0)
                heights_pred = np.asarray([
                    np.percentile(heights_pred[:, 0], 70),
                    np.percentile(heights_pred[:, 1], 70)
                ])

                baselines_list.append(downsample * pos.astype(np.float))
                heights_list.append([downsample * heights_pred[0],
                                     downsample * heights_pred[1]])

        return baselines_list, heights_list, out_map[:,:,4]

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
                tot_height = np.abs(textline[0,1] - textline[-1,1])
                textlines_dilated.append(textline_poly.buffer(tot_height))

            distances = np.ones((len(textlines), len(textlines)))
            for i in range(len(textlines)):
                for j in range(i+1, len(textlines)):
                    if textlines_dilated[i].intersects(textlines_dilated[j]):
                        penalty = self.get_penalty(textlines[i]/downsample, textlines[j]/downsample, layout_separator_map)
                        distances[i, j] = penalty
                        distances[j, i] = penalty

            adjacency = distances < threshold
            graph = csr_matrix(adjacency)
            _, clusters_array = connected_components(csgraph=graph, directed=False, return_labels=True)

            return clusters_array

        else:
            return [0]

    def filter_polygons(self, polygons, region_textlines):
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
                        score_i += sg.Polygon(line).intersection(poly_intersection).area
                    score_j = 0
                    for line in region_textlines[j]:
                        score_j += sg.Polygon(line).intersection(poly_intersection).area
                    if score_i > score_j:
                        polygons[i] = polygons[i].union(poly_intersection)
                    else:
                        polygons[j] = polygons[j].union(poly_intersection)
        return [polygon for i, polygon in enumerate(polygons) if i not in inds_to_remove]

def alpha_shape(points, alpha):
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return sg.MultiPoint(list(points)).convex_hull

    # coords = np.array([point.coords[0] for point in points])
    tri = Delaunay(points)
    triangles = points[tri.vertices]
    a = ((triangles[:,0,0] - triangles[:,1,0]) ** 2 + (triangles[:,0,1] - triangles[:,1,1]) ** 2) ** 0.5
    b = ((triangles[:,1,0] - triangles[:,2,0]) ** 2 + (triangles[:,1,1] - triangles[:,2,1]) ** 2) ** 0.5
    c = ((triangles[:,2,0] - triangles[:,0,0]) ** 2 + (triangles[:,2,1] - triangles[:,0,1]) ** 2) ** 0.5
    s = ( a + b + c ) / 2.0
    areas = (s*(s-a)*(s-b)*(s-c)) ** 0.5
    circums = a * b * c / (4.0 * (areas + 0.0001))
    filtered = triangles[circums < alpha]
    edge1 = filtered[:,(0,1)]
    edge2 = filtered[:,(1,2)]
    edge3 = filtered[:,(2,0)]
    edge_points = np.unique(np.concatenate((edge1,edge2,edge3)), axis = 0).tolist()
    m = sg.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles)
