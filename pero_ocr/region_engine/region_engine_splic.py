import tensorflow as tf
import numpy as np
import cv2
import hdbscan

from scipy import sparse, misc, ndimage, interpolate
from scipy.ndimage.morphology import distance_transform_edt
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.sparse.linalg import eigsh, lobpcg

from skimage.measure import block_reduce, label
from sklearn.utils.validation import check_array
import shapely.geometry

from pyamg import smoothed_aggregation_solver

from pero_ocr.document_ocr import layout
from pero_ocr.line_engine import line_postprocessing as pp
from pero_ocr.region_engine import spectral_clustering as sc

class EngineRegionSPLIC(object):

    def __init__(self, model_path, downsample=4, use_cpu=False,
                 reduce_factor=8, n_components=24, pad=52):

        self.downsample = downsample # downsample factor before CNN inference
        self.reduce_factor = reduce_factor # another downsample factor before spectral clustering (target shouldn't be much bigger than 256 x 256)
        self.n_components = n_components # neumber of eigenvectors for clustering
        self.pad = pad # CNN training pad

        saver = tf.train.import_meta_graph(model_path + '.meta')
        if use_cpu:
            tf_config = tf.ConfigProto(device_count={'GPU': 0})
        else:
            tf_config = tf.ConfigProto(device_count={'GPU': 1})
            tf_config.gpu_options.allow_growth = True
        self.session = tf.Session(config=tf_config)
        saver.restore(self.session, model_path)

    def detect(self, image):

        polygons_list = []
        region_baselines_list = []
        region_heights_list = []
        region_textlines_list = []

        out_map = self.get_maps(image)
        baselines_list, heights_list, embeddings_list = self.parse_maps(out_map)
        textlines_list = [pp.baseline_to_textline(baseline, heights) for baseline, heights in zip(baselines_list, heights_list)]
        clusters_array = self.cluster_lines(embeddings_list)
        # check noise lines for adjacancy to previous region in reading order due to eigenvector-based clustering errors
        clusters_array = self.postprocess_noisy_lines(clusters_array, baselines_list, heights_list, out_map[:,:,3])

        for i in range(np.amax(clusters_array)+1):
            region_baselines = []
            region_heights = []
            region_textlines = []
            for baseline, heights, textline, cluster in zip(baselines_list, heights_list, textlines_list, clusters_array):
                if cluster == i:
                    region_baselines.append(baseline)
                    region_heights.append(heights)
                    region_textlines.append(textline)

            region_baselines, region_heights, region_textlines = pp.order_lines_vertical(region_baselines, region_heights, region_textlines)
            region_polygon = shapely.geometry.Polygon(np.concatenate(region_textlines, axis=0))

            polygons_list.append(region_polygon.convex_hull.exterior.coords)
            region_baselines_list.append(region_baselines)
            region_heights_list.append(region_heights)
            region_textlines_list.append(region_textlines)

        return polygons_list, region_baselines_list, region_heights_list, region_textlines_list

    def get_maps(self, img):

        img = cv2.resize(img, (0,0), fx=1/self.downsample, fy=1/self.downsample, interpolation=cv2.INTER_AREA)
        img = np.pad(img, [(self.pad, self.pad), (self.pad, self.pad), (0,0)], 'constant')

        new_shape_x = int(np.ceil(img.shape[0] / 64) * 64)
        new_shape_y = int(np.ceil(img.shape[1] / 64) * 64)
        test_img_canvas = np.zeros((1, new_shape_x, new_shape_y, 3))
        test_img_canvas[0, :img.shape[0], :img.shape[1], :] = img

        out_map = self.session.run('test_probs:0', feed_dict={'test_dataset:0': test_img_canvas[:,:,:]/256.})
        out_map = out_map[0, self.pad:img.shape[0]-self.pad, self.pad:img.shape[1]-self.pad, :]

        return out_map

    def parse_maps(self, out_map):
        """Parse input baseline, height and region map into list of baselines coords, heights and embeddings
        :param baseline_map: array of baseline and endpoint probabilities
        :param heights_map: array of estimated heights
        """
        baselines_list = []
        heights_list = []
        embeddings_list = []

        baselines_map = pp.nonmaxima_suppression(out_map[:,:,2] - 3 * out_map[:,:,3]) > 0.2
        heights_map = out_map[:,:,:2]
        embedding_map = self.edges_to_embeddings(out_map[:,:,3],
                                            n_components=self.n_components,
                                            reduce_factor=self.reduce_factor)

        baselines_img, num_detections = ndimage.measurements.label(baselines_map, structure=np.ones((3, 3)))
        inds = np.where(baselines_img > 0)
        labels = baselines_img[inds[0], inds[1]]

        for i in range(1, num_detections+1):
            baseline_inds, = np.where(labels == i)
            if len(baseline_inds) > 5:
                pos_all = np.stack([inds[1][baseline_inds], inds[0][baseline_inds]], axis=1)  # go from matrix indexing to image indexing

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

                heights_pred = heights_map[inds[0][baseline_inds], inds[1][baseline_inds], :]
                if np.all(np.amax(heights_pred, axis=0) > 0):  # percentile will fail on zero vector, discard the baseline in such case
                    heights_pred = np.maximum(heights_pred, 0)
                    heights_pred = np.asarray([
                        np.percentile(heights_pred[:, 0], 70),
                        np.percentile(heights_pred[:, 1], 70)
                    ])

                    embeddings = embedding_map[inds[0][baseline_inds]-int(heights_pred[0]/2), np.clip(np.amin(inds[1][baseline_inds]+20), 0, embedding_map.shape[1]-1), :]
                    embeddings = np.average(embeddings, axis=0)

                    baselines_list.append(self.downsample * pos)
                    heights_list.append([self.downsample * heights_pred[0],
                                         self.downsample * heights_pred[1]])
                    embeddings_list.append(embeddings)

        return baselines_list, heights_list, embeddings_list

    def edges_to_embeddings(self, edge_map, n_components=16, reduce_factor=4):
        edge_map_reduced  = block_reduce(edge_map, (reduce_factor,reduce_factor), func=np.amax).astype(np.float32)
        adjacency = sc.img_to_graph(edge_map_reduced)
        eigenvectors = sc.spectral_embedding(adjacency, n_components=n_components)

        eigenvectors = np.reshape(eigenvectors, (edge_map_reduced.shape[0],edge_map_reduced.shape[1], n_components))

        return cv2.resize(eigenvectors, (edge_map.shape[1], edge_map.shape[0]))

    def cluster_lines(self, embeddings_list):
        if len(embeddings_list)>1:
            line_clusterer = hdbscan.HDBSCAN(min_cluster_size=4, min_samples=1, prediction_data=True)
            line_clusterer = line_clusterer.fit(embeddings_list)
            # clusters_probs = hdbscan.all_points_membership_vectors(line_clusterer)
            clusters_array = line_clusterer.labels_

            noise_lines = np.where(clusters_array<0)[0]
            for noise_line in noise_lines:
                clusters_array[noise_line] = np.amax(clusters_array) + 1

            return clusters_array
        else:
            return [0]

    def postprocess_noisy_lines(self, clusters_array, baselines_list, heights_list, edge_map, threshold=0.5):
        for line_num, (cluster, baseline, heights) in enumerate(zip(clusters_array, baselines_list, heights_list)):
            if cluster == -1:

                y1 = int(np.clip(((baseline[0][1]/self.downsample+baseline[-1][1]/self.downsample)/2)-1.5*heights[0]/self.downsample, 0, edge_map.shape[0]))
                y2 = int(np.clip(((baseline[0][1]/self.downsample+baseline[-1][1]/self.downsample)/2)-0.5*heights[0]/self.downsample, 0, edge_map.shape[0]))
                x1 = int(np.clip(baseline[0][0]/self.downsample, 0, edge_map.shape[1]))
                x2 = int(np.clip(baseline[-1][0]/self.downsample, 0, edge_map.shape[1]))
                if y1 >= y2 or x1 >= x2:
                    continue
                upper_severance = np.sum(np.amax(edge_map[y1:y2, x1:x2], axis=0)) / (x2-x1+0.0001)

                if upper_severance < threshold:
                    upper_index = -1
                    smallest_vert_diff = np.inf
                    for neighbor_num, upper_baseline in enumerate(baselines_list):
                        if baseline[0][0] < upper_baseline[-1][0] and baseline[-1][0] > upper_baseline[0][0]:
                            vert_diff = (baseline[0][1] + baseline[-1][1]) / 2 - (upper_baseline[0][1] + upper_baseline[-1][1]) / 2
                            if vert_diff > 0 and vert_diff < smallest_vert_diff and vert_diff < 4 * heights[0]:
                                smallest_vert_diff = vert_diff
                                upper_index = neighbor_num

                    if upper_index >= 0:
                        clusters_array[line_num] = clusters_array[upper_index]

        return clusters_array
