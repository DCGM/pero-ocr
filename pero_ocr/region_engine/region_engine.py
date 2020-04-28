import tensorflow as tf
import numpy as np
import cv2

from scipy import sparse, misc, ndimage, interpolate
from scipy.ndimage.morphology import distance_transform_edt
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.sparse.linalg import eigsh, lobpcg

from skimage.measure import block_reduce, label
from skimage.segmentation import felzenszwalb
from skimage.morphology import skeletonize
from sklearn.utils.validation import check_array
import shapely.geometry

from pyamg import smoothed_aggregation_solver

from pero_ocr.region_engine import spectral_clustering as sc

class EngineRegionDetector(object):

    def __init__(self, model_path, downsample=4, use_cpu=False,
                 reduce_factor=4, n_components=16,
                 scale=400, sigma=0.02, min_size=400,
                 median_smooth=(5,5,1), smooth=5):

        self.downsample = downsample # downsample factor before CNN inference
        self.reduce_factor = reduce_factor # another downsample factor before spectral clustering (target shouldn't be much bigger than 256 x 256)
        self.n_components = n_components # neumber of eigenvectors for clustering
        self.median_smooth = median_smooth # median smoothing of eigenvector map before clustering
        self.scale = scale # felzenszwalb parameter: bigger means bigger clusters
        self.sigma = sigma # felzenszwalb parameter: gaussian filter pre-processing
        self.min_size = min_size # felzenszwalb parameter: minimum cluster size, works strangely, see skimage docs
        self.smooth = smooth # structure element of morphological posprocessing (bigger means more compact cluster shapes)
        self.simplification = 3 # error threshold for bounding polygon point removal for easier editing

        saver = tf.train.import_meta_graph(model_path + '.meta')
        if use_cpu:
            tf_config = tf.ConfigProto(device_count={'GPU': 0})
        else:
            tf_config = tf.ConfigProto(device_count={'GPU': 1})
            tf_config.gpu_options.allow_growth = True
        self.session = tf.Session(config=tf_config)
        saver.restore(self.session, model_path)

    def detect(self, image):
        out_map = self.get_maps(image)
        labels = self.cluster_image(out_map)

        page_layout = []
        for region in range(1,np.amax(labels)+1):
            labeled_map, num_labels = label(labels==region, return_num=True)
            for j in range(1, num_labels+1):
                region_mask = labeled_map==j
                if np.any(region_mask):
                    region_coods = np.where(ndimage.morphology.binary_dilation(region_mask) ^ region_mask)
                    region_poly = shapely.geometry.Polygon(zip(region_coods[0], region_coods[1]))
                    simplified_poly = region_poly.convex_hull.simplify(3).exterior.coords
                    page_layout.append(np.asarray(simplified_poly) * self.downsample)

        return page_layout

    def get_maps(self, img):

        img = cv2.resize(img, (0,0), fx=1/self.downsample, fy=1/self.downsample)

        new_shape_x = img.shape[0]
        new_shape_y = img.shape[1]
        while not new_shape_x % 64 == 0:
            new_shape_x += 1
        while not new_shape_y % 64 == 0:
            new_shape_y += 1
        test_img_canvas = np.zeros((1, new_shape_x, new_shape_y, 3))
        test_img_canvas[0, :img.shape[0], :img.shape[1], :] = img

        out_map = self.session.run('inderence:0', feed_dict={'inference_input:0': test_img_canvas[:,:,:]/256.})
        out_map = out_map[0, :img.shape[0], :img.shape[1], :]

        return out_map

    def cluster_image(self, img):
        edge_img = skeletonize(block_reduce(img[:,:,2], (self.reduce_factor, self.reduce_factor), func=np.amax)>0.1).astype(np.float64)
        adjacency = sc.img_to_graph(edge_img)
        eigenvectors = sc.spectral_embedding(adjacency, n_components=self.n_components)

        eigenvectors = eigenvectors/np.amax(eigenvectors)
        eigenvectors = np.reshape(eigenvectors, (edge_img.shape[0],edge_img.shape[1], self.n_components))
        eigenvectors = ndimage.median_filter(eigenvectors, size=self.median_smooth)

        print('cluestering..')
        labels = felzenszwalb(eigenvectors, scale=self.scale, sigma=self.sigma, min_size=self.min_size)
        labels = cv2.resize(labels, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        labels[np.argmax(img, axis=2)==0] = 0
        labels[np.argmax(img, axis=2)==1] += 1

        labels_post = np.zeros((labels.shape[0], labels.shape[1], np.amax(labels)+1))
        for i in range(1, np.amax(labels)+1):
            labels_post[:,:,i] = labels == i
        labels_post = ndimage.morphology.binary_erosion(labels_post, structure=np.ones((self.smooth, self.smooth, 1)))
        labels_post = ndimage.morphology.binary_dilation(labels_post, structure=np.ones((self.smooth, self.smooth, 1)))

        return np.argmax(labels_post, axis=2)
