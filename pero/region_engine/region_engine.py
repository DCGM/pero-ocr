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
        adjacency = img_to_graph(edge_img)
        eigenvectors = spectral_embedding(adjacency, n_components=self.n_components)

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

def deterministic_vector_sign_flip(u):
    max_abs_rows = np.argmax(np.abs(u), axis=1)
    signs = np.sign(u[range(u.shape[0]), max_abs_rows])
    u *= signs[:, np.newaxis]
    return u

def set_diag(laplacian, value, norm_laplacian):
    n_nodes = laplacian.shape[0]

    laplacian = laplacian.tocoo()
    if norm_laplacian:
        diag_idx = (laplacian.row == laplacian.col)
        laplacian.data[diag_idx] = value

    n_diags = np.unique(laplacian.row - laplacian.col).size
    if n_diags <= 7:
        laplacian = laplacian.todia()
    else:
        laplacian = laplacian.tocsr()
    return laplacian

def spectral_embedding(adjacency, n_components=8,
                       random_state=np.random.RandomState(), eigen_tol=0.0,
                       norm_laplacian=True, drop_first=True):

    n_nodes = adjacency.shape[0]
    # Whether to drop the first eigenvector
    if drop_first:
        n_components = n_components + 1

    print('solving eigenvectors...')
    laplacian, dd = csgraph_laplacian(adjacency, normed=norm_laplacian,
                                      return_diag=True)

    laplacian = check_array(laplacian, dtype=np.float64,
                                accept_sparse=True)
    laplacian = set_diag(laplacian, 1, norm_laplacian)

    diag_shift = 1e-5 * sparse.eye(laplacian.shape[0])
    laplacian += diag_shift
    ml = smoothed_aggregation_solver(check_array(laplacian, 'csr'))
    laplacian -= diag_shift

    M = ml.aspreconditioner()
    X = random_state.rand(laplacian.shape[0], n_components + 1)
    X[:, 0] = dd.ravel()
    for attempt_num in range(1, 4):
        try:
            _, diffusion_map = lobpcg(laplacian, X, M=M, tol=1.e-5,
                                largest=False)
            continue
        except:
            print('LOBPCG eigensolver failed, attempting to recondition on different eigenvector approximation (attempt {}/3)'.format(attempt_num))
            X = random_state.rand(laplacian.shape[0], n_components + 1)
            X[:, 0] = dd.ravel()

    embedding = diffusion_map.T
    if norm_laplacian:
        embedding = embedding / dd
    if embedding.shape[0] == 1:
        raise ValueError
    # laplacian = _set_diag(laplacian, 1, norm_laplacian)
    # laplacian *= -1
    # v0 = random_state.uniform(-1, 1, laplacian.shape[0])
    # lambdas, diffusion_map = eigsh(laplacian, k=n_components,
    #                                sigma=1.0, which='LM',
    #                                tol=eigen_tol, v0=v0)
    # embedding = diffusion_map.T[n_components::-1]
    # if norm_laplacian:
    #     embedding = embedding / dd

    embedding = deterministic_vector_sign_flip(embedding)
    if drop_first:
        return embedding[1:n_components].T
    else:
        return embedding[:n_components].T

def make_edges_3d(n_x, n_y, n_z=1):
    vertices = np.arange(n_x * n_y * n_z).reshape((n_x, n_y, n_z))
    edges_deep = np.vstack((vertices[:, :, :-1].ravel(),
                            vertices[:, :, 1:].ravel()))
    edges_right = np.vstack((vertices[:, :-1].ravel(),
                             vertices[:, 1:].ravel()))
    edges_down = np.vstack((vertices[:-1].ravel(),
                            vertices[1:].ravel()))
    edges = np.hstack((edges_deep, edges_right, edges_down))
    return edges

def compute_weights_3d(edges, img):
    _, n_y, n_z = img.shape

    # img = img*0.8+0.1

    weights = np.stack((img[edges[0] // (n_y * n_z),
                      (edges[0] % (n_y * n_z)) // n_z,
                      (edges[0] % (n_y * n_z)) % n_z],
                      img[edges[1] // (n_y * n_z),
                      (edges[1] % (n_y * n_z)) // n_z,
                      (edges[1] % (n_y * n_z)) % n_z]), axis=1)
    # weights = np.exp(-4*(((weights[:,0]==1) & (weights[:,1]==2)) | ((weights[:,0]==2) & (weights[:,1]==1))).astype(np.float64))
    # weights = np.exp(-4*np.amax(weights, axis=1))
    weights = np.exp(-4*weights[:,0])
    # print(weights)
    # print(weights)
    return weights

# XXX: Why mask the image after computing the weights?

def mask_edges_weights(mask, edges, weights=None):
    """Apply a mask to edges (weighted or not)"""
    inds = np.arange(mask.size)
    inds = inds[mask.ravel()]
    ind_mask = np.logical_and(np.in1d(edges[0], inds),
                              np.in1d(edges[1], inds))
    edges = edges[:, ind_mask]
    if weights is not None:
        weights = weights[ind_mask]
    if len(edges.ravel()):
        maxval = edges.max()
    else:
        maxval = 0
    order = np.searchsorted(np.unique(edges.ravel()), np.arange(maxval + 1))
    edges = order[edges]
    if weights is None:
        return edges
    else:
        return edges, weights

def to_graph(n_x, n_y, n_z, mask=None, img=None,
              return_as=sparse.coo_matrix, dtype=None):
    """Auxiliary function for img_to_graph and grid_to_graph
    """
    edges = make_edges_3d(n_x, n_y, n_z)

    if dtype is None:
        if img is None:
            dtype = np.int
        else:
            dtype = img.dtype

    if img is not None:
        img = np.atleast_3d(img)
        weights = compute_weights_3d(edges, img)
        if mask is not None:
            edges, weights = mask_edges_weights(mask, edges, weights)
            diag = img.squeeze()[mask]
        else:
            diag = img.ravel()
        n_voxels = diag.size
    else:
        if mask is not None:
            mask = mask.astype(dtype=np.bool, copy=False)
            mask = np.asarray(mask, dtype=np.bool)
            edges = mask_edges_weights(mask, edges)
            n_voxels = np.sum(mask)
        else:
            n_voxels = n_x * n_y * n_z
        weights = np.ones(edges.shape[1], dtype=dtype)
        diag = np.ones(n_voxels, dtype=dtype)

    diag_idx = np.arange(n_voxels)
    i_idx = np.hstack((edges[0], edges[1]))
    j_idx = np.hstack((edges[1], edges[0]))
    graph = sparse.coo_matrix((np.hstack((weights, weights, diag)),
                              (np.hstack((i_idx, diag_idx)),
                               np.hstack((j_idx, diag_idx)))),
                              (n_voxels, n_voxels),
                              dtype=dtype)
    if return_as is np.ndarray:
        return graph.toarray()
    return return_as(graph)


def img_to_graph(img, mask=None, return_as=sparse.coo_matrix, dtype=None):
    img = np.atleast_3d(img)
    n_x, n_y, n_z = img.shape
    return to_graph(n_x, n_y, n_z, mask, img, return_as, dtype)
