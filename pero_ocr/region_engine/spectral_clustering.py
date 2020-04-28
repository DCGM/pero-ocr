import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import distance_transform_edt
from itertools import product
import numbers
from scipy import sparse
from numpy.lib.stride_tricks import as_strided
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.sparse.linalg import eigsh
from pyamg import smoothed_aggregation_solver
from sklearn.utils.validation import check_array
from scipy.sparse.linalg import lobpcg

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

    laplacian, dd = csgraph_laplacian(adjacency, normed=norm_laplacian,
                                      return_diag=True)

    laplacian = check_array(laplacian, dtype=np.float64,
                                accept_sparse=True)
    laplacian = set_diag(laplacian, 1, norm_laplacian)

    # AMG preconditioner
    diag_shift = 1e-5 * sparse.eye(laplacian.shape[0])
    laplacian += diag_shift
    ml = smoothed_aggregation_solver(check_array(laplacian, 'csr'))
    M = ml.aspreconditioner()
    laplacian -= diag_shift


    X = random_state.rand(laplacian.shape[0], n_components + 1)
    X[:, 0] = dd.ravel()

    solved = False
    for attempt_num in range(1, 4):
        try:
            _, diffusion_map = lobpcg(laplacian, X, tol=1.e-5,
                                largest=False)
            break
        except:
            print('LOBPCG eigensolver failed, attempting to recondition on different eigenvector approximation (attempt {}/3)'.format(attempt_num))
            X = random_state.rand(laplacian.shape[0], n_components + 1)
            X[:, 0] = dd.ravel()

    embedding = diffusion_map.T
    if norm_laplacian:
        embedding = embedding / dd
    if embedding.shape[0] == 1:
        raise ValueError

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
    weights = np.amin(np.exp(-weights), axis=1)

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

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.feature_extraction import image
    from sklearn.cluster import spectral_clustering

    l = 100
    x, y = np.indices((l, l))

    center1 = (28, 24)
    center2 = (40, 50)
    center3 = (67, 58)
    center4 = (24, 70)

    radius1, radius2, radius3, radius4 = 16, 14, 15, 14

    circle1 = (x - center1[0]) ** 2 + (y - center1[1]) ** 2 < radius1 ** 2
    circle2 = (x - center2[0]) ** 2 + (y - center2[1]) ** 2 < radius2 ** 2
    circle3 = (x - center3[0]) ** 2 + (y - center3[1]) ** 2 < radius3 ** 2
    circle4 = (x - center4[0]) ** 2 + (y - center4[1]) ** 2 < radius4 ** 2

    # #############################################################################
    # 4 circles
    img = circle1 + circle2 + circle3 + circle4

    # We use a mask that limits to the foreground: the problem that we are
    # interested in here is not separating the objects from the background,
    # but separating them one from the other.
    mask = img.astype(bool)

    img = img.astype(float)
    img += 1 + 0.2 * np.random.randn(*img.shape)

    # Convert the image into a graph with the value of the gradient on the
    # edges.
    graph = image.img_to_graph(img, mask=mask)

    # Take a decreasing function of the gradient: we take it weakly
    # dependent from the gradient the segmentation is close to a voronoi
    graph.data = np.exp(-graph.data / graph.data.std())

    # Force the solver to be arpack, since amg is numerically
    # unstable on this example
    labels = spectral_clustering(graph, n_clusters=4, eigen_solver='amg', assign_labels='discretize')
    label_im = np.full(mask.shape, -1.)
    label_im[mask] = labels

    plt.matshow(img)
    plt.matshow(label_im)

    # #############################################################################
    # 2 circles
    # img = circle1 + circle2
    # mask = img.astype(bool)
    # img = img.astype(float)
    #
    # img += 1 + 0.2 * np.random.randn(*img.shape)
    #
    # graph = image.img_to_graph(img, mask=mask)
    # graph.data = np.exp(-graph.data / graph.data.std())
    #
    # labels = spectral_clustering(graph, n_clusters=2, eigen_solver='arpack')
    # label_im = np.full(mask.shape, -1.)
    # label_im[mask] = labels
    #
    # plt.matshow(img)
    # plt.matshow(label_im)

    plt.show()
