
import cv2
import numpy as np

from shapely.geometry import Polygon

from pero_ocr.document_ocr.layout import PageLayout, RegionLayout


class SimpleThresholdRegion:
    def __init__(self, config, config_path=''):
        pass

    def process_page(self, img: np.ndarray, page_layout: PageLayout):
        polygons = SimpleThresholdRegion._compute_layout(img)
        page_layout.regions = [RegionLayout(f'r-{idx}', polygon[:, ::-1]) for idx, polygon in enumerate(polygons)]
        return page_layout

    @staticmethod
    def _split_components(img_segment: np.ndarray, min_point_per_component=0):
        """
        Split all components so that each one is in separate array.

        :param img_segment: array with connected components having positive values and background as -1
                       (output of connectedComponents)
        :param min_point_per_component: if component is composed number of points smaller than given amount
                                        it is not returned
        :return: array of components (numpy array where nonzero pixel is part of component)
        """

        # get most common nonzero value
        vals, counts = np.unique(img_segment, return_counts=True)
        max_idxs = filter(lambda x: counts[x] if vals[x] != 0 and counts[x] >= min_point_per_component else 0,
                          range(len(vals)))

        # translate indices to values
        max_ocurr_vals = [vals[max_idx] for max_idx in max_idxs]

        components = []

        for max_val in max_ocurr_vals:
            component = img_segment.copy()
            component[component != max_val] = 0
            components.append(component)

        # return points
        return components

    @staticmethod
    def _compute_layout(img: np.ndarray,
                        downscale=4,
                        open_kernel_size=28,
                        poly_simplify_tolerance=20,
                        denoising_strength=20,
                        border_dist=45,
                        threshold_block_size=100,
                        threshold_mean_subtract=80,
                        precise_envelope: bool = True,
                        min_point_per_component=100):
        """
        :param img: input image (BGR or grayscale format)
        :param downscale: image downscale factor before segmentation
        :param open_kernel_size: kernel size of Open transform
        :param poly_simplify_tolerance: maximal distance for every point form its origin in polygon
                                        valid only if precise envelope is computed
        :param denoising_strength: denoising kernel size (less detail with bigger kernel)
        :param border_dist: bounding polygon distance from the text
        :param threshold_block_size: size of a pixel neighborhood used to calculate threshold value for a pixel
                                     threshold_block_size / scale must be odd number!
        :param threshold_mean_subtract: number subtracted from the mean
        :param precise_envelope: precise envelope around text regions is computed if set to True
                                 convex envelope is computed otherwise
        :param min_point_per_component: component is skipped if it's composed of less components than given value
        :return: list of polygons (polygon = list of [x, y] pairs)
        """

        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, None, fx=1 / downscale, fy=1 / downscale)
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        # padding
        border_vals = np.concatenate((img[0, :], img[-1, :], img[:, 0], img[:, -1]))
        median_val = max(np.median(border_vals), 100)

        h, w = img.shape
        img = cv2.copyMakeBorder(img, h // 10, h // 10, w // 10, w // 10, cv2.BORDER_CONSTANT, value=median_val)

        # denoising
        img = cv2.fastNlMeansDenoising(img, h=denoising_strength // downscale)

        # thresholding
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                    threshold_block_size // downscale, threshold_mean_subtract)
        img = 255 - img

        # open
        kernel = np.ones((open_kernel_size // downscale, open_kernel_size // downscale), np.uint8)
        opened = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        # distance transform
        dist = cv2.distanceTransform(255 - opened, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        mask = (dist < border_dist // downscale).astype(np.uint8)

        # segmentation
        _, labels = cv2.connectedComponents(mask, connectivity=8)
        components = SimpleThresholdRegion._split_components(labels, min_point_per_component // downscale)

        regions = []

        for component in components:
            # sort polygon points
            contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                continue

            points = np.array([[v[0][0], v[0][1]] for v in contours[0]])

            if precise_envelope:
                # simplify polygon
                poly = Polygon(points)
                poly = poly.simplify(poly_simplify_tolerance // downscale)

                xl, yl = poly.exterior.xy
                region = np.array(list(zip(yl, xl)), dtype=np.int32)
            else:
                # compute convex hull
                hull = cv2.convexHull(points)
                hull = [[v[0][0], v[0][1]] for v in hull]
                region = np.array([(y, x) for x, y in hull], dtype=np.int32)

            # subtract padding and scale coordinates to original picture
            region = (region - np.array([h // 10, w // 10])) * downscale
            regions.append(region)

        return regions


def main():
    pass


if __name__ == "__main__":
    main()
