#!/usr/bin/env python3

import numpy as np

from configparser import SectionProxy
from sklearn.cluster import DBSCAN
from typing import List

from pero_ocr.document_ocr.layout import PageLayout, RegionLayout


class Region:
    def __init__(self, region_layout: RegionLayout):
        self.region_layout = region_layout
        self.x_arr, self.y_arr = region_layout.polygon.transpose((1, 0))

    def __eq__(self, other: "Region"):
        return self.id == other.id

    @property
    def id(self):
        return self.region_layout.id

    @property
    def x_min(self):
        return self.x_arr.min()

    @property
    def x_max(self):
        return self.x_arr.max()

    @property
    def y_min(self):
        return self.y_arr.min()

    @property
    def y_max(self):
        return self.y_arr.max()


class NaiveRegionSorter:
    def __init__(self, config: SectionProxy, config_path=""):
        # minimal distance between clusters = page_width / width_denom
        self.width_denom = config.getint('ImageWidthDenominator', fallback=10)

    def process_page(self, image, page_layout: PageLayout):
        regions = []

        for region in page_layout.regions:
            regions.append(Region(region))

        eps = image.shape[1] // self.width_denom
        order = NaiveRegionSorter.sort_regions(regions, eps)

        page_layout.regions = [page_layout.regions[idx] for idx in order]

        return page_layout

    @staticmethod
    def sort_regions(regions: List[Region], eps: float):
        """

        :param regions: list of Region objects
        :param eps: maximal distance between points in cluster
        :return: sorted indices to regions array
        """
        x_points = np.array([region.y_min for region in regions])
        y_points = [region.y_min for region in regions]

        labels = DBSCAN(eps=eps, min_samples=1).fit_predict(x_points.reshape((-1, 1)))

        # indices are pointing to one point from each cluster
        clusters, cluster_idxs = np.unique(labels, return_index=True)
        sorted_cluster_idxs = sorted(clusters, key=lambda x: x_points[cluster_idxs[x]])

        order = []

        for cluster_id in sorted_cluster_idxs:
            point_idxs = np.argwhere(labels == cluster_id).reshape(-1)
            sorted_idxs = sorted(point_idxs, key=lambda x: y_points[x])

            order.extend(sorted_idxs)

        return order
