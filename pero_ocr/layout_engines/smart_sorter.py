#!/usr/bin/env python3

import cv2
import math
import numpy as np

from configparser import SectionProxy
from copy import deepcopy
from itertools import tee
from shapely import geometry, affinity
from typing import List, Dict, Union, Optional

from pero_ocr.document_ocr.layout import PageLayout, RegionLayout


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class Region:
    def __init__(self, region: Union[RegionLayout, np.ndarray]):
        if isinstance(region, RegionLayout):
            self.id = region.id
            self.x_arr, self.y_arr = region.polygon.transpose((1, 0))
        elif isinstance(region, np.ndarray):
            assert len(region[0]) == len(region[1]), "Not equal number of coord pairs"
            self.id = "TEST"
            self.x_arr, self.y_arr = region
        else:
            raise Exception("Wrong Region parameter type.")

        self.x_min = self.x_arr.min()
        self.x_max = self.x_arr.max()
        self.y_min = self.y_arr.min()
        self.y_max = self.y_arr.max()

    def get_middle_coords(self) -> Dict[str, List]:
        return {self.id: [(self.x_arr.max() + self.x_arr.min()) // 2, (self.y_arr.min() + self.y_arr.max()) // 2]}

    def intersect(self, regions: Union["Region", "CoupledRegions"], vertical: bool, intersect_param: float = 0.1) -> bool:
        """
        Return True if two regions intersect each other (sides are enough) horizontally or vertically
        """

        if vertical and self.x_min <= regions.x_max and regions.x_min <= self.x_max:
            intersection = np.min(np.abs((self.x_min - regions.x_max, regions.x_min - self.x_max)))

            if intersection / (self.x_max - self.x_min) > intersect_param and intersection / (regions.x_max - regions.x_min) > intersect_param:
                return True

        elif not vertical and self.y_min <= regions.y_max and regions.y_min <= self.y_max:
            intersection = np.min(np.abs((self.y_min - regions.y_max, regions.y_min - self.y_max)))

            if intersection / (self.y_max - self.y_min) > intersect_param and intersection / (regions.y_max - regions.y_min) > intersect_param:
                return True

        return False

    def get_corners(self):
        return self.x_min, self.y_min, self.x_max, self.y_max

    def pretty_print(self, indent):
        print(" " * indent + self.id)

    def __eq__(self, other: "Region"):
        return self.id == other.id


class CoupledRegions:
    """
    Class for holding intersecting regions.
    """
    def __init__(self, regions=List[Union["CoupledRegions", Region]], parent: Optional["CoupledRegions"] = None, intersect_param=0.1):
        if isinstance(regions, PageLayout):
            self.region_list = []

            for region in regions.regions:
                self.region_list.append(Region(region))

        elif isinstance(regions, list):
            assert len(regions) > 0, "Given empty region list!"
            self.region_list: List[Union[CoupledRegions, Region]] = regions

        self.intersect_param = intersect_param
        self.parent: CoupledRegions = parent
        self.x_min, self.x_max, self.y_min, self.y_max = 1e5, 0, 1e5, 0

        # get min and max of all polygons
        for reg in self.region_list:
            l, t, r, b = reg.get_corners()
            self.update_corners(l, t, r, b)

    def __eq__(self, other: Union["CoupledRegions", Region]):
        # might be compared with Region object, but single Region object is never in CoupledRegions object
        if not isinstance(other, CoupledRegions):
            return False

        if len(self.region_list) != len(other.region_list):
            return False

        # TODO: think about better method of comparing CoupledRegions for equality
        for region in self.region_list:
            if region not in other.region_list:
                return False

        return True

    def update_corners(self, l, t, r, b):
        if l < self.x_min:
            self.x_min = l
        if t < self.y_min:
            self.y_min = t
        if r > self.x_max:
            self.x_max = r
        if b > self.y_max:
            self.y_max = b

    def get_corners(self):
        return self.x_min, self.y_min, self.x_max, self.y_max

    def add_regions(self, regions: Union["CoupledRegions", Region]):
        if isinstance(regions, Region):
            self.region_list.append(regions)

            l, t, r, b = regions.get_corners()
            self.update_corners(l, t, r, b)

        elif isinstance(regions, CoupledRegions):
            self.region_list.extend(regions.region_list)

            l, t, r, b = regions.get_corners()
            self.update_corners(l, t, r, b)

    def get_middle_coords(self) -> Dict[str, List]:
        coords = {}

        for reg in self.region_list:
            coords.update(reg.get_middle_coords())

        return coords

    def pretty_print(self, indent=0):
        for region in self.region_list:
            if isinstance(region, CoupledRegions):
                print(" " * indent + "Coupled")
                region.pretty_print(indent + 4)
            elif isinstance(region, Region):
                region.pretty_print(indent)
                # print(" " * indent + "Coupled")

    def intersect(self, regions: Union[Region, "CoupledRegions"], vertical: bool, intersect_param: float = 0.1):
        """
        Return True if two polygons intersect each other (sides are enough) horizontally or vertically
        :param regions: list of Region or CoupledRegions
        :param vertical: True if we are coupling regions vertically (below or above each other)
        """

        if vertical and self.x_min <= regions.x_max and regions.x_min <= self.x_max:
            intersection = np.min(np.abs((self.x_min - regions.x_max, regions.x_min - self.x_max)))

            if intersection / (self.x_max - self.x_min) > intersect_param and intersection / (
                    regions.x_max - regions.x_min) > intersect_param:
                return True

        elif not vertical and self.y_min <= regions.y_max and regions.y_min <= self.y_max:
            intersection = np.min(np.abs((self.y_min - regions.y_max, regions.y_min - self.y_max)))

            if intersection / (self.y_max - self.y_min) > intersect_param and intersection / (
                    regions.y_max - regions.y_min) > intersect_param:
                return True

        return False

    def divide_and_order(self, vertical: bool = False):
        if len(self.region_list) == 1:
            return

        aligned = []
        non_aligned = deepcopy(self.region_list)

        # try to divide objects
        while len(non_aligned):
            coupled = non_aligned.pop(0) \
                if isinstance(non_aligned[0], CoupledRegions) \
                else CoupledRegions([non_aligned.pop(0)], self, self.intersect_param)

            changed = True

            while changed:
                changed = False

                # add intersection regions to coupled
                for idx, region in enumerate(non_aligned):
                    if coupled.intersect(region, vertical):
                        non_aligned.pop(idx)
                        coupled.add_regions(region)

                        changed = True
                        break

            # store CoupledRegions object
            aligned.append(coupled)

        self.region_list = aligned

        # if parent already tried to decouple the same region and we both failed -> plan B
        if len(aligned) == 1 and self.parent is not None and self in self.parent.region_list:
            self.decouple()

        for idx, coupled in enumerate(self.region_list):
            if len(coupled.region_list) > 1:
                self.region_list[idx].divide_and_order(not vertical)
            # else:
            #     self.region_list[idx] = coupled.region_list[0]

        if vertical:
            self.region_list = sorted(self.region_list, key=lambda reg: reg.x_min)
        else:
            self.region_list = sorted(self.region_list, key=lambda reg: reg.y_min)

    def decouple(self):
        """
        Decouple intersecting regions in self.region_list and store them there
        Fallback method when regions are intersecting horizontally and vertically
        Possibly could create hierarchy of CoupledRegions with Region objects as leaves
        """
        # TODO: sorting by minimum or maximum? => both and select the one with biggest difference
        regions = self.region_list[0].region_list

        x_sort = sorted(regions, key=lambda x: x.x_min)
        x_diffs = 0

        for l, r in pairwise(x_sort):
            x_diffs += np.abs(l.x_min - r.x_min)

        y_sort = sorted(regions, key=lambda x: x.y_min)
        y_diffs = 0

        for u, d in pairwise(y_sort):
            y_diffs += np.abs(u.y_min - d.y_min)

        # sort coupled components by axis with larger differences between min points
        # key = lambda x: x.x_min if x_diffs > y_diffs else lambda x: x.y_min
        if x_diffs > y_diffs:
            key = lambda r: r.x_min
        else:
            key = lambda r: r.y_min

        aligned = sorted(regions, key=key)

        # sort by x axis and compute local diffs, sort by y axis and compute local diffs
        # sort by axis with larger sum of local diffs
        self.region_list = [CoupledRegions([region], self, self.intersect_param) for region in aligned]

    def get_ordered_ids(self) -> List:
        """
        :return: list of IDs of regions in order
        """
        ids = []

        for regions in self.region_list:
            if isinstance(regions, Region):
                ids.append(regions.id)

            elif isinstance(regions, CoupledRegions):
                ids.extend(regions.get_ordered_ids())

        return ids


class SmartRegionSorter:
    def __init__(self, config: SectionProxy, config_path=""):
        # if intersection of two regions is less than given parameter w.r.t. both regions, intersection doesn't count
        self.intersect_param = config.getfloat('FakeIntersectionParameter', fallback=0.1)

    def process_page(self, image, page_layout: PageLayout):
        regions = []

        if len(page_layout.regions) < 2:
            return page_layout

        rotation = SmartRegionSorter.get_rotation(max(*page_layout.regions, key=lambda reg: len(reg.lines)).lines)
        page_layout = SmartRegionSorter.rotate_page_layout(page_layout, -rotation)

        for region in page_layout.regions:
            regions.append(Region(region))

        regions = CoupledRegions(regions, intersect_param=self.intersect_param)
        regions.divide_and_order()

        # get ordered region IDs
        ordered_ids = regions.get_ordered_ids()

        # substitute every region with
        region_idxs = [next((idx for idx, region in enumerate(page_layout.regions) if region.id == region_id)) for region_id in ordered_ids]

        page_layout.regions = [page_layout.regions[idx] for idx in region_idxs]
        page_layout = SmartRegionSorter.rotate_page_layout(page_layout, rotation)

        return page_layout

    @staticmethod
    def rotate_page_layout(page: PageLayout, angle, origin=(0, 0)):
        if angle == 0:
            return page

        rot_matrix = cv2.getRotationMatrix2D(origin, angle, 1)

        for reg_idx, region in enumerate(page.regions):
            region.polygon = SmartRegionSorter.rotate_polygon(region.polygon, angle)
            # region.polygon = SmartRegionSorter.rotate_coords(region.polygon, rot_matrix)

            for line_idx, line in enumerate(region.lines):
                line.polygon = SmartRegionSorter.rotate_polygon(line.polygon, angle)
                # line.polygon = SmartRegionSorter.rotate_coords(line.polygon, rot_matrix)
                line.baseline = SmartRegionSorter.rotate_line(line.baseline, angle)
                # line.baseline = SmartRegionSorter.rotate_coords(line.baseline, rot_matrix)

        return page

    @staticmethod
    def rotate_coords(coords, rot_matrix):
        """Rotate coords around given center point
        :param coords: points to rotate
        :param rot_matrix: rotation matrix
        """
        change_coords = [[item[0], item[1]] for item in coords]
        coords = np.array([change_coords])
        rotated_coords = cv2.transform(coords, rot_matrix)[0]
        out_coords = [[item[0], item[1]] for item in rotated_coords]

        return np.asarray(out_coords)

    @staticmethod
    def rotate_polygon(polygon, angle):
        line_poly = geometry.Polygon(polygon)
        line_poly = affinity.rotate(line_poly, angle, origin=(0, 0))
        return np.stack(line_poly.exterior.coords.xy, axis=1)

    @staticmethod
    def rotate_line(baseline, angle):
        baseline_obj = geometry.LineString(baseline)
        baseline_obj = affinity.rotate(baseline_obj, angle, origin=(0, 0))
        return np.array(baseline_obj)

    @staticmethod
    def get_rotation(lines):
        """Get mean baseline tilt as angle.
        :param baselines: list of baselines
        """
        lines_info = []

        if len(lines) == 0:
            return 0

        for line in lines:
            first_line_point = line.baseline[0].astype(np.float64)
            last_line_point = line.baseline[-1].astype(np.float64)

            if last_line_point[1] != first_line_point[1]:
                # rotation = math.degrees(
                #     math.atan((last_line_point[1] - first_line_point[1]) / (last_line_point[0] - first_line_point[0])))
                length = math.sqrt(
                    math.pow(last_line_point[0] - first_line_point[0], 2)
                    + math.pow(last_line_point[1] - first_line_point[1], 2))
                rotation = math.degrees(math.sin((last_line_point[1] - first_line_point[1]) / length))
                lines_info.append((length, rotation))
            else:
                lines_info.append((0, 0))

        lines_info = sorted(lines_info, key = lambda x: x[0], reverse = True)
        lines_info = lines_info[0: int(len(lines_info) / 2)]
        rotation_sum = sum(item[1] for item in lines_info)
        rotation = 0

        if len(lines_info) > 0:
            rotation = rotation_sum/len(lines_info)

        return rotation


def test():
    region1 = Region(np.array([[20, 100, 100, 20], [20, 20, 120, 120]]))
    region2 = Region(np.array([[120, 220, 220, 120], [20, 20, 120, 120]]))

    region3 = Region(np.array([[50, 200, 200, 100, 100, 50], [50, 50, 100, 100, 200, 200]]))
    region4 = Region(np.array([[220, 300, 300, 120, 120, 220], [50, 50, 200, 200, 120, 120]]))

    coupled = CoupledRegions([region1])

    print(f"Should intersect: {coupled.intersect(region2, False)}")
    print(f"Should not intersect: {coupled.intersect(region2, True)}")

    coupled2 = CoupledRegions([region3, region4])
    coupled2.divide_and_order(False)
    coupled2.pretty_print()

    # regions = CoupledRegions(PageLayout(file=f"./page/0022_53fe7036-121f-11e8-8b6e-005056b73ae5.xml"))


if __name__ == '__main__':
    test()
