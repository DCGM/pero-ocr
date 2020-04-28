import os
import time
import argparse
import random
import numpy as np
import copy

import cv2
import skimage.draw
from shapely.geometry import Polygon, Point, LineString, MultiPoint
from scipy import  ndimage

import pero_ocr.document_ocr as pero
import pero_ocr.document_ocr.layout as layout
import pero_ocr.line_engine.line_postprocessing as linepp
import pero_ocr.region_engine.region_engine_splic as region_engine

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image-dir', help='Path to input image folder', required=True)
    parser.add_argument('-p', '--page-dir', help='Path to input PAGE XML folder (optional)', required=False)
    parser.add_argument('-o', '--output-dir', help='Path to output PAGE XML folder (optional, overwrites input page files if left blank)', required=False)
    args = parser.parse_args()
    return args

def select_polygons(polygon_coords, points):
    selected = []
    if len(points) > 1:
        selection_geometry = LineString(points)
        for l_num, polygon in enumerate(polygon_coords):
            polygon_geometry = Polygon(polygon)
            if selection_geometry.intersects(polygon_geometry) and not l_num in selected:
                selected.append(l_num)
    for point in points:
        l_num = get_region_by_point(polygon_coords, point)
        if l_num is not None and not l_num in selected:
            selected.append(l_num)
    return selected

def check_intersection(line1, line2):
    line1 = LineString(line1)
    line2 = LineString(line2)
    return line1.intersects(line2)

def get_region_by_point(regions_coords, point):
    num_chosen = None
    point = Point(point)
    for r_num, region in enumerate(regions_coords):
        poly = Polygon(region)
        if poly.contains(point):
            num_chosen = r_num
            return r_num

    return None

# def sync_heights(heights, selected_lines):
#     heights_selection = list()
#     for h_num, height in enumerate(heights):
#         if h_num in selected_lines:
#             heights_selection.append(height)
#     heights_med = np.median(np.asarray(heights_selection), axis=0).astype(np.uint16).tolist()
#     for h_num, height in enumerate(heights):
#         if h_num in selected_lines: heights[h_num] = heights_med
#     return heights

def resample_layout(page_layout, ds):
    for line in page_layout.lines_iterator():
        line.baseline = line.baseline * ds
        line.heights = (np.asarray(line.heights) * ds).tolist()
        line.polygon = line.polygon * ds
    for region in page_layout.regions:
        region.polygon = region.polygon * ds
    return page_layout

class MouseEditorCallback(object):
    def __init__(self):
        self.points = []
        self.drawing = False
        self.down_time = 0
        self.down_pos = None
        self.click_time = 0.2
        self.double_clicked = False

    def del_point(self):
        self.points = self.points[:-1]
        if not self.points:
            self.drawing = False

    def clear(self):
        self.points = []
        self.drawing = False

    def get_selection_points(self):
        if self.drawing:
            return self.points[:-1]
        else:
            return self.points

    def get_remaining_points(self):
        if self.drawing:
            return self.points[-2:]
        else:
            return []

    def callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.clear()
            # dont start drawing again right after clearing points
            self.double_clicked = True

        elif event == cv2.EVENT_LBUTTONDOWN:
            self.down_pos = (x, y)
            self.down_time = time.time()

        elif event == cv2.EVENT_LBUTTONUP and time.time() - self.down_time < self.click_time:
            if not self.double_clicked:
                if not self.drawing:
                    self.points.append(self.down_pos)
                self.points.append(self.down_pos)
                self.drawing = True
            else:
                self.double_clicked = False

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing and self.points:
            self.points[-1] = (x, y)

class PageEditor(object):
    def __init__(self, line_thickness=2, downsample=1, show_hint=False, cursor=0):
        self.line_thickness = line_thickness
        self.ds = downsample
        self.hint = show_hint
        self.cursor = cursor

    def select_layout_elements(self):
        if self.clicker.points:
            self.selected_lines = select_polygons(
                [line.polygon for line in self.page_layout.lines_iterator()],
                self.clicker.get_selection_points())
            self.selected_regions = select_polygons(
            [region.polygon for region in self.page_layout.regions],
                self.clicker.get_selection_points())
            self.lines_to_select = select_polygons(
                [line.polygon for line in self.page_layout.lines_iterator()],
                self.clicker.get_remaining_points())
        else:
            self.selected_lines = []
            self.selected_regions = []
            self.lines_to_select = []

    def render_selected_elements(self, image):
        for r_num in self.selected_regions:
            layout.draw_lines(image, [self.page_layout.regions[r_num].polygon], color=(155, 70, 155), close=True, thickness=self.line_thickness)
        for l_num in self.selected_lines:
            layout.draw_lines(image, [self.lines[l_num].polygon], color=(155, 70, 155), close=True, thickness=self.line_thickness)
        for l_num in self.lines_to_select:
            if not l_num in self.selected_lines:
                layout.draw_lines(image, [self.lines[l_num].polygon], color=(155, 70, 155), close=True, thickness=self.line_thickness)
        return image

    def clear_interaction(self):
        self.clicker.clear()
        self.selected_lines.clear()
        self.lines_to_select.clear()
        self.selected_regions.clear()

    def update_selected_lines(self, bs=0, asc=0, dsc=0, start=0, end=0):
        for l_num in self.selected_lines:
            self.lines[l_num].heights = [self.lines[l_num].heights[0]+float(asc), self.lines[l_num].heights[1]+float(dsc)]
            self.lines[l_num].baseline[:,1] += float(bs)
            self.lines[l_num].baseline[0,0] += float(start)
            self.lines[l_num].baseline[-1,0] += float(end)
            self.lines[l_num].polygon = linepp.baseline_to_textline(self.lines[l_num].baseline, self.lines[l_num].heights)

        self.render()

    def create_line(self):
        if len(self.clicker.points) > 2:
            if self.lines:
                heights = np.median(np.array([[line.heights[0], line.heights[1]] for line in self.lines]), axis=0)
            else:
                heights = [5, 5]

            new_line = layout.TextLine(
                id = None,
                baseline = np.array(self.clicker.points[:-1], dtype=np.float),
                heights = heights
            )
            new_line.polygon = linepp.baseline_to_textline(new_line.baseline, new_line.heights)
            dummy_region = layout.RegionLayout(id='dummy', polygon=[[-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0]])
            dummy_region.lines.append(new_line)
            self.page_layout.regions.append(dummy_region)
            self.lines.append(new_line)
            self.clear_interaction()
            self.render()

    def create_region(self):
        if len(self.clicker.points) > 3:
            new_region = layout.RegionLayout(
                id = 'r{}'.format(len(self.page_layout.regions)+1),
                polygon = np.array(self.clicker.points[:-1], dtype=np.float))
            self.page_layout.regions.append(new_region)
            self.clear_interaction()
            self.render()

    def create_region_from_lines(self):
        if self.selected_lines:
            poly_points = []
            max_alpha = 0
            for l_num in self.selected_lines:
                poly_points += self.lines[l_num].polygon.tolist()
                # heuristic to estimate appropriate alpha for region polygon
                x_dist = np.amax(np.diff(self.lines[l_num].baseline[:,0]))
                height = self.lines[l_num].heights[0] + self.lines[l_num].heights[1]
                max_alpha = np.amax([max_alpha, x_dist, height])

            poly = region_engine.alpha_shape(np.stack(poly_points, axis=1).T, 1.5*max_alpha)
            new_region = layout.RegionLayout(
                id = 'r{}'.format(len(self.page_layout.regions)+1),
                polygon = np.array(poly.simplify(5).exterior.coords, dtype=np.float))
            self.page_layout.regions.append(new_region)
            self.clear_interaction()
            self.render()

    def remove_selected_lines(self):
        lines_to_remove = [self.lines[l_num] for l_num in self.selected_lines]
        for line_to_remove in lines_to_remove:
            self.lines.remove(line_to_remove)
            for region in self.page_layout.regions:
                if line_to_remove in region.lines:
                    region.lines.remove(line_to_remove)
        self.clear_interaction()
        self.render()

    def remove_selected_regions(self):
        if self.selected_regions:
            for r_num in self.selected_regions:
                self.page_layout.regions[r_num].id = 'dummy'
                self.page_layout.regions[r_num].polygon = [[-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0]]
            self.clear_interaction()
            self.render()

    def update_layout(self):
        # assign lines to regions
        for region in self.page_layout.regions:
            region.lines = []
            region_geometry = Polygon(region.polygon)
            for l_num, line in enumerate(self.lines):
                line = copy.deepcopy(line)
                line_geometry = LineString(line.baseline)
                if line_geometry.intersects(region_geometry):
                    bs_is, tl_is = linepp.mask_textline_by_region(line.baseline, line.polygon, region.polygon)
                    if bs_is is not None and tl_is is not None:
                        line.baseline = bs_is
                        line.polygon = tl_is
                        line.id = '{}-l{:03d}'.format(region.id, l_num)
                        region.lines.append(line)
        self.lines = [line for line in self.page_layout.lines_iterator()]
        # delete dummy regions
        for region in self.page_layout.regions:
            if region.id == 'dummy':
                self.page_layout.regions.remove(region)
        self.render()

    def render(self):
        self.rendered_image = self.page_layout.render_to_image(self.page_image.copy(), thickness=self.line_thickness, circles=False)

    def annotate(self, page_image, page_layout):

        self.page_image = cv2.resize(page_image, (0, 0), fx=1/self.ds, fy=1/self.ds)
        self.page_layout = resample_layout(page_layout, 1/self.ds)

        if self.hint:
            self.hint_image = cv2.imread(os.path.join(os.path.dirname(__file__), 'hint.png'))
            cv2.namedWindow('User help')
            cv2.imshow('User help', self.hint_image)

        # self.page_image = page_image
        # self.page_layout = page_layout

        self.lines = [line for line in self.page_layout.lines_iterator()]

        self.selected_lines = []
        self.selected_regions = []
        self.lines_to_select = []
        self.clicker = MouseEditorCallback()

        cv2.namedWindow('Textline Editor', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Textline Editor', self.clicker.callback)

        self.render()
        image = self.rendered_image.copy()
        while True:
            cv2.imshow('Textline Editor', image)
            key = cv2.waitKey(1)
            image = self.rendered_image.copy()

            ### render stuff each tick ###
            if self.clicker.points:
                layout.draw_lines(image, [self.clicker.points], close=False, color=(0, 150, 150), thickness=self.line_thickness)
                self.select_layout_elements()
                image = self.render_selected_elements(image)

            ### keyboard callbacks ###
            # line editing
            if key == ord('r'):
                self.update_selected_lines(bs=-1)
            elif key == ord('f'):
                self.update_selected_lines(bs=1)
            elif key == ord('t'):
                self.update_selected_lines(asc=1)
            elif key == ord('g'):
                self.update_selected_lines(asc=-1)
            elif key == ord('z'):
                self.update_selected_lines(dsc=-1)
            elif key == ord('h'):
                self.update_selected_lines(dsc=1)
            elif key == ord('i'):
                self.update_selected_lines(start=-1)
            elif key == ord('o'):
                self.update_selected_lines(start=1)
            elif key == ord('k'):
                self.update_selected_lines(end=-1)
            elif key == ord('l'):
                self.update_selected_lines(end=1)

            # line adding/deleting
            elif key == ord('y'):
                self.create_line()
            elif key == ord('x'):
                self.remove_selected_lines()

            # region adding/deleting
            elif key == ord('v'):
                self.create_region()
            elif key == ord('b'):
                self.create_region_from_lines()
            elif key == ord('n'):
                self.remove_selected_regions()

            # whole page ops
            elif key == ord('c'):
                self.clicker.del_point()
            elif key == ord('p'):
                self.update_layout()
            elif key == ord('a'):
                self.update_layout()
                self.cursor = -1
                print('SAVE AND EXIT')
                break
            elif key == ord('q'):
                self.update_layout()
                self.cursor -= 1
                print('SAVE AND MOVE TO PREVIOUS')
                break
            elif key == ord('w'):
                self.update_layout()
                self.cursor += 1
                print('SAVE AND MOVE TO NEXT')
                break

        return resample_layout(self.page_layout, self.ds)

def main():
    args = parse_arguments()

    assert args.page_dir is not None or args.output_dir is not None, "Specify input page folder and/or output page folder"
    assert os.path.exists(args.image_dir), "Can't find input image folder"
    if args.page_dir is not None:
        assert os.path.exists(args.page_dir), "Can't find input page folder"
    if args.output_dir is not None:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    filename_list = [x for x in os.listdir(args.image_dir)]
    editor = PageEditor(downsample=2, line_thickness=2, show_hint=True, cursor=0)

    while editor.cursor > -1 and editor.cursor < len(filename_list):

        page_filename = os.path.splitext(filename_list[editor.cursor])[0]+'.xml'
        cur_image = cv2.imread(os.path.join(args.image_dir, filename_list[editor.cursor]))

        if args.output_dir is not None and os.path.exists(os.path.join(args.output_dir, page_filename)):
            cur_layout = layout.PageLayout(file=os.path.join(args.output_dir, page_filename))
        elif args.page_dir is not None and os.path.exists(os.path.join(args.page_dir, page_filename)):
            cur_layout = layout.PageLayout(file=os.path.join(args.page_dir, page_filename))
        else:
            cur_layout = layout.PageLayout(
                id=os.path.splitext(filename_list[editor.cursor])[0],
                page_size=(cur_image.shape[1], cur_image.shape[0]))

        edited_layout = editor.annotate(cur_image, cur_layout)
        if args.output_dir is not None:
            cur_layout.to_pagexml(os.path.join(args.output_dir, page_filename))
        else:
            cur_layout.to_pagexml(os.path.join(args.page_dir, page_filename))

if __name__=='__main__':
    main()
