import tensorflow as tf
import numpy as np
import cv2
import shapely
from scipy import ndimage
from scipy import signal
from scipy.ndimage.morphology import binary_erosion
from skimage.draw import polygon2mask

from . import line_postprocessing as pp

class EngineLineDetectorSimple(object):
    def __init__(self, adaptive_threshold=91, block_size=20,
                 minimum_length=6, ignored_border_pixels=10):
        self.adaptive_threshold = adaptive_threshold
        self.block_size = block_size
        self.minimum_length = minimum_length
        self.ignored_border_pixels = ignored_border_pixels

    def detect_lines(self, img, region):
        """Performs simple line extraction in single text region using thresholding,
        correlation and connected component analysis.
        :param img: input image array
        :param region: target region polygon
        """

        baselines_list = []
        heights_list = []

        y1 = np.amin(region[:, 0].astype(np.int32))
        y2 = np.amax(region[:, 0].astype(np.int32))
        x1 = np.amin(region[:, 1].astype(np.int32))
        x2 = np.amax(region[:, 1].astype(np.int32))

        if y2 == y1 or x1 == x2:
            return [], [], []

        column_width = x2 - x1
        column_height = y2 - y1

        img_mask = polygon2mask(img.shape[0:2], region)
        img_mask = img_mask[y1:y2, x1:x2]
        img_mask = binary_erosion(img_mask, structure=np.ones((1, 2 * self.ignored_border_pixels + 1)))

        img_crop = img[y1:y2, x1:x2, :]
        img_crop = img_crop.mean(axis=2).astype(np.uint8)
        img_crop = cv2.adaptiveThreshold(img_crop, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, self.block_size, self.adaptive_threshold) == 0

        img_crop = img_crop * img_mask

        img_crop_labeled, num_features = ndimage.measurements.label(img_crop)
        proj = np.sum(img_crop, axis=1)
        corr = np.correlate(proj, proj, mode='full')[proj.shape[0]:]
        corr_peaks = signal.find_peaks(corr, prominence=0, distance=1)[0]
        if len(corr_peaks) > 0:
            line_period = float(signal.find_peaks(corr, prominence=0, distance=1)[0][0])
        else:
            line_period = 1
        target_signal = - np.diff(proj)
        target_signal[target_signal < 0] = 0

        baseline_coords = signal.find_peaks(target_signal, distance=int(round(0.85*line_period)))[0]
        region = shapely.geometry.polygon.Polygon(region)
        used_inds = []

        for baseline_coord in baseline_coords[::-1]:
            valid_baseline = True
            matching_objects = np.unique(img_crop_labeled[baseline_coord-10, :])[1:]
            if len(matching_objects) > 0:
                for ind in matching_objects:
                    if ind in used_inds:
                        valid_baseline = False
                    used_inds.append(ind)

                for yb1 in range(baseline_coord, 0, -3):
                    line_inds_to_check = img_crop_labeled[yb1, :]
                    if not np.any(np.intersect1d(matching_objects, line_inds_to_check)):
                        break

                for yb2 in range(baseline_coord, column_height, 3):
                    line_inds_to_check = img_crop_labeled[yb2, :]
                    if not np.any(np.intersect1d(matching_objects, line_inds_to_check)):
                        break

                xb1, xb2 = 0, column_width

                if yb2 - yb1 < self.minimum_length:
                    valid_baseline = False

                line = shapely.geometry.LineString([[y1+baseline_coord, x1+xb1-20], [y1+baseline_coord, x1+xb2+20]])
                intersection = region.intersection(line)
                if not intersection.is_empty:
                    if valid_baseline:
                        baselines_list.append(np.round(np.asarray(list(region.intersection(line).coords[:]))).astype(np.int16))
                        heights_list.append([baseline_coord-yb1, yb2-baseline_coord])

        textlines_list = [pp.baseline_to_textline(baseline, heights) for baseline, heights in zip(baselines_list, heights_list)]

        return baselines_list, heights_list, textlines_list


class EngineLineDetectorCNN(object):
    def __init__(self, model_path, downsample=4, pad=50, use_cpu=False,
                 order_lines='reading_order', detection_threshold=0.5,
                 stretch_lines=0):

        self.downsample = downsample
        self.pad = pad
        self.order_lines = order_lines
        self.detection_threshold = detection_threshold
        self.stretch_lines = stretch_lines

        tf.reset_default_graph()
        saver = tf.train.import_meta_graph(model_path + '.meta')
        if use_cpu:
            tf_config = tf.ConfigProto(device_count={'GPU': 0})
        else:
            tf_config = tf.ConfigProto(device_count={'GPU': 1})
            tf_config.gpu_options.allow_growth = True
        self.session = tf.Session(config=tf_config)
        saver.restore(self.session, model_path)

    def infer_maps(self, img):
        """CNN Model inference for baseline pixelwise probabilities and heights.
        :param img: input image array
        """

        img = cv2.resize(img, (0,0), fx=1/self.downsample, fy=1/self.downsample)
        img = np.pad(img, [(self.pad, self.pad), (self.pad, self.pad), (0,0)], 'constant')

        new_shape_x = img.shape[0]
        new_shape_y = img.shape[1]
        while not new_shape_x % 64 == 0:
            new_shape_x += 1
        while not new_shape_y % 64 == 0:
            new_shape_y += 1
        test_img_canvas = np.zeros((1, new_shape_x, new_shape_y, 3))
        test_img_canvas[0, :img.shape[0], :img.shape[1], :] = img

        out_map = self.session.run('test_probs:0', feed_dict={'test_dataset:0' : test_img_canvas/256.})
        out_map = out_map[0, :img.shape[0], :img.shape[1], :]
        out_map = out_map[self.pad:-self.pad,self.pad:-self.pad,:]
        heights_map = out_map[:,:,:2].astype(np.uint16)
        baselines_map = pp.nonmaxima_suppression(out_map[:,:,2]-out_map[:,:,3]) > self.detection_threshold

        return baselines_map, heights_map

    def parse_maps(self, baselines_map, heights_map):
        """Parse input baseline and height map into list of baselines coords and heights
        :param baseline_map: array of baseline and endpoint probabilities
        :param heights_map: array of estimated heights
        """
        baselines_list = []
        heights_list = []

        baselines_img, num_detections = ndimage.measurements.label(baselines_map, structure=np.ones((3, 3)))
        inds = np.where(baselines_img > 0)
        labels = baselines_img[inds[0], inds[1]]
        inds = np.stack([inds[0], inds[1]], axis=1)

        for i in range(1, num_detections+1):
            baseline_inds, = np.where(labels == i)
            if len(baseline_inds) > 15:
                pos = inds[baseline_inds]
                _, indices = np.unique(pos[:, 1], return_index=True)
                pos = pos[indices]
                x_index = np.argsort(pos[:, 1])
                pos = pos[x_index]

                pos_step = np.amax([15, pos.shape[0]//10])
                pos = np.concatenate([pos[::pos_step, :], pos[pos.shape[0]-1:pos.shape[0]]], axis=0)

                pos = pos.tolist()
                if pos[-1] == pos[-2]:
                    pos = pos[:-1]
                pos = np.asarray(pos, dtype=np.int32)

                heights_pred = heights_map * (baselines_img == i)[:, :, np.newaxis]
                if np.amax(heights_pred[:, :, 0]) > 0 and np.amax(heights_pred[:, :, 1]) > 0:  # percentile will fail on zero vector, discard the baseline in such case
                    heights_pred = np.asarray([
                        np.percentile(heights_pred[:, :, 0][heights_pred[:, :, 0] > 0], 90),
                        np.percentile(heights_pred[:, :, 1][heights_pred[:, :, 1] > 0], 90)
                    ])
                    baselines_list.append(self.downsample * pos)
                    heights_list.append([int(self.downsample * round(heights_pred[0])),
                                         int(self.downsample * round(heights_pred[1]))])

        return baselines_list, heights_list

    def detect_lines(self, img):
        """Detect lines in document image.
        :param img: input image array
        """
        baselines_map, heights_map = self.infer_maps(img)
        baselines_list, heights_list = self.parse_maps(baselines_map, heights_map)

        if self.stretch_lines > 0:
            baselines_list = pp.stretch_baselines(baselines_list, self.stretch_lines)

        rotation = pp.get_rotation(baselines_list)
        baselines_list = [pp.rotate_coords(baseline, rotation, (0, 0)) for baseline in baselines_list]

        textlines_list = []
        for baseline, height in zip(baselines_list, heights_list):
            textlines_list.append(pp.baseline_to_textline(baseline, height))

        if self.order_lines == 'vertical':
            baselines_list, heights_list, textlines_list = pp.order_lines_vertical(baselines_list, heights_list, textlines_list)
        elif self.order_lines == 'reading_order':
            baselines_list, heights_list, textlines_list = pp.order_lines_general(baselines_list, heights_list, textlines_list)
        else:
            raise ValueError("Argument order_lines must be either 'vertical' or 'reading_order'.")

        textlines_list = [pp.rotate_coords(textline, -rotation, (0, 0)) for textline in textlines_list]
        baselines_list = [pp.rotate_coords(baseline, -rotation, (0, 0)) for baseline in baselines_list]

        return baselines_list, heights_list, textlines_list


if __name__ == '__main__':
    from pero_ocr.document_ocr import layout

    test_layout = layout.PageLayout(id='test')
    test_p = layout.RegionLayout('r', np.zeros((4, 2)))

    engine_instance = EngineLineDetectorCNN('/mnt/matylda1/hradis/PERO/layout_engines/baselines_prod/parsenet_multi_alldata_fix_exported')
    image = cv2.imread('/mnt/matylda1/ikodym/junk/refactor_test/8e41ecc2-57ed-412a-aa4f-d945efa7c624.jpg')
    baselines, heights, textlines = engine_instance.detect_lines(image)

    for baseline, height, textline in zip(baselines, heights, textlines):
        new_textline = layout.TextLine(baseline=baseline, heights=height, polygon=textline)
        test_p.lines.append(new_textline)

    test_layout.regions.append(test_p)
    test_layout.render_to_image(image, '/mnt/matylda1/ikodym/junk/refactor_test/')
