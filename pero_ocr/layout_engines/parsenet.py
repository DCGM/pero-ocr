import numpy as np
import cv2

class Net(object):

    def __init__(self, model_path, downsample=4, use_cpu=False, prefix='prefix',
                 pad=52, max_mp=5, gpu_fraction=None):

        import tensorflow as tf
        self.downsample = downsample  # downsample before first CNN inference
        self.pad = pad
        self.max_megapixels = max_mp if max_mp is not None else 5
        self.gpu_fraction = gpu_fraction

        self.prefix = prefix
        if model_path is not None:
            with tf.gfile.GFile(model_path, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
            with tf.Graph().as_default() as graph:
                tf.import_graph_def(graph_def, name=prefix)
                self.graph = graph
            print(f"{model_path} loaded")

            if use_cpu:
                tf_config = tf.ConfigProto(device_count={'GPU': 0})
            else:
                tf_config = tf.ConfigProto(device_count={'GPU': 1})
                if self.gpu_fraction is None:
                    tf_config.gpu_options.allow_growth = True
                else:
                    tf_config.gpu_options.per_process_gpu_memory_fraction = self.gpu_fraction
            self.session = tf.Session(graph=self.graph, config=tf_config)

            if self.prefix is not None:
                out_map = self.session.run(
                    '{}/test_probs:0'.format(self.prefix),
                    feed_dict={'{}/test_dataset:0'.format(self.prefix): np.zeros([1, 128, 128, 3], dtype=np.uint8)}
                    )
            else:
                out_map = self.session.run(
                    'test_probs:0',
                    feed_dict={'test_dataset:0': np.zeros([1, 128, 128, 3], dtype=np.uint8)}
                )
            print('graph initialized')

    def get_maps(self, img, downsample):
        '''
        Parsenet CNN inference
        '''
        img = cv2.resize(img, (0,0), fx=1/downsample, fy=1/downsample, interpolation=cv2.INTER_AREA)
        img = np.pad(img, [(self.pad, self.pad), (self.pad, self.pad), (0, 0)], 'constant')

        new_shape_x = int(np.ceil(img.shape[0] / 64) * 64)
        new_shape_y = int(np.ceil(img.shape[1] / 64) * 64)
        test_img_canvas = np.zeros((1, new_shape_x, new_shape_y, 3))
        test_img_canvas[0, :img.shape[0], :img.shape[1], :] = img

        if self.prefix is not None:
            out_map = self.session.run(
                '{}/test_probs:0'.format(self.prefix),
                feed_dict={'{}/test_dataset:0'.format(self.prefix): test_img_canvas[:, :, :] / 256.})
            out_map = out_map[0, self.pad:img.shape[0]-self.pad, self.pad:img.shape[1]-self.pad, :]
        else:
            out_map = self.session.run(
                'test_probs:0',
                feed_dict={'test_dataset:0': test_img_canvas[:, :, :] / 256.})
            out_map = out_map[0, self.pad:img.shape[0] - self.pad, self.pad:img.shape[1] - self.pad, :]

        return out_map


class ParseNet(Net):

    def __init__(self, model_path, downsample=4, use_cpu=False, prefix='parsenet',
                 pad=52, max_mp=5, gpu_fraction=None, detection_threshold=0.2):

        super().__init__(
            model_path, downsample=downsample, use_cpu=False, prefix=prefix,
            pad=pad, max_mp=max_mp, gpu_fraction=gpu_fraction)

        self.detection_threshold = detection_threshold
        self.tmp_downsample = None

    def get_maps_with_optimal_resolution(self, img):
        '''
        Memory-safe Parsenet CNN inference with optimal downsampling
        '''
        # check that big images are rescaled before first CNN run
        downsample = self.downsample
        if (img.shape[0]/downsample) * (img.shape[1]/downsample) > self.max_megapixels * 10e5:
            downsample = np.sqrt((img.shape[0] * img.shape[1]) / (self.max_megapixels * 10e5))
        # first run with default downsample
        out_map = self.get_maps(img, downsample)
        # adapt second CNN run so that text height is between 10 and 14 downscaled pixels
        med_height = self.get_med_height(out_map)
        if med_height > 14 or med_height < 10:
            downsample = max(
                    np.sqrt((img.shape[0] * img.shape[1]) / (self.max_megapixels * 10e5)),
                    downsample * (med_height / 12)
                    )
            out_map = self.get_maps(img, downsample)
        self.tmp_downsample = downsample

        return out_map

    def get_med_height(self, out_map):
        '''
        Compute median line height from CNN output
        '''
        heights = (out_map[:, :, 2] > self.detection_threshold).astype(np.float) * out_map[:, :, 0]
        med_height = np.median(heights[heights > 0])

        return med_height


class TiltNet(Net):
    def __init__(self, model_path, downsample=4, use_cpu=False, prefix='tiltnet',
                 pad=52, max_mp=5, gpu_fraction=None, detection_threshold=0.2):

        super().__init__(
            model_path, downsample=downsample, use_cpu=False, prefix=prefix,
            pad=pad, max_mp=max_mp, gpu_fraction=gpu_fraction)
