import numpy as np
import cv2

class Net(object):

    def __init__(self, model_path, use_cpu=False, prefix='prefix',
                 pad=52, max_mp=5, gpu_fraction=None):
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
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
                tf_config = tf.ConfigProto(intra_op_parallelism_threads=6, inter_op_parallelism_threads=1, device_count={'GPU': 0})
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
        test_img_canvas = np.zeros((1, new_shape_x, new_shape_y, 3), dtype=np.float32)
        test_img_canvas[0, :img.shape[0], :img.shape[1], :] = img
        print("LAYOUT_CNN_DOWNSAMPLE", downsample, 'INPUT_SHAPE', test_img_canvas.shape)

        if self.prefix is not None:
            out_map = self.session.run(
                '{}/test_probs:0'.format(self.prefix),
                feed_dict={'{}/test_dataset:0'.format(self.prefix): test_img_canvas[:, :, :] / np.float32(256.)})
        else:
            out_map = self.session.run(
                'test_probs:0',
                feed_dict={'test_dataset:0': test_img_canvas[:, :, :] / np.float32(256.)})

        out_map = out_map[0, self.pad:img.shape[0] - self.pad, self.pad:img.shape[1] - self.pad, :]

        return out_map


class ParseNet(Net):

    def __init__(self, model_path, downsample=4, use_cpu=False, prefix='parsenet',
                 pad=52, max_mp=5, gpu_fraction=None, detection_threshold=0.2, adaptive_downsample=True):

        super().__init__(
            model_path, use_cpu=use_cpu, prefix=prefix,
            pad=pad, max_mp=max_mp, gpu_fraction=gpu_fraction)

        self.detection_threshold = detection_threshold
        self.adaptive_downsample = adaptive_downsample
        self.init_downsample = downsample
        self.last_downsample = downsample
        self.downsample_line_pixel_adapt_threshold = 100
        self.min_line_processing_height = 8
        self.max_line_processing_height = 13
        self.optimal_line_processing_height = 11
        self.min_downsample = 1
        self.max_downsample = 8


    def get_maps_with_optimal_resolution(self, img):
        '''
        Memory-safe Parsenet CNN inference with optimal downsampling
        '''
        # check that big images are rescaled before first CNN run

        first_downsample = max(
            self.last_downsample,
            np.sqrt((img.shape[0] * img.shape[1]) / (self.max_megapixels * 10e5)))

        # first run with default downsample
        net_downsample = first_downsample
        out_map = self.get_maps(img, net_downsample)
        if not self.adaptive_downsample:
            return out_map, net_downsample

        second_downsample = first_downsample
        if (out_map[:, :, 2] > self.detection_threshold).sum() > self.downsample_line_pixel_adapt_threshold:
            med_height = self.get_med_height(out_map)
            #print('MEDIAN HEIGHT', med_height, med_height * first_downsample)
            if med_height > self.max_line_processing_height or med_height < self.min_line_processing_height:
                second_downsample = first_downsample * (med_height / self.optimal_line_processing_height)
                second_downsample = min(second_downsample, self.max_downsample)
                second_downsample = max(second_downsample, self.min_downsample)
                self.last_downsample = second_downsample
                second_downsample = max(
                    self.last_downsample,
                    np.sqrt((img.shape[0] * img.shape[1]) / (self.max_megapixels * 10e5)))

                if second_downsample / first_downsample < 0.8 or second_downsample / first_downsample > 1.2:
                    net_downsample = second_downsample
                    out_map = self.get_maps(img, net_downsample)

        return out_map, net_downsample

    def get_med_height(self, out_map):
        '''
        Compute median line height from CNN output
        '''
        heights = (out_map[:, :, 2] > self.detection_threshold).astype(np.float) * out_map[:, :, 0]
        med_height = np.median(heights[heights > 0])

        return med_height


class TiltNet(Net):
    def __init__(self, model_path, use_cpu=False, prefix='tiltnet',
                 pad=52, max_mp=5, gpu_fraction=None):
        super().__init__(
            model_path, use_cpu=use_cpu, prefix=prefix,
            pad=pad, max_mp=max_mp, gpu_fraction=gpu_fraction)
