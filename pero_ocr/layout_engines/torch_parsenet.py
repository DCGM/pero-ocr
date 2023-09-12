import numpy as np
import cv2
import torch
import multiprocessing
from typing import List
import pickle
import zmq
import time
import msgpack
import msgpack_numpy as m


class Net(object):

    def __init__(self, model_path, use_cpu=False, max_mp=5):
        self.max_megapixels = max_mp if max_mp is not None else 5

        if use_cpu:
            self.device = 'cpu'
        else:
            self.device = 'cuda'

        if model_path is not None:
            self.net = torch.jit.load(model_path, map_location=self.device)
        else:
            self.net = None



class NetProcess(multiprocessing.Process):
    def __init__(self, model_file):
        super(NetProcess, self).__init__(daemon=True)
        self.model_file = model_file
        self.device = None
        self.counter = 0

    def _load_exported_model(self):
        if self.device.type == "cpu":
            self.checkpoint += ".cpu"
        self.model = torch.jit.load(self.model_file, map_location=self.device)
        self.model = self.model.to(self.device)

    def run(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("ipc:///tmp/0")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_exported_model()

        with torch.no_grad():
            while True:
                with open('log.txt', 'a') as f:
                    t1 = time.time()
                    batch_data = self.socket.recv()
                    print('XXX 1', time.time() - t1, file=f)
                    batch_data = msgpack.unpackb(batch_data, object_hook=m.decode)
                    print('XXX 2', time.time() - t1, file=f)
                    batch_data = torch.from_numpy(batch_data).to(self.device).float() * (1/255.)
                    batch_data = batch_data.permute(0, 3, 1, 2)
                    print('XXX 3', time.time() - t1, file=f)
                    result, _ = self.model(batch_data)
                    print('XXX 4', time.time() - t1, file=f)

                    result = result.to(torch.float16).permute(0, 2, 3, 1).cpu().numpy()
                    mpxs = batch_data.shape[2] * batch_data.shape[3] / 1e6 / (time.time() - t1)
                    print('XXX 5', time.time() - t1, batch_data.shape, mpxs, file=f)
                    msg = msgpack.packb(result, default=m.encode)
                    self.socket.send(msg)
                    print('XXX 6', time.time() - t1, len(msg) / 1e6, result.shape, file=f)
                    self.counter += 1
                    if self.counter % 100 == 0:
                        torch.cuda.empty_cache()

class TorchParseNet(object):

    def __init__(self, model_path, downsample=4, use_cpu=False, max_mp=5, detection_threshold=0.2, adaptive_downsample=True, start_engines=True):

        super().__init__()
        if start_engines:
            print('STARTING ENGINES')
            multiprocessing.set_start_method('spawn')
            NetProcess(model_path).start()
        else:
            print('NOT STARTING ENGINES')


        self.max_megapixels = max_mp if max_mp  else 5
        self.detection_threshold = detection_threshold
        self.adaptive_downsample = adaptive_downsample
        self.init_downsample = downsample
        self.last_downsample = downsample
        self.downsample_line_pixel_adapt_threshold = 100
        self.min_line_processing_height = 9
        self.max_line_processing_height = 15
        self.optimal_line_processing_height = 12
        self.min_downsample = 1
        self.max_downsample = 8
        self.socket = None

    def get_socket(self):
        if not self.socket:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect("ipc:///tmp/0")
        return self.socket

    def get_maps(self, img, downsample):
        '''
        ParseNet CNN inference
        '''
        socket = self.get_socket()
        img = cv2.resize(img, (0, 0), fx=1/downsample, fy=1/downsample, interpolation=cv2.INTER_AREA)

        new_shape_x = int(np.ceil(img.shape[0] / 64) * 64)
        new_shape_y = int(np.ceil(img.shape[1] / 64) * 64)
        test_img_canvas = np.zeros((1, new_shape_x, new_shape_y, 3), dtype=np.uint8)
        test_img_canvas[0, :img.shape[0], :img.shape[1], :] = img

        print(f'NET INPUT {new_shape_x * new_shape_y} Mpx.')
        t1 = time.time()
        msg = msgpack.packb(test_img_canvas, default=m.encode)
        print('1', time.time() - t1)
        socket.send(msg)
        print('2', time.time() - t1)
        msg = socket.recv()
        print('3', time.time() - t1)
        out_map = msgpack.unpackb(msg, object_hook=m.decode)
        print('4', time.time() - t1)
        out_map_2 = out_map[0, :img.shape[0], :img.shape[1], :].astype(np.float32)

        return out_map_2

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


class TorchOrientationNet(Net):
    def __init__(self, model_path, use_cpu=False, max_mp=5, start_engines=True):
        super().__init__(model_path, use_cpu=use_cpu, max_mp=max_mp)

        if not start_engines:
            logging.error('Stat engines "False" is not supported for TorchOrientationNet')
            exit(-1)

    def get_maps(self, img, downsample):
        '''
        OrientationNet CNN inference
        '''
        img = cv2.resize(img, (0, 0), fx=1/downsample, fy=1/downsample, interpolation=cv2.INTER_AREA)
        img = img / np.float32(256.)

        new_shape_x = int(np.ceil(img.shape[0] / 64) * 64)
        new_shape_y = int(np.ceil(img.shape[1] / 64) * 64)
        test_img_canvas = np.zeros((1, new_shape_x, new_shape_y, 3), dtype=np.float32)
        test_img_canvas[0, :img.shape[0], :img.shape[1], :] = img

        test_img_canvas = torch.from_numpy(test_img_canvas).to(self.device).float().permute(0, 3, 1, 2)
        out_map = self.net(test_img_canvas)
        out_map = out_map.permute(0, 2, 3, 1).cpu().numpy()
        out_map = out_map[0, :img.shape[0], :img.shape[1], :]

        return out_map
