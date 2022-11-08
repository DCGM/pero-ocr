import numpy as np
import cv2
import torch

class Net(object):

    def __init__(self, model_path, device, max_mp=5):
        self.max_megapixels = max_mp if max_mp is not None else 5

        self.device = device
        if self.device.type == "cpu":
            model_path += ".cpu"

        if model_path is not None:
            self.net = torch.jit.load(model_path, map_location=self.device)
        else:
            self.net = None


class TorchParseNet(Net):

    def __init__(self, model_path, device, downsample=4, max_mp=5, detection_threshold=0.2, adaptive_downsample=True):

        super().__init__(model_path, device=device, max_mp=max_mp)

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

    def get_maps(self, img, downsample):
        '''
        ParseNet CNN inference
        '''

        img = cv2.resize(img, (0, 0), fx=1/downsample, fy=1/downsample, interpolation=cv2.INTER_AREA)

        new_shape_x = int(np.ceil(img.shape[0] / 64) * 64)
        new_shape_y = int(np.ceil(img.shape[1] / 64) * 64)
        test_img_canvas = np.zeros((1, new_shape_x, new_shape_y, 3), dtype=np.uint8)
        test_img_canvas[0, :img.shape[0], :img.shape[1], :] = img

        with torch.no_grad():
            print(f'NET INPUT {new_shape_x * new_shape_y} Mpx.')
            test_img_canvas = torch.from_numpy(test_img_canvas).to(self.device).float().permute(0, 3, 1, 2) * (1/255.)
            out_map, _ = self.net(test_img_canvas)
            out_map = out_map.permute(0, 2, 3, 1).cpu().numpy()
        if self.device != 'cuda':
            torch.cuda.empty_cache()
        out_map = out_map[0, :img.shape[0], :img.shape[1], :]

        return out_map

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
    def __init__(self, model_path, device, max_mp=5):
        super().__init__(model_path, device=device, max_mp=max_mp)

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
