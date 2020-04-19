import numpy as np
import cv2
from scipy import interpolate
from numba import jit


class EngineLineCropper(object):

    def __init__(self, correct_slant=False, line_height=32, poly=0, scale=1, blend_border=4):
        self.correct_slant = correct_slant
        self.line_height = line_height
        self.poly = poly
        self.scale = scale
        self.blend_border = blend_border

    @jit
    def reverse_value_mapping(self, forward_mapping, sample_positions, sampled_values):
        backward_mapping = np.zeros_like(sample_positions)
        forward_position = 0
        for i in range(sample_positions.shape[0]):
            while forward_mapping[forward_position] > sample_positions[i]:
                forward_position += 1
            d = forward_mapping[forward_position] - forward_mapping[forward_position-1]
            da = (sample_positions[i] - forward_mapping[forward_position-1]) / d
            backward_mapping[i] = (1 - da) * sampled_values[forward_position - 1] + da * sampled_values[forward_position]
        return backward_mapping

    def get_crop_inputs(self, baseline, line_heights, target_height):
        line_heights = [line_heights[0], line_heights[1]]

        coords = np.asarray(baseline).copy().astype(int)
        if self.poly:
            if coords.shape[0] > 2:
                line_interpf = np.poly1d(np.polyfit(coords[:,0], coords[:,1], self.poly))
            else:
                line_interpf = interpolate.interp1d(coords[:,0], coords[:,1], kind='linear',)
        else:
            try:
                line_interpf = interpolate.interp1d(coords[:,0], coords[:,1], kind='cubic',)
            except: # fall back to linear interpolation in case y_values fails (usually with very short baselines)
                line_interpf = interpolate.interp1d(coords[:,0], coords[:,1], kind='linear',)

        left = coords[:, 0].min()
        right = coords[:, 0].max()
        line_x_values = np.arange(left, right)
        line_y_values = line_interpf(line_x_values) # positions in source
        line_length = ((line_x_values[:-1] - line_x_values[1:])**2 + (line_y_values[:-1] - line_y_values[1:])**2) ** 0.5
        mapping_x_to_line_pos = np.concatenate([np.zeros(1), np.cumsum(line_length)]) # mapping of source to t

        scale = target_height / (line_heights[0] + line_heights[1])

        horizontal_sample_count = int(mapping_x_to_line_pos[-1] * scale) # number of target samples

        tmp = np.linspace(0, mapping_x_to_line_pos[-1], horizontal_sample_count)
        output_x_positions = self.reverse_value_mapping( # get source x baseline positions in target pixels
            mapping_x_to_line_pos, tmp, line_x_values)

        output_y_positions = line_interpf(output_x_positions) # get source baseline y positions in target pixels

        d_x = np.full_like(output_x_positions, 0.1)
        d_y = output_y_positions - line_interpf(output_x_positions + 0.1)
        norm_scales = (d_x**2 + d_y**2) ** 0.5 # get normals

        norm_x = -d_y / norm_scales
        norm_y = d_x / norm_scales

        vertical_map = np.linspace(-line_heights[0], line_heights[1], target_height).reshape(-1, 1)
        vertical_map_x = norm_x.reshape(1, -1) * vertical_map + output_x_positions.reshape(1, -1) # get the rest of source x positions for target pixels computed from normals
        vertical_map_y = norm_y.reshape(1, -1) * vertical_map + output_y_positions.reshape(1, -1) # get the rest of source y positions for target pixels computed from normals

        coords = np.stack((vertical_map_x, vertical_map_y), axis=2).astype(np.float32)

        return coords

    def crop(self, img, baseline, heights, return_mapping=False):
        try:
            line_coords = self.get_crop_inputs(baseline, heights, self.line_height)

            line_crop = cv2.remap(img, line_coords[:, :, 0], line_coords[:, :, 1],
                                      interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        except:
            print("ERROR: line crop failed.", heights, baseline)
            line_crop = np.zeros([self.line_height, 32, 3], dtype=np.uint8)

        if return_mapping:
            line_mapping = self.reverse_mapping_fast(line_coords, img.shape)
            return line_crop, line_mapping
        else:
            return line_crop

    def blend_in(self, img, line_crop, mapping):

        mapping = np.nan_to_num(mapping).astype(np.float32)

        blended_img = img.copy()
        cv2.remap(line_crop, mapping[:, :, 0], mapping[:, :, 1],
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT,
            dst=blended_img)
        mask = (mapping[:,:,1] > -1)
        mask = ndimage.morphology.binary_erosion(mask, iterations=self.blend_border).astype(np.float)
        blur_kernel = np.ones((self.blend_border+1, self.blend_border+1)) / (self.blend_border+1) ** 2
        mask = ndimage.convolve(mask, blur_kernel)
        mask = mask[:, :, np.newaxis]
        blended_img = (1 - mask) * img + mask * blended_img

        return np.round(blended_img).astype(np.uint8)

    def rigid_crop(self, img, baseline, height):
        line_height = height[1] + height[0]
        dy = 10
        do = height[0] / 10
        up = height[1] / 10

        baseline = np.asarray(baseline)

        one_line_points = []

        p1 = baseline[0, :]
        p2 = baseline[-1, :]
        dir = (p2 - p1)
        dir = dir / (dir ** 2).sum() ** 0.5
        n = dir[::-1] * dy
        n[0] = -n[0]
        p1 = p1 - dir * dy * 0.5
        p2 = p2 + dir * dy * 0.5

        pts1 = np.asarray([p1 - do * n, p2 - do * n, p2 + up * n, p1 - up * n]).astype(np.float32)
        one_line_points.append(np.copy(pts1))
        pts1 = pts1[:3]

        width = ((p2 - p1) ** 2).sum() ** 0.5 / (up + do) / dy * line_height
        pts2 = np.asarray([(0, 0), (width, 0), (width, line_height)]).astype(np.float32)
        pts1 = np.reshape(pts1, [-1, 1, 2])
        pts2 = np.reshape(pts2, [-1, 1, 2])

        T = cv2.getAffineTransform(pts1, pts2)
        line_crop = cv2.warpAffine(img, T, (int(width + 0.5), int(line_height)))

        return line_crop

    @jit
    def reverse_mapping_fast(self, forward_mapping, shape):
        y_mapping = forward_mapping[:,:,1]
        y_mapping = np.clip(cv2.resize(y_mapping, (0,0), fx=4, fy=4, interpolation=cv2.INTER_LINEAR), 0, shape[1]-1)
        y_mapping = np.round(y_mapping).astype(np.int)
        x_mapping = forward_mapping[:,:,0]
        x_mapping = np.clip(cv2.resize(x_mapping, (0,0), fx=4, fy=4, interpolation=cv2.INTER_LINEAR), 0, shape[0]-1)
        x_mapping = np.round(x_mapping).astype(np.int)

        y_map = np.tile(np.arange(0, forward_mapping.shape[0]), (forward_mapping.shape[1], 1)).T.astype(np.float32)
        y_map = cv2.resize(y_map, (0,0), fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
        x_map = np.tile(np.arange(0, forward_mapping.shape[1]), (forward_mapping.shape[0], 1)).astype(np.float32)
        x_map = cv2.resize(x_map, (0,0), fx=4, fy=4, interpolation=cv2.INTER_LINEAR)

        reverse_mapping = np.ones((shape[0], shape[1], 2), dtype=np.float32) * -1

        for sx, sy, dx, dy in zip(x_map.flatten(), y_map.flatten(), x_mapping.flatten(), y_mapping.flatten()):
            reverse_mapping[dy, dx, 0] = sx
            reverse_mapping[dy, dx, 1] = sy

        return reverse_mapping
