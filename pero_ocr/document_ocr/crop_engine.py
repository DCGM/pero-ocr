import numpy as np
import cv2
from scipy import interpolate, ndimage
# from numba import jit
from pero_ocr.utils import jit

class EngineLineCropper(object):
    def __init__(self, correct_slant=False, line_height=32, poly=0, scale=1, blend_border=4):
        self.correct_slant = correct_slant
        self.line_height = line_height
        self.poly = poly
        self.scale = scale
        self.blend_border = blend_border

    def crop(self, img, baseline, heights, return_mapping=False):
        try:
            line_coords = self.get_crop_inputs(baseline, heights, self.line_height)

            line_crop = cv2.remap(img, line_coords[:, :, 0], line_coords[:, :, 1],
                                      interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        except:
            print("ERROR: line crop failed.", heights, baseline)
            line_crop = np.zeros([self.line_height, 32, 3], dtype=np.uint8)

        if return_mapping:
            line_mapping, offset = self.reverse_xy_mapping(line_coords, img.shape)
            return line_crop, line_mapping, offset
        else:
            return line_crop

    def blend_in(self, img, line_crop, mapping, offset):
        ystart = offset[0]
        ystop = ystart + mapping.shape[0]
        xstart = offset[1]
        xstop = xstart + mapping.shape[1]

        blended_img = img[ystart:ystop,xstart:xstop].copy()
        mask = self.get_blend_mask(mapping)

        cv2.remap(line_crop, mapping[:, :, 0], mapping[:, :, 1],
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT,
            dst=blended_img)

        blended_img = np.round(
            (1 - mask) * img[ystart:ystop,xstart:xstop] +
            mask * blended_img
            ).astype(np.uint8)

        img[ystart:ystop,xstart:xstop] = blended_img

        return img

    def get_crop_inputs(self, baseline, line_heights, target_height):
        line_heights = [line_heights[0] * self.scale, line_heights[1] * self.scale]
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
        output_x_positions = self.reverse_line_mapping( # get source x baseline positions in target pixels
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

    @jit
    def reverse_line_mapping(self, forward_mapping, sample_positions, sampled_values):
        backward_mapping = np.zeros_like(sample_positions)
        forward_position = 0
        for i in range(sample_positions.shape[0]):
            while forward_mapping[forward_position] > sample_positions[i]:
                forward_position += 1
            d = forward_mapping[forward_position] - forward_mapping[forward_position-1]
            da = (sample_positions[i] - forward_mapping[forward_position-1]) / d
            backward_mapping[i] = (1 - da) * sampled_values[forward_position - 1] + da * sampled_values[forward_position]
        return backward_mapping

    # @jit
    def reverse_xy_mapping(self, forward_mapping, shape):

        y_mapping = forward_mapping[:,:,1]
        y_mapping = np.clip(cv2.resize(y_mapping, (0,0), fx=4, fy=4, interpolation=cv2.INTER_LINEAR), 0, shape[0]-1)
        y_mapping = np.round(y_mapping).astype(np.int)
        ystart = np.round(np.amin(y_mapping)).astype(np.int)
        ystop = np.round(np.amax(y_mapping)).astype(np.int) + 1

        x_mapping = forward_mapping[:,:,0]
        x_mapping = np.clip(cv2.resize(x_mapping, (0,0), fx=4, fy=4, interpolation=cv2.INTER_LINEAR), 0, shape[1]-1)
        x_mapping = np.round(x_mapping).astype(np.int)
        xstart = np.round(np.amin(x_mapping)).astype(np.int)
        xstop = np.round(np.amax(x_mapping)).astype(np.int) + 1

        y_map = np.tile(np.arange(0, forward_mapping.shape[0]), (forward_mapping.shape[1], 1)).T.astype(np.float32)
        y_map = cv2.resize(y_map, (0,0), fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
        x_map = np.tile(np.arange(0, forward_mapping.shape[1]), (forward_mapping.shape[0], 1)).astype(np.float32)
        x_map = cv2.resize(x_map, (0,0), fx=4, fy=4, interpolation=cv2.INTER_LINEAR)

        reverse_mapping = np.ones((ystop-ystart, xstop-xstart, 2), dtype=np.float32) * -1

        for sx, sy, dx, dy in zip(x_map.flatten(), y_map.flatten(), x_mapping.flatten(), y_mapping.flatten()):
            # print(dy, ystart, dx, xstart)
            reverse_mapping[dy-ystart, dx-xstart, 0] = sx
            reverse_mapping[dy-ystart, dx-xstart, 1] = sy

        return reverse_mapping, (ystart, xstart)

    def get_blend_mask(self, mapping):
        mask = mapping[:,:,0] > -1
        mask = np.pad(mask, ((self.blend_border,self.blend_border), (self.blend_border,self.blend_border)))
        mask = ndimage.uniform_filter(mask.astype(np.float), size=2*self.blend_border+1)
        mask = mask[self.blend_border:-self.blend_border, self.blend_border:-self.blend_border]
        mask = 2 * np.clip(mask-0.5, 0, 1)
        return mask[:, :, np.newaxis]


def main():
    from pero_ocr.document_ocr import layout
    import matplotlib.pyplot as plt

    page_img = cv2.imread('../../../example/82f4ac84-6f1e-43ba-b1d5-e2b28d69508d.jpg')
    page_layout = layout.PageLayout(file='../../../example/82f4ac84-6f1e-43ba-b1d5-e2b28d69508d.xml')

    cropper = EngineLineCropper(line_height=48, poly=2, scale=1)
    cropped_line, mapping = cropper.crop(page_img, page_layout.regions[2].lines[0].baseline, page_layout.regions[2].lines[0].heights, return_mapping=True)
    back_mapped = cropper.blend_in(page_img, cropped_line, mapping)

    plt.subplot(131)
    plt.imshow(cropped_line)
    plt.subplot(132)
    plt.imshow(back_mapped[410:420, 1200:1240])
    plt.subplot(133)
    plt.imshow(np.concatenate((mapping, mapping[:,:,:1]), axis=2).astype(np.int)[410:420, 1200:1240])
    plt.show()


if __name__ == '__main__':
    main()
