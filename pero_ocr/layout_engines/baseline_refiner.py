import numpy as np
from scipy.ndimage import measurements
from scipy import interpolate
from sklearn import linear_model

from pero_ocr.layout_engines import layout_helpers as helpers


def refine_baseline(baseline, heights, detection_maps, downsample, crop_engine, detection_threshold=0.3):
    """
    Refines the input baseline using fitting 2nd order polynom to the baseline detection map.
    :param baseline: numpy array of input baseline coords
    :param heights: list of input line heights [ascender, descender]
    :param detection_maps: channel 0: ascender heights, channel 1: descender heights, channel 2: baseline detections,
    channel 3: baseline endpoints, channel 4: region detections
    :param downsample: ds factor
    :param crop_engine: EngineLineCropper() object
    :return: numpy array of refined baseline coords
    """

    try:  # multiple steps can fail unpredictably due to cropper or failed polynom fitting
        baseline = baseline.copy() / downsample
        tolerance = (heights[0] + heights[1]) / (2 * downsample)

        line_crop, line_mapping = crop_engine.crop(
                detection_maps[:, :, 2:3], baseline, [tolerance, tolerance], return_forward_mapping=True)
        line_crop[line_crop < detection_threshold] = 0
        indices = np.where(line_crop)

        bs_pos_in_line = int(np.round(line_crop.shape[0] * heights[0]/(heights[0] + heights[1])))
        weights_above = np.linspace(0, 1.0, bs_pos_in_line)
        weights_below = np.linspace(1.0, 0, line_crop.shape[0] - bs_pos_in_line)
        positional_weights = np.tile(np.concatenate((weights_above, weights_below))[:, np.newaxis], (1, line_crop.shape[1]))

        weights = (line_crop * positional_weights)[indices[0], indices[1]]
        line_interpf = np.poly1d(np.polyfit(indices[1], indices[0], 3, w=weights))

        line_x_indices = np.arange(0, line_crop.shape[1])
        line_y_indices = np.round(np.clip(line_interpf(line_x_indices), 0, line_crop.shape[0]-1)).astype(int)
        line_x_indices = np.round(line_x_indices)

        line_values = line_crop[line_y_indices, line_x_indices]
        line_x_indices = np.delete(line_x_indices, np.where(line_values < detection_threshold))

        min_x = np.maximum(np.amin(line_x_indices)-10, 0)
        max_x = np.minimum(np.amax(line_x_indices)+10, line_crop.shape[1]-1)

        line_length = line_mapping[bs_pos_in_line, np.clip(max_x, 0, line_mapping.shape[1]-1), 0] - line_mapping[bs_pos_in_line, np.clip(min_x, 0, line_mapping.shape[1]-1), 0]
        num_steps = np.minimum(
            10,
            int(np.round(np.maximum(
                2,
                line_length/(tolerance * 2)
            ))))

        new_x_indices = np.linspace(min_x, max_x, num_steps)
        new_y_indices = np.round(line_interpf(new_x_indices)).astype(int)
        new_x_indices = np.round(new_x_indices).astype(int)

        new_y_indices = np.clip(new_y_indices, 0, line_mapping.shape[0] - 1)
        new_x_indices = np.clip(new_x_indices, 0, line_mapping.shape[1] - 1)

        new_baseline_x = line_mapping[new_y_indices, new_x_indices, 0]
        new_baseline_y = line_mapping[new_y_indices, new_x_indices, 1]
        return np.stack([new_baseline_x, new_baseline_y], axis=1) * downsample

    except:
        print(f'Baseline refinement failed for baseline {baseline * downsample}')
        return baseline * downsample


def refine_baseline_linear_regression(baseline, heights, detection_maps, downsample, crop_engine, detection_threshold=0.3):
    baseline = baseline.copy() / downsample
    tolerance = (heights[0] + heights[1]) / (2 * downsample)

    line_crop, line_mapping = crop_engine.crop(
        detection_maps[:, :, 2:3], baseline, [tolerance, tolerance], return_forward_mapping=True)
    line_crop[line_crop < detection_threshold] = 0
    line_crop[line_crop > 0] = 1

    ys, xs = np.where(line_crop == 1)
    ransac = linear_model.RANSACRegressor(max_trials=1000)

    try:
        ransac.fit(xs.reshape(-1, 1), ys)
    except ValueError:
        print(f'Baseline refinement with linear regression failed for baseline {baseline * downsample}: not enough points for RANSAC')
        return baseline * downsample

    num_steps = 10
    xs = np.linspace(0, line_crop.shape[1] - 1, num_steps)
    ys = ransac.predict(xs.reshape(-1, 1))

    xs = np.round(xs).astype(int)
    ys = np.round(ys).astype(int)

    mask = np.bitwise_and(0 < ys, ys < line_crop.shape[0])
    xs = xs[mask]
    ys = ys[mask]

    if len(xs) < 2:
        print(f'Baseline refinement with linear regression failed for baseline {baseline * downsample}: not enough points for baseline')
        return baseline * downsample

    new_baseline_x = line_mapping[ys, xs, 0]
    new_baseline_y = line_mapping[ys, xs, 1]
    return np.stack([new_baseline_x, new_baseline_y], axis=1) * downsample
