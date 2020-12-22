import numpy as np

from pero_ocr.layout_engines import layout_helpers as helpers


class PostprocessingEngine(object):
    def __init__(self, stretch_lines, resample_lines, heights_from_regions):
        self.stretch_lines = stretch_lines
        self.resample_lines = resample_lines
        self.heights_from_regions = heights_from_regions

    def postprocess(self, region):
        if region.lines:
            redo_textlines = False
            if self.stretch_lines == 'max' or self.stretch_lines > 0:
                self.stretch_baselines(region)
                redo_textlines = True
            if self.resample_lines:
                self.resample_baselines(region)
                redo_textlines = True
            if self.heights_from_regions:
                self.get_heights_from_regions(region)
                redo_textlines = True

            if redo_textlines:
                for line in region.lines:
                    line.polygon = helpers.baseline_to_textline(
                        line.baseline, line.heights)

        return region

    def stretch_baselines(self, region):
        baselines = [line.baseline for line in region.lines]
        rotation = helpers.get_rotation(baselines)
        baselines = [helpers.rotate_coords(baseline, rotation, (0, 0)) for baseline in baselines]

        if self.stretch_lines == 'max':
            region_poly = helpers.rotate_coords(region.polygon, rotation, (0, 0))
            baselines_stretched = []
            region_poly = np.concatenate((region_poly, region_poly[:1, :]), axis=0)
            for baseline in baselines:
                line_interpf = np.poly1d(np.polyfit(baseline[:, 0], baseline[:, 1], 1))
                y_1 = line_interpf(np.amin(region[:, 0]))
                y_2 = line_interpf(np.amax(region[:, 0]))
                baseline_ls = sg.LineString(
                    [(np.amin(region[:, 0]), y_1), (np.amax(region[:, 0]), y_2)])
                region_ls = sg.LineString(region)

                intersections_ls = region_ls.intersection(baseline_ls)
                #intersection can be empty due to borderline baselines and integer coordinate rotations
                if isinstance(intersections_ls, sg.MultiPoint):
                    intersections = np.squeeze(np.asarray(
                        [intersection.coords.xy for intersection in intersections_ls]))
                    intersection_left = intersections[np.argmin(intersections[:, 0]), :]
                    intersection_right = intersections[np.argmax(intersections[:, 0]), :]

                    baselines_stretched.append(np.concatenate(
                        (intersection_left[np.newaxis, :], baseline, intersection_right[np.newaxis, :]), axis=0))

        elif self.stretch_lines > 0:
            baselines_stretched = []
            for baseline in baselines:
                last_point = baseline[-1:, :].copy()
                last_point[0, 0] += self.stretch_lines
                first_point = baseline[:1, :].copy()
                first_point[0, 0] -= self.stretch_lines
                baselines_stretched.append(np.concatenate((first_point, baseline, last_point), axis=0))

        baselines_stretched = [helpers.rotate_coords(baseline, -rotation, (0, 0)) for baseline in baselines_stretched]
        for line, baseline in zip(region.lines, baselines_stretched):
            line.baseline = baseline

    def resample_baselines(self, region):
        baselines = [line.baseline for line in region.lines]
        baselines_resampled = helpers.resample_baselines(baselines)
        for line, baseline in zip(region.lines, baselines_resampled):
            line.baseline = baseline

    def get_heights_from_regions(self, region):
        """Computes line heights from regions and discards all but the biggest
        line the region
        """
        scores = []
        r_h_list = []
        for line in region.lines:
            height_asc = int(round(
                np.amin(line.baseline[:, 1]) - np.amin(region.polygon[:, 1])))
            height_des = int(round(
                np.amax(region.polygon[:, 1]) - np.amax(line.baseline[:, 1])))
            r_h_list.append((height_asc, height_des))
            # the final line in the bounding box should be the longest and in case of ambiguity, also have the biggest ascender height
            scores.append(
                np.amax(baseline[:, 0]) - np.amin(baseline[:, 0]) + height_asc)
        best_ind = np.argmax(np.asarray(scores))
        region.lines = [region.lines[best_ind]]
        region.lines[0].heights = r_h_list[best_ind]
