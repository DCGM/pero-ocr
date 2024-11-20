#!/usr/bin/env python3

from enum import Enum
import itertools

from configparser import SectionProxy
import torch
import cv2
import numpy as np

from pero_ocr.core.layout import PageLayout


class QueryType(Enum):
    PADDING = 0
    BBOX = 1
    CLASS_QUERY = 2


def parse_regions(
    layout: PageLayout,
    max_len: int,
) -> np.ndarray:
    region_bboxes = []

    for region in layout.regions:
        polygon = region.polygon
        l, t = np.min(polygon, axis=0)
        r, b = np.max(polygon, axis=0)
        region_bboxes.append([l, t, r, b])

    region_bboxes = np.array(region_bboxes, dtype=np.float64)

    # Normalize to [0, 1]
    region_bboxes[:, 0] /= layout.page_size[1]
    region_bboxes[:, 1] /= layout.page_size[0]
    region_bboxes[:, 2] /= layout.page_size[1]
    region_bboxes[:, 3] /= layout.page_size[0]

    # Trim if larger than max
    if region_bboxes.shape[0] > max_len:
        region_bboxes = region_bboxes[:max_len]

    region_bboxes = region_bboxes.astype(np.float32)
    region_bboxes = np.clip(region_bboxes, a_min=0.0, a_max=1.0)

    return region_bboxes


def page_layout_to_model_inputs(
    page_layout: PageLayout,
    max_bbox_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    region_bboxes = parse_regions(
        layout=page_layout,
        max_len=max_bbox_count,
    )

    query_types = [QueryType.BBOX.value] * len(region_bboxes) + [QueryType.PADDING.value] * (max_bbox_count - len(region_bboxes))

    region_bboxes = np.pad(
        region_bboxes,
        ((0, max_bbox_count - region_bboxes.shape[0]), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    return region_bboxes, np.array(query_types)


def find_shortest_path(cost_mat: np.ndarray, max_nodes: int = 8):
    merged_nodes = [([i], 0) for i in range(cost_mat.shape[0])]
    np.fill_diagonal(cost_mat, 9)

    while len(merged_nodes) > max_nodes:
        best_cost = 1e9
        best_merge_from = None
        best_merge_to = None

        for idx in range(len(merged_nodes)):
            end_cost = cost_mat[idx, :].min()
            if end_cost < best_cost:
                best_cost = end_cost
                to = np.argmin(cost_mat[idx, :])
                best_merge = (idx, to)

        from_idx, to_idx = best_merge
        from_nodes, from_cost = merged_nodes[from_idx]
        to_nodes, to_cost = merged_nodes[to_idx]
        new_nodes = from_nodes + to_nodes
        new_cost = from_cost + to_cost + cost_mat[from_idx, to_idx]
        merged_nodes.append((new_nodes, new_cost))
        merged_nodes = [x for i, x in enumerate(merged_nodes) if i != from_idx and i != to_idx]

        n = cost_mat.shape[0]
        new_cost_mat = np.zeros((n + 1, n + 1))
        new_cost_mat[:-1, :-1] = cost_mat
        new_cost_mat[-1, :-1] = cost_mat[to_idx, :]
        new_cost_mat[:-1, -1] = cost_mat[:, from_idx]
        new_cost_mat[-1, -1] = 9

        new_cost_mat = np.delete(new_cost_mat, [from_idx, to_idx], axis=0)
        new_cost_mat = np.delete(new_cost_mat, [from_idx, to_idx], axis=1)
        cost_mat = new_cost_mat

    best_cost = 1e9
    best_order = None
    for perm in itertools.permutations(range(len(merged_nodes))):
        cost = 0
        for i in range(len(perm)-1):
            cost += cost_mat[perm[i], perm[i+1]]
        if cost < best_cost:
            best_cost = cost
            best_order = perm

    final_order = [merged_nodes[i][0] for i in best_order]
    final_order = list(itertools.chain(*final_order))

    return final_order


class TransformerRegionSorter:
    def __init__(
        self,
        config: SectionProxy,
        config_path="",
    ):
        self.model = torch.jit.load(config["MODEL_PATH"])
        self.model.eval()
        self.max_bbox_count = int(config["MAX_BBOX_COUNT"])
        self.image_width = int(config["IMAGE_WIDTH"])
        self.image_height = int(config["IMAGE_HEIGHT"])

    def process_page(
        self,
        image,
        page_layout: PageLayout,
    ) -> PageLayout:
        if len(page_layout.regions) < 2:
            return page_layout

        new_regions = []
        
        region_bboxes, query_types = page_layout_to_model_inputs(
            page_layout=page_layout,
            max_bbox_count=self.max_bbox_count,
        )
        region_bboxes = torch.tensor(region_bboxes, dtype=torch.float32, device="cuda").unsqueeze(0)
        query_types = torch.tensor(query_types, dtype=torch.int64, device="cuda").unsqueeze(0)

        image_resized = cv2.resize(image, (self.image_height, self.image_width))
        image_tensor = torch.tensor(image_resized, dtype=torch.uint8, device="cuda").permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(
                region_bboxes,
                query_types,
                image_tensor,
            ).squeeze()

            valid_count = (query_types == QueryType.BBOX.value).sum().item()
            prob = torch.nn.functional.softmax(outputs[:valid_count, :valid_count], dim=0)

        order = find_shortest_path(1-prob.cpu().numpy().T)
        new_regions = [page_layout.regions[i] for i in order]
        remainder = set(page_layout.regions) - set(new_regions)
        new_regions += list(remainder)
        page_layout.regions = new_regions
        return page_layout
