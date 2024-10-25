# Copyright 2024 wlli
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
# import torch
# from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

from typing import Any, Dict, List, Optional, Tuple
from itertools import product
import math

import sys
sys.path.append("/home/wlli/project_in_wd/deepYeast/deeplab/")
from postprocess.post_process_utils import post_process_panoptic


def generate_crop_boxes(
    im_size: Tuple[int, ...], crop_size: int, overlap_ratio: float
) -> Tuple[List[List[int]], List[int]]:
    """
    Generates a list of crop boxes of different sizes. Each layer
    has (2**i)**2 boxes for the ith layer.
    """
    crop_boxes = []
    im_h, im_w = im_size
    def slide_step(orig_len, crop_len, overlap):
        start = 0
        end = orig_len - crop_len
        step = crop_len * (1 - overlap)
        crop_box0 = [int(x) for x in np.arange(start, end, step)]
        crop_box0.append(end)
        return crop_box0

    crop_box_x0 = slide_step(im_w, crop_size, overlap_ratio)
    crop_box_y0 = slide_step(im_h, crop_size, overlap_ratio)

    # Crops in XYWH format
    for x0, y0 in product(crop_box_x0, crop_box_y0):
        box = [x0, y0, min(x0 + crop_size, im_w), min(y0 + crop_size, im_h)]
        crop_boxes.append(box)
    return crop_boxes


class SlideWindowPredictor:
    def __init__(self, model, crop_size=513, overlap_ratio=0.25, area_threshold=200) -> None:
        self.model = model
        self.crop_size = crop_size
        self.overlap_ratio = overlap_ratio
        self.area_threshold = area_threshold

    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: List[int],
    ):
        # Crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_box
        cropped_im = image[x0:x1, y0:y1]
        # cropped_im_size = cropped_im.shape[:2]
        prediction = self.model.predict(cropped_im)
        prediction = prediction["panoptic_pred"][0].numpy()
        post_ouput = post_process_panoptic(prediction,
                                           expand_disk=7,
                                           clean_border_region=True,
                                           area_threshold=self.area_threshold)
        return post_ouput

    def generate_masks(self, image: np.ndarray):
        orig_size = image.shape[:2]
        crop_boxes = generate_crop_boxes(orig_size, self.crop_size, self.overlap_ratio)

        # Iterate over image crops
        data = []
        for crop_box in crop_boxes:
            crop_data = self._process_crop(image, crop_box)
            data.append(crop_data)
        orig_size_mask = self.contact_cropbox(image, data, crop_boxes)
        return orig_size_mask

    def contact_cropbox(self, image, mask_list, crop_boxes):
        existed_max_id = 1
        orig_size_mask = np.zeros(image.shape, dtype=np.uint16)
        for mask, crop_box in zip(mask_list, crop_boxes):
            x0, y0, x1, y1 = crop_box
            overlap_mask = (orig_size_mask[x0:x1, y0:y1] != 0)
            overlap_id = np.unique(mask[overlap_mask])
            overlap_id = overlap_id[overlap_id != 0]
            for id in overlap_id:
                mask[mask == id] = 0
            new_mask, existed_max_id = reset_value(mask, existed_max_id)
            orig_size_mask[x0:x1, y0:y1] += new_mask
        return orig_size_mask


def reset_value(old_array, existed_max_id=1):
    unique_id = np.unique(old_array)[1:]
    sem_label = unique_id // 1000
    mapping = {0: 0}
    for old_id, sem_id in zip(unique_id, sem_label):
        new_id = existed_max_id + sem_id*1000
        mapping[old_id] = new_id
        existed_max_id += 1
    map_function = np.vectorize(mapping.get)
    new_array = map_function(old_array)
    return new_array.astype(np.uint16), existed_max_id
