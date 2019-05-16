"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import numpy as np
import glob
import cv2
import os

from utils import *
from datasets.plane_dataset import *

class InferenceDataset(Dataset):
    """ This class creates a dataloader for custom images """

    def __init__(self, options, config, image_list, camera, random=False):
        """ camera: [fx, fy, cx, cy, image_width, image_height, dummy, dummy, dummy, dummy] """
        
        self.options = options
        self.config = config
        self.random = random
        self.camera = camera
        self.imagePaths = image_list
        self.anchors = generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                      config.RPN_ANCHOR_RATIOS,
                                                      config.BACKBONE_SHAPES,
                                                      config.BACKBONE_STRIDES,
                                                      config.RPN_ANCHOR_STRIDE)
        return

    def __getitem__(self, index):
        t = int(time.time() * 1000000)
        np.random.seed(((t & 0xff000000) >> 24) +
                       ((t & 0x00ff0000) >> 8) +
                       ((t & 0x0000ff00) << 8) +
                       ((t & 0x000000ff) << 24))
        if self.random:
            index = np.random.randint(len(self.imagePaths))
        else:
            index = index % len(self.imagePaths)
            pass

        imagePath = self.imagePaths[index]
        image = cv2.imread(imagePath)
        extrinsics = np.eye(4, dtype=np.float32)

        if isinstance(self.camera, list):
            if isinstance(self.camera[index], str):
                camera = np.zeros(6)
                with open(self.camera[index], 'r') as f:
                    for line in f:
                        values = [float(token.strip()) for token in line.split(' ') if token.strip() != '']
                        for c in range(6):
                            camera[c] = values[c]
                            continue
                        break
                    pass
            else:
                camera = self.camera[index]
                pass
        elif len(self.camera) == 6:
            camera = self.camera
        else:
            assert(False)
            pass

        image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)
        camera[[0, 2, 4]] *= 640.0 / camera[4]        
        camera[[1, 3, 5]] *= 480.0 / camera[5]

        ## The below codes just fill in dummy values for all other data entries which are not used for inference. You can ignore everything except some preprocessing operations on "image".
        depth = np.zeros((self.config.IMAGE_MIN_DIM, self.config.IMAGE_MAX_DIM), dtype=np.float32)
        segmentation = np.zeros((self.config.IMAGE_MIN_DIM, self.config.IMAGE_MAX_DIM), dtype=np.int32)


        planes = np.zeros((segmentation.max() + 1, 3))

        instance_masks = []
        class_ids = []
        parameters = []

        if len(planes) > 0:
            if 'joint' in self.config.ANCHOR_TYPE:
                distances = np.linalg.norm(np.expand_dims(planes, 1) - self.config.ANCHOR_PLANES, axis=-1)
                plane_anchors = distances.argmin(-1)
            elif self.config.ANCHOR_TYPE == 'Nd':
                plane_offsets = np.linalg.norm(planes, axis=-1)
                plane_normals = planes / np.expand_dims(plane_offsets, axis=-1)
                distances_N = np.linalg.norm(np.expand_dims(plane_normals, 1) - self.config.ANCHOR_NORMALS, axis=-1)
                normal_anchors = distances_N.argmin(-1)
                distances_d = np.abs(np.expand_dims(plane_offsets, -1) - self.config.ANCHOR_OFFSETS)
                offset_anchors = distances_d.argmin(-1)
            elif self.config.ANCHOR_TYPE in ['normal', 'patch']:
                plane_offsets = np.linalg.norm(planes, axis=-1)
                plane_normals = planes / np.maximum(np.expand_dims(plane_offsets, axis=-1), 1e-4)
                distances_N = np.linalg.norm(np.expand_dims(plane_normals, 1) - self.config.ANCHOR_NORMALS, axis=-1)
                normal_anchors = distances_N.argmin(-1)
            elif self.config.ANCHOR_TYPE == 'normal_none':
                plane_offsets = np.linalg.norm(planes, axis=-1)
                plane_normals = planes / np.expand_dims(plane_offsets, axis=-1)
                pass
            pass

        for planeIndex, plane in enumerate(planes):
            m = segmentation == planeIndex
            if m.sum() < 1:
                continue
            instance_masks.append(m)
            if self.config.ANCHOR_TYPE == 'none':
                class_ids.append(1)
                parameters.append(np.concatenate([plane, np.zeros(1)], axis=0))
            elif 'joint' in self.config.ANCHOR_TYPE:
                class_ids.append(plane_anchors[planeIndex] + 1)
                residual = plane - self.config.ANCHOR_PLANES[plane_anchors[planeIndex]]
                parameters.append(np.concatenate([residual, np.array([0, plane_info[planeIndex][-1]])], axis=0))
            elif self.config.ANCHOR_TYPE == 'Nd':
                class_ids.append(normal_anchors[planeIndex] * len(self.config.ANCHOR_OFFSETS) + offset_anchors[planeIndex] + 1)
                normal = plane_normals[planeIndex] - self.config.ANCHOR_NORMALS[normal_anchors[planeIndex]]
                offset = plane_offsets[planeIndex] - self.config.ANCHOR_OFFSETS[offset_anchors[planeIndex]]
                parameters.append(np.concatenate([normal, np.array([offset])], axis=0))
            elif self.config.ANCHOR_TYPE == 'normal':
                class_ids.append(normal_anchors[planeIndex] + 1)
                normal = plane_normals[planeIndex] - self.config.ANCHOR_NORMALS[normal_anchors[planeIndex]]
                parameters.append(np.concatenate([normal, np.zeros(1)], axis=0))
            elif self.config.ANCHOR_TYPE == 'normal_none':
                class_ids.append(1)
                normal = plane_normals[planeIndex]
                parameters.append(np.concatenate([normal, np.zeros(1)], axis=0))
            else:
                assert(False)
                pass
            continue

        parameters = np.array(parameters)
        mask = np.stack(instance_masks, axis=2)
        class_ids = np.array(class_ids, dtype=np.int32)

        image, image_metas, gt_class_ids, gt_boxes, gt_masks, gt_parameters = load_image_gt(self.config, index, image, depth, mask, class_ids, parameters, augment=False)
        ## RPN Targets
        rpn_match, rpn_bbox = build_rpn_targets(image.shape, self.anchors,
                                                gt_class_ids, gt_boxes, self.config)

        ## If more instances than fits in the array, sub-sample from them.
        if gt_boxes.shape[0] > self.config.MAX_GT_INSTANCES:
            ids = np.random.choice(
                np.arange(gt_boxes.shape[0]), self.config.MAX_GT_INSTANCES, replace=False)
            gt_class_ids = gt_class_ids[ids]
            gt_boxes = gt_boxes[ids]
            gt_masks = gt_masks[:, :, ids]
            gt_parameters = gt_parameters[ids]
            pass

        ## Add to batch
        rpn_match = rpn_match[:, np.newaxis]
        image = utils.mold_image(image.astype(np.float32), self.config)

        depth = np.concatenate([np.zeros((80, 640)), depth, np.zeros((80, 640))], axis=0).astype(np.float32)
        segmentation = np.concatenate([np.full((80, 640), fill_value=-1), segmentation, np.full((80, 640), fill_value=-1)], axis=0).astype(np.float32)

        data_pair = [image.transpose((2, 0, 1)).astype(np.float32), image_metas, rpn_match.astype(np.int32), rpn_bbox.astype(np.float32), gt_class_ids.astype(np.int32), gt_boxes.astype(np.float32), gt_masks.transpose((2, 0, 1)).astype(np.float32), gt_parameters[:, :-1].astype(np.float32), depth.astype(np.float32), extrinsics.astype(np.float32), planes.astype(np.float32), segmentation.astype(np.int64), gt_parameters[:, -1].astype(np.int32)]
        data_pair = data_pair + data_pair

        data_pair.append(np.zeros(7, np.float32))

        data_pair.append(planes)
        data_pair.append(planes)
        data_pair.append(np.zeros((len(planes), len(planes))))
        data_pair.append(camera.astype(np.float32))
        return data_pair

    def __len__(self):
        return len(self.imagePaths)
