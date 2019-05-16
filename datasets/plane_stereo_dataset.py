"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
from torch.utils.data import Dataset

import numpy as np
import time
import utils as utils
import os
import cv2

from datasets.scannet_scene import ScanNetScene
from datasets.plane_dataset import *

class PlaneDataset(PlaneDatasetSingle):
    def __init__(self, options, config, split, random=True, image_only=False, load_semantics=False, load_boundary=False, write_invalid_indices=False):
        super().__init__(options, config, split, random, load_semantics=load_semantics, load_boundary=load_boundary)
        self.image_only = image_only
        self.load_semantics = load_semantics
        self.load_boundary = load_boundary
        self.write_invalid_indices = write_invalid_indices
        return

    def __getitem__(self, index):
        t = int(time.time() * 1000000)
        np.random.seed(((t & 0xff000000) >> 24) +
                       ((t & 0x00ff0000) >> 8) +
                       ((t & 0x0000ff00) << 8) +
                       ((t & 0x000000ff) << 24))
        if self.random:
            index = np.random.randint(len(self.sceneImageIndices))
        else:
            index = index % len(self.sceneImageIndices)
            if self.options.testingIndex >= 0 and index != self.options.testingIndex:
                return 0
            pass

        sceneIndex, imageIndex = self.sceneImageIndices[index]
        scene = self.scenes[sceneIndex]

        while True:
            if self.random:
                index = np.random.randint(len(self.sceneImageIndices))
            else:
                index = (index + 1) % len(self.sceneImageIndices)
                pass
        
            sceneIndex, imageIndex = self.sceneImageIndices[index]
            scene = self.scenes[sceneIndex]
            
            if imageIndex + self.options.frameGap < len(scene.imagePaths):
                imageIndex_2 = imageIndex + self.options.frameGap
            else:
                imageIndex_2 = imageIndex - self.options.frameGap
                pass

            if (sceneIndex * 10000 + imageIndex_2) in self.invalid_indices:
                continue
            
            try:
                image_1, planes_1, plane_info_1, segmentation_1, depth_1, camera_1, extrinsics_1, semantics_1 = scene[imageIndex]                
            except:                
                if self.write_invalid_indices:
                    print('invalid')
                    print(str(index) + ' ' + str(sceneIndex) + ' ' + str(imageIndex) + '\n', file=open(self.dataFolder + '/invalid_indices_' + self.split + '.txt', 'a'))
                    return 1
                continue

            if self.write_invalid_indices:
                return 0
            
            info_1 = [image_1, planes_1, plane_info_1, segmentation_1, depth_1, camera_1, extrinsics_1, semantics_1]
            
            try:
                image_2, planes_2, plane_info_2, segmentation_2, depth_2, camera_2, extrinsics_2, semantics_2 = scene[imageIndex_2]
            except:
                continue
            
            info_2 = [image_2, planes_2, plane_info_2, segmentation_2, depth_2, camera_2, extrinsics_2, semantics_2]
            break
        if self.image_only:
            data_pair = []
            for info in [info_1, info_2]:
                image, planes, plane_info, segmentation, depth, camera, extrinsics, semantics = info
                image = cv2.resize(image, (depth.shape[1], depth.shape[0]))
                image, window, scale, padding = utils.resize_image(
                    image,
                    min_dim=self.config.IMAGE_MAX_DIM,
                    max_dim=self.config.IMAGE_MAX_DIM,
                    padding=self.config.IMAGE_PADDING)
                
                image = utils.mold_image(image.astype(np.float32), self.config)                
                image = torch.from_numpy(image.transpose(2, 0, 1)).float()
                depth = np.concatenate([np.zeros((80, 640)), depth, np.zeros((80, 640))], axis=0)
                data_pair += [image, depth.astype(np.float32), camera]
                continue
            return data_pair
        data_pair = []
        extrinsics_pair = []
        for info in [info_1, info_2]:
            image, planes, plane_info, segmentation, depth, camera, extrinsics, semantics = info

            image = cv2.resize(image, (depth.shape[1], depth.shape[0]))
            
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
                elif 'normal' in self.config.ANCHOR_TYPE or self.config.ANCHOR_TYPE == 'patch':
                    plane_offsets = np.linalg.norm(planes, axis=-1)                    
                    plane_normals = planes / np.expand_dims(plane_offsets, axis=-1)
                    distances_N = np.linalg.norm(np.expand_dims(plane_normals, 1) - self.config.ANCHOR_NORMALS, axis=-1)
                    normal_anchors = distances_N.argmin(-1)
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
                elif 'normal' in self.config.ANCHOR_TYPE:
                    class_ids.append(normal_anchors[planeIndex] + 1)
                    normal = plane_normals[planeIndex] - self.config.ANCHOR_NORMALS[normal_anchors[planeIndex]]
                    parameters.append(np.concatenate([normal, np.array([plane_info[planeIndex][-1]])], axis=0))
                else:
                    assert(False)
                    pass
                continue

            parameters = np.array(parameters)
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)

                        
            image, image_metas, gt_class_ids, gt_boxes, gt_masks, gt_parameters = load_image_gt(self.config, index, image, depth, mask, class_ids, parameters, augment=self.split == 'train')
            
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

            depth = np.concatenate([np.zeros((80, 640)), depth, np.zeros((80, 640))], axis=0)
            segmentation = np.concatenate([np.full((80, 640), fill_value=-1, dtype=np.int32), segmentation, np.full((80, 640), fill_value=-1, dtype=np.int32)], axis=0)

            ## Convert
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            image_metas = torch.from_numpy(image_metas)
            rpn_match = torch.from_numpy(rpn_match)
            rpn_bbox = torch.from_numpy(rpn_bbox).float()
            gt_class_ids = torch.from_numpy(gt_class_ids)
            gt_boxes = torch.from_numpy(gt_boxes).float()
            gt_masks = torch.from_numpy(gt_masks.astype(np.float32)).transpose(1, 2).transpose(0, 1)
            plane_indices = torch.from_numpy(gt_parameters[:, -1]).long()
            gt_parameters = torch.from_numpy(gt_parameters[:, :-1]).float()
            data_pair += [image, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, gt_parameters, depth.astype(np.float32), extrinsics.astype(np.float32), planes.astype(np.float32), segmentation, plane_indices]

            if self.load_semantics or self.load_boundary:
                semantics = np.concatenate([np.full((80, 640), fill_value=-1, dtype=np.int32), semantics, np.full((80, 640), fill_value=-1, dtype=np.int32)], axis=0)                
                data_pair[-1] = semantics
                pass
            
            extrinsics_pair.append(extrinsics)
            continue

        transformation = np.matmul(extrinsics_pair[1], np.linalg.inv(extrinsics_pair[0]))
        rotation = transformation[:3, :3]
        translation = transformation[:3, 3]
        axis, angle = utils.rotationMatrixToAxisAngle(rotation)

        data_pair.append(np.concatenate([translation, axis, np.array([angle])], axis=0).astype(np.float32))

        correspondence = np.zeros((len(info_1[1]), len(info_2[1])), dtype=np.float32)
        for planeIndex_1, plane_info_1 in enumerate(info_1[2]):
            for planeIndex_2, plane_info_2 in enumerate(info_2[2]):
                if plane_info_1[-1] == plane_info_2[-1]:
                    correspondence[planeIndex_1][planeIndex_2] = 1
                    pass
                continue
            continue
        data_pair.append(info_1[1].astype(np.float32))
        data_pair.append(info_2[1].astype(np.float32))        
        data_pair.append(correspondence)
        data_pair.append(camera.astype(np.float32))

        return data_pair

