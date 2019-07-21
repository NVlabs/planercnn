import numpy as np
import glob
import cv2
import os

from utils import *
from datasets.plane_dataset import *
import h5py
import scipy.io as sio

## This class handle one scene of the scannet dataset and provide interface for dataloaders
class NYUDataset(Dataset):
    def __init__(self, options, config, split, random=True):
        self.options = options
        self.config = config
        self.random = random
        
        #'../../Data/NYU_RGBD/' + split + '/'
        dataPath = options.dataFolder
        self.camera = np.array([5.8262448167737955e+02, 5.8269103270988637e+02, 3.1304475870804731e+02, 2.3844389626620386e+02, 640, 480], dtype=np.float32)
        split = sio.loadmat(dataPath + '/splits.mat')
        indices = split['testNdxs'].reshape(-1) - 1
        data = h5py.File(dataPath + '/nyu_depth_v2_labeled.mat')
        self.images = np.array(data['images'])
        self.depths = np.array(data['depths'])
        self.images = self.images[indices]
        self.depths = self.depths[indices]
        
        self.anchors = generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                      config.RPN_ANCHOR_RATIOS,
                                                      config.BACKBONE_SHAPES,
                                                      config.BACKBONE_STRIDES,
                                                      config.RPN_ANCHOR_STRIDE)
        self.load_ori = True
        return

    def __getitem__(self, index):
        t = int(time.time() * 1000000)
        np.random.seed(((t & 0xff000000) >> 24) +
                       ((t & 0x00ff0000) >> 8) +
                       ((t & 0x0000ff00) << 8) +
                       ((t & 0x000000ff) << 24))
        if self.random:
            index = np.random.randint(len(self.images))
        else:
            index = index % len(self.images)
            pass

        image = self.images[index].transpose((2, 1, 0)).astype(np.uint8)[:, :, ::1]
        depth = self.depths[index].transpose((1, 0))
        camera = self.camera.copy()

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
        segmentation = np.full((640, 640), fill_value=0)
        
        extrinsics = np.eye(4, dtype=np.float32)

        data_pair = [image.transpose((2, 0, 1)).astype(np.float32), image_metas, rpn_match.astype(np.int32), rpn_bbox.astype(np.float32), gt_class_ids.astype(np.int32), gt_boxes.astype(np.float32), gt_masks.transpose((2, 0, 1)).astype(np.float32), gt_parameters[:, :-1].astype(np.float32), depth.astype(np.float32), extrinsics.astype(np.float32), planes.astype(np.float32), segmentation.astype(np.int64), gt_parameters[:, -1].astype(np.int32)]
        data_pair = data_pair + data_pair

        data_pair.append(np.zeros(7, np.float32))

        data_pair.append(planes)
        data_pair.append(planes)        
        data_pair.append(np.zeros((len(planes), len(planes))))
        data_pair.append(camera)        
        return data_pair

    def __len__(self):
        return len(self.images)
