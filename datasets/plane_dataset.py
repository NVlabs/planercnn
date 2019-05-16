"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
from torch.utils.data import Dataset

import numpy as np
import time
import os
import cv2
import sys
import utils
from datasets.scannet_scene import ScanNetScene

class PlaneDatasetSingle(Dataset):
    def __init__(self, options, config, split, random=True, loadNeighborImage=False, load_semantics=False, load_boundary=False):
        self.options = options
        self.config = config
        self.split = split
        self.random = random
        
        self.dataFolder = options.dataFolder
        
        self.scenes = []
        self.sceneImageIndices = []

        self.loadClassMap()
        
        planenet_scene_ids_val = np.load('datasets/scene_ids_val.npy')
        planenet_scene_ids_val = {scene_id.decode('utf-8'): True for scene_id in planenet_scene_ids_val}
        with open(self.dataFolder + '/ScanNet/Tasks/Benchmark/scannetv1_' + split + '.txt') as f:
            for line in f:
                scene_id = line.strip()
                if split == 'test':
                    ## Remove scenes which are in PlaneNet's training set for fair comparison
                    if scene_id not in planenet_scene_ids_val:
                        continue
                    pass
                scenePath = self.dataFolder + '/scans/' + scene_id
                if not os.path.exists(scenePath + '/' + scene_id + '.txt') or not os.path.exists(scenePath + '/annotation/planes.npy'):
                    continue
                scene = ScanNetScene(options, scenePath, scene_id, self.confident_labels, self.layout_labels, load_semantics=load_semantics, load_boundary=load_boundary)
                self.scenes.append(scene)
                self.sceneImageIndices += [[len(self.scenes) - 1, imageIndex] for imageIndex in range(len(scene.imagePaths))]
                continue
            pass
        
        if random:
            t = int(time.time() * 1000000)
            np.random.seed(((t & 0xff000000) >> 24) +
                           ((t & 0x00ff0000) >> 8) +
                           ((t & 0x0000ff00) << 8) +
                           ((t & 0x000000ff) << 24))
        else:
            np.random.seed(0)
            pass
        np.random.shuffle(self.sceneImageIndices)

        self.invalid_indices = {}

        with open(self.dataFolder + '/invalid_indices_' + split + '.txt', 'r') as f:
            for line in f:
                tokens = line.split(' ')
                if len(tokens) == 3:
                    assert(int(tokens[2]) < 10000)
                    invalid_index = int(tokens[1]) * 10000 + int(tokens[2])
                    if invalid_index not in self.invalid_indices:
                        self.invalid_indices[invalid_index] = True
                        pass
                    pass
                continue
            pass

        self.sceneImageIndices = [[sceneIndex, imageIndex] for sceneIndex, imageIndex in self.sceneImageIndices if (sceneIndex * 10000 + imageIndex) not in self.invalid_indices]

        print('num images', len(self.sceneImageIndices))
        
        self.anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                      config.RPN_ANCHOR_RATIOS,
                                                      config.BACKBONE_SHAPES,
                                                      config.BACKBONE_STRIDES,
                                                      config.RPN_ANCHOR_STRIDE)


        self.loadNeighborImage = loadNeighborImage

        return

    def loadClassMap(self):
        classLabelMap = {}
        with open(self.dataFolder + '/scannetv2-labels.combined.tsv') as info_file:
            line_index = 0
            for line in info_file:
                if line_index > 0:
                    line = line.split('\t')
                    key = line[1].strip()
                    
                    if line[4].strip() != '':
                        label = int(line[4].strip())
                    else:
                        label = -1
                        pass
                    classLabelMap[key] = label
                    classLabelMap[key + 's'] = label
                    classLabelMap[key + 'es'] = label                                        
                    pass
                line_index += 1
                continue
            pass

        confidentClasses = {'wall': True, 
                            'floor': True,
                            'cabinet': True,
                            'bed': True,
                            'chair': False,
                            'sofa': False,
                            'table': True,
                            'door': True,
                            'window': True,
                            'bookshelf': False,
                            'picture': True,
                            'counter': True,
                            'blinds': False,
                            'desk': True,
                            'shelf': False,
                            'shelves': False,
                            'curtain': False,
                            'dresser': True,
                            'pillow': False,
                            'mirror': False,
                            'entrance': True,
                            'floor mat': True,
                            'clothes': False,
                            'ceiling': True,
                            'book': False,
                            'books': False,                      
                            'refridgerator': True,
                            'television': True, 
                            'paper': False,
                            'towel': False,
                            'shower curtain': False,
                            'box': True,
                            'whiteboard': True,
                            'person': False,
                            'night stand': True,
                            'toilet': False,
                            'sink': False,
                            'lamp': False,
                            'bathtub': False,
                            'bag': False,
                            'otherprop': False,
                            'otherstructure': False,
                            'otherfurniture': False,
                            'unannotated': False,
                            '': False
        }

        self.confident_labels = {}
        for name, confidence in confidentClasses.items():
            if confidence and name in classLabelMap:
                self.confident_labels[classLabelMap[name]] = True
                pass
            continue
        self.layout_labels = {1: True, 2: True, 22: True, 9: True}
        return
    
    def __len__(self):
        return len(self.sceneImageIndices)

    def transformPlanes(self, transformation, planes):
        planeOffsets = np.linalg.norm(planes, axis=-1, keepdims=True)
        
        centers = planes
        centers = np.concatenate([centers, np.ones((planes.shape[0], 1))], axis=-1)
        newCenters = np.transpose(np.matmul(transformation, np.transpose(centers)))
        newCenters = newCenters[:, :3] / newCenters[:, 3:4]

        refPoints = planes - planes / np.maximum(planeOffsets, 1e-4)
        refPoints = np.concatenate([refPoints, np.ones((planes.shape[0], 1))], axis=-1)
        newRefPoints = np.transpose(np.matmul(transformation, np.transpose(refPoints)))
        newRefPoints = newRefPoints[:, :3] / newRefPoints[:, 3:4]

        planeNormals = newRefPoints - newCenters
        planeNormals /= np.linalg.norm(planeNormals, axis=-1, keepdims=True)
        planeOffsets = np.sum(newCenters * planeNormals, axis=-1, keepdims=True)
        newPlanes = planeNormals * planeOffsets
        return newPlanes
    
    def __getitem__(self, index):
        t = int(time.time() * 1000000)
        np.random.seed(((t & 0xff000000) >> 24) +
                       ((t & 0x00ff0000) >> 8) +
                       ((t & 0x0000ff00) << 8) +
                       ((t & 0x000000ff) << 24))

        if self.config.ANCHOR_TYPE == 'layout':
            return self.getItemLayout(index)
        
        if self.config.ANCHOR_TYPE == 'structure':
            return self.getItemStructure(index)        

        while True:
            if self.random:
                index = np.random.randint(len(self.sceneImageIndices))
            else:
                index = index % len(self.sceneImageIndices)
                pass

            sceneIndex, imageIndex = self.sceneImageIndices[index]

            scene = self.scenes[sceneIndex]

            try:
                image, planes, plane_info, segmentation, depth, camera, extrinsics = scene[imageIndex]
                if len(planes) == 0:
                    index += 1                    
                    continue
            except:
                index += 1
                continue
                pass
            
            if segmentation.max() < 0:
                index += 1
                continue
            break

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
                plane_normals = planes / np.expand_dims(plane_offsets, axis=-1)
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
                parameters.append(np.concatenate([residual, np.zeros(1)], axis=0))
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

        parameters = np.array(parameters, dtype=np.float32)
        mask = np.stack(instance_masks, axis=2)

        class_ids = np.array(class_ids, dtype=np.int32)
        image, image_metas, gt_class_ids, gt_boxes, gt_masks, gt_parameters = load_image_gt(self.config, index, image, mask, class_ids, parameters, augment=self.split == 'train')
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

        ## Add to batch
        rpn_match = rpn_match[:, np.newaxis]
        image = utils.mold_image(image.astype(np.float32), self.config)

        depth = np.concatenate([np.zeros((80, 640)), depth, np.zeros((80, 640))], axis=0)
        segmentation = np.concatenate([np.full((80, 640), fill_value=-1, dtype=np.int32), segmentation, np.full((80, 640), fill_value=-1, dtype=np.int32)], axis=0)        
        
        info = [image.transpose((2, 0, 1)).astype(np.float32), image_metas, rpn_match, rpn_bbox.astype(np.float32), gt_class_ids, gt_boxes.astype(np.float32), gt_masks.transpose((2, 0, 1)).astype(np.float32), gt_parameters, depth.astype(np.float32), segmentation, camera.astype(np.float32)]
        
        if self.loadNeighborImage:
            if imageIndex + self.options.frameGap < len(scene.imagePaths):
                imagePath = scene.imagePaths[imageIndex + self.options.frameGap]
            else:
                imagePath = scene.imagePaths[imageIndex - self.options.frameGap]
                pass

            image_2 = cv2.imread(imagePath)
            
            image_2 = cv2.resize(image_2, (self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM))
            
            info.append(image_2.transpose((2, 0, 1)).astype(np.float32))

            extrinsics_2_inv = []
            posePath = imagePath.replace('color', 'pose').replace('.jpg', '.txt')            
            with open(posePath, 'r') as f:
                for line in f:
                    extrinsics_2_inv += [float(value) for value in line.strip().split(' ') if value.strip() != '']
                    continue
                f.close()
                pass
            extrinsics_2_inv = np.array(extrinsics_2_inv).reshape((4, 4))
            extrinsics_2 = np.linalg.inv(extrinsics_2_inv)

            temp = extrinsics_2[1].copy()
            extrinsics_2[1] = extrinsics_2[2]
            extrinsics_2[2] = -temp
            
            transformation = np.matmul(extrinsics_2, np.linalg.inv(extrinsics))
            if np.any(np.isnan(transformation)):
                transformation = np.concatenate([np.diag(np.ones(3)), np.zeros((3, 1))], axis=-1)
                pass
        
            rotation = transformation[:3, :3]
            translation = transformation[:3, 3]
            axis, angle = utils.rotationMatrixToAxisAngle(rotation)
            pose = np.concatenate([translation, axis * angle], axis=0).astype(np.float32)
            info.append(pose)
            info.append(scene.scenePath + ' ' + str(imageIndex))
            pass
            
        return info
    
    
    def getAnchorPlanesNormalOffset(self, visualize=False):
        for k in [7, ]:
            print('k', k)
            filename_N = self.dataFolder + '/anchor_planes_N_' + str(k) + '.npy'
            filename_d = self.dataFolder + '/anchor_planes_d.npy'
            if os.path.exists(filename_N) and os.path.exists(filename_d) and False:
                return

            if os.path.exists('test/anchor_planes/all_planes.npy'):
                all_planes = np.load('test/anchor_planes/all_planes.npy')
            else:
                all_planes = []
                for sceneIndex, imageIndex in self.sceneImageIndices[:10000]:
                    if len(all_planes) % 100 == 0:
                        print(len(all_planes))
                        pass
                    scene = self.scenes[sceneIndex]

                    image, planes, plane_info, segmentation, depth, camera, extrinsics = scene[imageIndex]
                    planes = planes[np.linalg.norm(planes, axis=-1) > 1e-4]
                    if len(planes) == 0:
                        continue
                    all_planes.append(planes)
                    continue
                all_planes = np.concatenate(all_planes, axis=0)
                np.save('test/anchor_planes/all_planes.npy', all_planes)                
                pass

            from sklearn.cluster import KMeans

            num_anchor_planes_N = k
            num_anchor_planes_d = 3

            offsets = np.linalg.norm(all_planes, axis=-1)
            normals = all_planes / np.expand_dims(offsets, -1)

            kmeans_N = KMeans(n_clusters=num_anchor_planes_N).fit(normals)
            self.anchor_planes_N = kmeans_N.cluster_centers_

            ## Global offset anchors
            kmeans_d = KMeans(n_clusters=num_anchor_planes_d).fit(np.expand_dims(offsets, -1))
            self.anchor_planes_d = kmeans_d.cluster_centers_            

            if visualize:
                color_map = utils.ColorPalette(max(num_anchor_planes_N, num_anchor_planes_d)).getColorMap()
                normals_rotated = normals.copy()
                normals_rotated[:, 1] = normals[:, 2]
                normals_rotated[:, 2] = -normals[:, 1]
                plane_cloud = np.concatenate([normals_rotated, color_map[kmeans_N.labels_]], axis=-1)
                utils.writePointCloud('test/anchor_planes/anchor_planes_N.ply', plane_cloud)

                plane_cloud = np.concatenate([all_planes, color_map[kmeans_d.labels_]], axis=-1)
                utils.writePointCloud('test/anchor_planes/anchor_planes_d.ply', plane_cloud)

                width = 500
                height = 500

                Us = np.round(np.arctan2(normals[:, 1], normals[:, 0]) / np.pi * width).astype(np.int32)
                Vs = np.round((1 - (np.arcsin(normals[:, 2]) + np.pi / 2) / np.pi) * height).astype(np.int32)
                indices = Vs * width + Us
                validMask = np.logical_and(np.logical_and(Us >=  0, Us < width), np.logical_and(Vs >=  0, Vs < height))
                indices = indices[validMask]

                normalImage = np.zeros((height * width, 3))
                normalImage[indices] = color_map[kmeans_N.labels_[validMask]]
                normalImage = normalImage.reshape((height, width, 3))
                cv2.imwrite('test/anchor_planes/normal_color_' + str(k) + '.png', normalImage)

                exit(1)
                pass
            np.save(filename_N, self.anchor_planes_N)
            np.save(filename_d, self.anchor_planes_d)
            continue
        return


def load_image_gt(config, image_id, image, depth, mask, class_ids, parameters, augment=False,
                  use_mini_mask=True):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    augment: If true, apply random image augmentation. Currently, only
        horizontal flipping is offered.
    use_mini_mask: If False, returns full-size masks that are the same height
        and width as the original image. These can be big, for example
        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
        224x224 and are generated by extracting the bounding box of the
        object and resizing it to MINI_MASK_SHAPE.

    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    ## Load image and mask
    shape = image.shape
    image, window, scale, padding = utils.resize_image(
        image,
        min_dim=config.IMAGE_MAX_DIM,
        max_dim=config.IMAGE_MAX_DIM,
        padding=config.IMAGE_PADDING)

    mask = utils.resize_mask(mask, scale, padding)
    
    ## Random horizontal flips.
    if augment and False:
        if np.random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)
            depth = np.fliplr(depth)            
            pass
        pass

    ## Bounding boxes. Note that some boxes might be all zeros
    ## if the corresponding mask got cropped out.
    ## bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = utils.extract_bboxes(mask)
    ## Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)
        pass

    active_class_ids = np.ones(config.NUM_CLASSES, dtype=np.int32)
    ## Image meta data
    image_meta = utils.compose_image_meta(image_id, shape, window, active_class_ids)

    if config.NUM_PARAMETER_CHANNELS > 0:
        if config.OCCLUSION:
            depth = utils.resize_mask(depth, scale, padding)            
            mask_visible = utils.minimize_mask(bbox, depth, config.MINI_MASK_SHAPE)
            mask = np.stack([mask, mask_visible], axis=-1)
        else:
            depth = np.expand_dims(depth, -1)
            depth = utils.resize_mask(depth, scale, padding).squeeze(-1)
            depth = utils.minimize_depth(bbox, depth, config.MINI_MASK_SHAPE)
            mask = np.stack([mask, depth], axis=-1)
            pass
        pass
    return image, image_meta, class_ids, bbox, mask, parameters


def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    ## RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    ## RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    ## Handle COCO crowds
    ## A crowd box in COCO is a bounding box around several instances. Exclude
    ## them from training. A crowd box is given a negative class ID.
    no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)
    
    ## Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    ## Match anchors to GT Boxes
    ## If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    ## If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    ## Neutral anchors are those that don't match the conditions above,
    ## and they don't influence the loss function.
    ## However, don't keep any GT box unmatched (rare, but happens). Instead,
    ## match it to the closest anchor (even if its max IoU is < 0.3).
    #
    ## 1. Set negative anchors first. They get overwritten below if a GT box is
    ## matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    ## 2. Set an anchor for each GT box (regardless of IoU value).
    ## TODO: If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    rpn_match[gt_iou_argmax] = 1
    ## 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    ## Subsample to balance positive and negative anchors
    ## Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        ## Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    ## Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        ## Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    ## For positive anchors, compute shift and scale needed to transform them
    ## to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  ## index into rpn_bbox
    ## TODO: use box_refinment() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        ## Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        ## Convert coordinates to center plus width/height.
        ## GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        ## Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        ## Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        ## Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1

    return rpn_match, rpn_bbox
