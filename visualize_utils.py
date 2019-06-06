"""
Copyright (c) 2017 Matterport, Inc.
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import random
import itertools
import numpy as np
from skimage.measure import find_contours
import cv2

from models.model import detection_layer, unmold_detections
from models.modules import *
from utils import *


def tileImages(image_list, padding_x=5, padding_y=5, background_color=0):
    """Tile images"""
    height = image_list[0][0].shape[0]
    width = image_list[0][0].shape[1]
    result_image = np.full((height * len(image_list) + padding_y * (len(image_list) + 1), width * len(image_list[0]) + padding_x * (len(image_list[0]) + 1), 3), fill_value=background_color, dtype=np.uint8)
    for index_y, images in enumerate(image_list):
        for index_x, image in enumerate(images):
            offset_x = index_x * width + (index_x + 1) * padding_x
            offset_y = index_y * height + (index_y + 1) * padding_y
            if image.ndim == 2:
                image = np.expand_dims(image, axis=-1).tile((1, 1, 3))
                pass
            result_image[offset_y:offset_y + height, offset_x:offset_x + width] = image
            continue
        continue
    return result_image

############################################################
#  Batch visualization
############################################################
def visualizeBatchDeMoN(options, input_dict, results, indexOffset=0, prefix='', concise=False):
    cornerColorMap = {'gt': np.array([255, 0, 0]), 'pred': np.array([0, 0, 255]), 'inp': np.array([0, 255, 0])}
    topdownSize = 256
    
    for batchIndex in range(len(input_dict['image_1'])):
        pose = input_dict['pose'][batchIndex]

        for resultIndex, result in enumerate(results):
            if concise and resultIndex < len(results) - 1:
                continue
            depth_pred = invertDepth(result['depth'][batchIndex]).detach().cpu().numpy().squeeze()
            depth_gt = input_dict['depth'][batchIndex].squeeze()
            if depth_pred.shape[0] != depth_gt.shape[0]:
                depth_pred = cv2.resize(depth_pred, (depth_gt.shape[1], depth_gt.shape[0]))
                pass
            
            if options.scaleMode != 'variant':
                valid_mask = np.logical_and(depth_gt > 1e-4, depth_pred > 1e-4)
                depth_gt_values = depth_gt[valid_mask]
                depth_pred_values = depth_pred[valid_mask]
                scale = np.exp(np.mean(np.log(depth_gt_values) - np.log(depth_pred_values)))
                depth_pred *= scale
                pass
            cv2.imwrite(options.test_dir + '/' + str(indexOffset + batchIndex) + '_depth_pred_' + str(len(results) - 1 - resultIndex) + '.png', drawDepthImage(depth_pred))
            
            
            if 'flow' in result:
                flow_pred = result['flow'][batchIndex, :2].detach().cpu().numpy().transpose((1, 2, 0))
                cv2.imwrite(options.test_dir + '/' + str(indexOffset + batchIndex) + '_flow_pred_' + str(len(results) - 1 - resultIndex) + '.png', cv2.resize(drawFlowImage(flow_pred), (256, 192)))
                pass
            if 'rotation' in result and resultIndex >= len(results) - 2:
                pass
            continue

        if not concise:
            cv2.imwrite(options.test_dir + '/' + str(indexOffset + batchIndex) + '_depth_gt.png', drawDepthImage(input_dict['depth'][batchIndex]))
            cv2.imwrite(options.test_dir + '/' + str(indexOffset + batchIndex) + '_image_0.png', (input_dict['image_1'][batchIndex].transpose((1, 2, 0)) + 0.5) * 255)
            cv2.imwrite(options.test_dir + '/' + str(indexOffset + batchIndex) + '_image_1.png', (input_dict['image_2'][batchIndex].transpose((1, 2, 0)) + 0.5) * 255)
            flow_gt = input_dict['flow'][batchIndex, :2].transpose((1, 2, 0))
            cv2.imwrite(options.test_dir + '/' + str(indexOffset + batchIndex) + '_flow_gt.png', cv2.resize(drawFlowImage(flow_gt), (256, 192)))
            pass
        continue
    return

def visualizeBatchPair(options, config, inp_pair, detection_pair, indexOffset=0, prefix='', suffix='', write_ply=False, write_new_view=False):
    detection_images = []    
    for pair_index, (input_dict, detection_dict) in enumerate(zip(inp_pair, detection_pair)):
        image_dict = visualizeBatchDetection(options, config, input_dict, detection_dict, indexOffset=indexOffset, prefix=prefix, suffix='_' + str(pair_index), prediction_suffix=suffix, write_ply=write_ply, write_new_view=write_new_view)
        detection_images.append(image_dict['detection'])
        continue
    detection_image = tileImages([detection_images])
    return

def visualizeBatchRefinement(options, config, input_dict, results, indexOffset=0, prefix='', suffix='', concise=False):
    if not concise:
        image = (input_dict['image'].detach().cpu().numpy().transpose((0, 2, 3, 1))[0] + 0.5) * 255
        cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_image_0.png', image)
        image_2 = (input_dict['image_2'].detach().cpu().numpy().transpose((0, 2, 3, 1))[0] + 0.5) * 255
        cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_image_1.png', image_2)
        depth_gt = input_dict['depth'].detach().cpu().numpy().squeeze()
        cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_depth_gt.png', drawDepthImage(depth_gt))
        flow_gt = input_dict['flow'][0, :2].detach().cpu().numpy().transpose((1, 2, 0))
        cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_flow_gt.png', cv2.resize(drawFlowImage(flow_gt), (256, 192)))
        pass
    numbers = []
    for resultIndex, result in enumerate(results):
        if 'mask' in result and (options.losses == '' or '0' in options.losses):
            masks = result['mask'].detach().cpu().numpy()
            masks = np.concatenate([np.maximum(1 - masks.sum(0, keepdims=True), 0), masks], axis=0).transpose((1, 2, 0))
            cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_segmentation_' + str(len(results) - 1 - resultIndex) + '.png', drawSegmentationImage(masks, blackIndex=0) * (masks.max(-1, keepdims=True) > 0.5).astype(np.uint8))
            pass
        if concise:
            continue
        if 'depth' in result and (options.losses == '' or '3' in options.losses):
            depth_pred = invertDepth(result['depth']).detach().cpu().numpy().squeeze()
            if depth_pred.shape[0] != depth_gt.shape[0]:
                depth_pred = cv2.resize(depth_pred, (depth_gt.shape[1], depth_gt.shape[0]))
                pass
            
            if options.scaleMode != 'variant':
                valid_mask = np.logical_and(depth_gt > 1e-4, depth_pred > 1e-4)
                depth_gt_values = depth_gt[valid_mask]
                depth_pred_values = depth_pred[valid_mask]
                scale = np.exp(np.mean(np.log(depth_gt_values) - np.log(depth_pred_values)))
                depth_pred *= scale
                pass
            cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_depth_pred_' + str(len(results) - 1 - resultIndex) + '.png', drawDepthImage(depth_pred))
            pass
        if 'plane_depth' in result and (options.losses == '' or '3' in options.losses):
            depth_pred = invertDepth(result['plane_depth']).detach().cpu().numpy().squeeze()
            if depth_pred.shape[0] != depth_gt.shape[0]:
                depth_pred = cv2.resize(depth_pred, (depth_gt.shape[1], depth_gt.shape[0]))
                pass
            
            if options.scaleMode != 'variant':
                valid_mask = np.logical_and(depth_gt > 1e-4, depth_pred > 1e-4)
                depth_gt_values = depth_gt[valid_mask]
                depth_pred_values = depth_pred[valid_mask]
                scale = np.exp(np.mean(np.log(depth_gt_values) - np.log(depth_pred_values)))
                depth_pred *= scale
                pass
            cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_depth_pred_plane_' + str(len(results) - 1 - resultIndex) + '.png', drawDepthImage(depth_pred))
            pass
        if 'flow' in result and (options.losses == '' or '1' in options.losses):
            flow_pred = result['flow'][0, :2].detach().cpu().numpy().transpose((1, 2, 0))
            cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_flow_pred_' + str(len(results) - 1 - resultIndex) + '.png', cv2.resize(drawFlowImage(flow_pred), (256, 192)))
            pass
        if 'rotation' in result and resultIndex >= len(results) - 2:
            pass
        if 'plane' in result and resultIndex > 0:
            numbers.append(np.linalg.norm(result['plane'].detach().cpu().numpy() - results[0]['plane'].detach().cpu().numpy()))
            pass
        if 'warped_image' in result and resultIndex >= len(results) - 2:
            warped_image = ((result['warped_image'].detach().cpu().numpy().transpose((0, 2, 3, 1))[0] + 0.5) * 255).astype(np.uint8)
            cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_image_warped_' + str(len(results) - 1 - resultIndex) + '.png', warped_image)
            pass

        if 'plane_depth_one_hot' in result:
            depth_pred = invertDepth(result['plane_depth_one_hot']).detach().cpu().numpy().squeeze()
            if depth_pred.shape[0] != depth_gt.shape[0]:
                depth_pred = cv2.resize(depth_pred, (depth_gt.shape[1], depth_gt.shape[0]))
                pass            
            cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_depth_pred_plane_onehot_' + str(len(results) - 1 - resultIndex) + '.png', drawDepthImage(depth_pred))
            pass
        
        continue
    if 'parameter' in options.suffix:
        print('plane diff', numbers)
        pass
    return

def visualizeBatchDetection(options, config, input_dict, detection_dict, indexOffset=0, prefix='', suffix='', prediction_suffix='', write_ply=False, write_new_view=False):
    image_dict = {}
    images = input_dict['image'].detach().cpu().numpy().transpose((0, 2, 3, 1))
    images = unmold_image(images, config)
    image = images[0]
    cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_image' + suffix + '.png', image[80:560])
    
    if 'warped_image' in input_dict:
        warped_images = input_dict['warped_image'].detach().cpu().numpy().transpose((0, 2, 3, 1))
        warped_images = unmold_image(warped_images, config)
        warped_image = warped_images[0]
        cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_image' + suffix + '_warped.png', warped_image[80:560])
        pass

    if 'warped_depth' in input_dict:
        warped_depth = input_dict['warped_depth'].detach().cpu().numpy()
        cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_depth' + suffix + '_warped.png', drawDepthImage(warped_depth[80:560]))
        pass

    if 'warped_mask' in input_dict:
        warped_mask = input_dict['warped_mask'].detach().cpu().numpy()[0]
        pass

    if 'depth' in input_dict:
        depths = input_dict['depth'].detach().cpu().numpy()                
        depth_gt = depths[0]
        cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_depth' + suffix + '.png', drawDepthImage(depth_gt[80:560]))
        pass

    windows = (0, 0, images.shape[1], images.shape[2])        
    windows = (0, 0, images.shape[1], images.shape[2])                
    class_colors = ColorPalette(config.NUM_CLASSES).getColorMap().tolist()        

    if 'mask' in input_dict:
        box_image = image.copy()
        boxes = input_dict['bbox'][0].detach().cpu().numpy()
        masks = input_dict['mask'][0].detach().cpu().numpy()
        if config.NUM_PARAMETER_CHANNELS > 0:
            depths = masks[:, :, :, 1]
            masks = masks[:, :, :, 0]
            pass

        segmentation_image = image * 0.0
        for box, mask in zip(boxes, masks):
            box = np.round(box).astype(np.int32)
            mask = cv2.resize(mask, (box[3] - box[1], box[2] - box[0]))
            segmentation_image[box[0]:box[2], box[1]:box[3]] = np.minimum(segmentation_image[box[0]:box[2], box[1]:box[3]] + np.expand_dims(mask, axis=-1) * np.random.randint(255, size=(3, ), dtype=np.int32), 255)
            continue
        cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_segmentation' + suffix + '.png', segmentation_image.astype(np.uint8)[80:560])
        if config.NUM_PARAMETER_CHANNELS > 0 and not config.OCCLUSION:
            depth_image = np.zeros((image.shape[0], image.shape[1]))
            for box, patch_depth in zip(boxes, depths):
                box = np.round(box).astype(np.int32)
                patch_depth = cv2.resize(patch_depth, (box[3] - box[1], box[2] - box[0]), cv2.INTER_NEAREST)
                depth_image[box[0]:box[2], box[1]:box[3]] = patch_depth
                continue
            cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_depth_patch' + suffix + '.png', drawDepthImage(depth_image[80:560]))
            pass
        pass

    if 'boundary' in detection_dict:
        boundary_pred = detection_dict['boundary'].detach().cpu().numpy()[0]
        boundary_gt = input_dict['boundary'].detach().cpu().numpy()[0]
        for name, boundary in [('gt', boundary_gt), ('pred', boundary_pred)]:
            boundary_image = image.copy()
            boundary_image[boundary[0] > 0.5] = np.array([255, 0, 0])
            boundary_image[boundary[1] > 0.5] = np.array([0, 0, 255])        
            cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_boundary' + suffix + '_' + name + '.png', boundary_image)
            continue
        pass
        
    if 'depth' in detection_dict:    
        depth_pred = detection_dict['depth'][0].detach().cpu().numpy()
        cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_depth' + suffix + prediction_suffix + '.png', drawDepthImage(depth_pred[80:560]))                    
        if options.debug:
            valid_mask = (depth_gt > 1e-4) * (input_dict['segmentation'].detach().cpu().numpy()[0] >= 0) * (detection_dict['mask'].detach().cpu().numpy().squeeze() > 0.5)
            pass
        pass
    
    if 'depth_np' in detection_dict:
        cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_depth' + suffix + prediction_suffix + '_np.png', drawDepthImage(detection_dict['depth_np'].squeeze().detach().cpu().numpy()[80:560]))
        pass

    if 'depth_ori' in detection_dict:
        cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_depth' + suffix + prediction_suffix + '_ori.png', drawDepthImage(detection_dict['depth_ori'].squeeze().detach().cpu().numpy()[80:560]))
        pass
    

    if 'detection' in detection_dict and len(detection_dict['detection']) > 0:
        detections = detection_dict['detection'].detach().cpu().numpy()
        detection_masks = detection_dict['masks'].detach().cpu().numpy().transpose((1, 2, 0))
        if 'flag' in detection_dict:
            detection_flags = detection_dict['flag']
        else:
            detection_flags = {}
            pass
        instance_image, normal_image, depth_image = draw_instances(config, image, depth_gt, detections[:, :4], detection_masks > 0.5, detections[:, 4].astype(np.int32), detections[:, 6:], detections[:, 5], draw_mask=True, transform_planes=False, detection_flags=detection_flags)
        image_dict['detection'] = instance_image
        cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_segmentation' + suffix + prediction_suffix + '.png', instance_image[80:560])
    else:
        image_dict['detection'] = np.zeros(image.shape, dtype=image.dtype)
        pass

    if write_new_view and False:
        detection_masks = detection_dict['masks']
        pose = np.eye(4)
        pose[:3, :3] = np.matmul(axisAngleToRotationMatrix(np.array([-1, 0, 0]), np.pi / 18 * 0), axisAngleToRotationMatrix(np.array([0, 0, -1]), np.pi / 18))
        pose[:3, 3] = np.array([-0.4, 0, 0])        
        drawNewViewDepth(options.test_dir + '/' + str(indexOffset) + '_new_view' + suffix + prediction_suffix + '.png', detection_masks[:, 80:560].detach().cpu().numpy(), detection_dict['plane_XYZ'].detach().cpu().numpy().transpose((0, 2, 3, 1))[:, 80:560], input_dict['camera'].detach().cpu().numpy(), pose)
        depth = depth_gt[80:560]
        ranges = config.getRanges(input_dict['camera']).detach().cpu().numpy()
        XYZ_gt = ranges * np.expand_dims(depth, axis=-1)
        drawNewViewDepth(options.test_dir + '/' + str(indexOffset) + '_new_view_depth_gt' + suffix + prediction_suffix + '.png', np.expand_dims(depth > 1e-4, 0), np.expand_dims(XYZ_gt, 0), input_dict['camera'].detach().cpu().numpy(), pose)
        depth = detection_dict['depth_np'].squeeze()[80:560]
        ranges = config.getRanges(input_dict['camera']).detach().cpu().numpy()
        XYZ_gt = ranges * np.expand_dims(depth, axis=-1)
        drawNewViewDepth(options.test_dir + '/' + str(indexOffset) + '_new_view_depth_pred' + suffix + prediction_suffix + '.png', np.expand_dims(depth > 1e-4, 0), np.expand_dims(XYZ_gt, 0), input_dict['camera'].detach().cpu().numpy(), pose)        
        pass

    if write_new_view:
        detection_masks = detection_dict['masks'][:, 80:560].detach().cpu().numpy()
        XYZ_pred = detection_dict['plane_XYZ'].detach().cpu().numpy().transpose((0, 2, 3, 1))[:, 80:560]
        depth = depth_gt[80:560]
        ranges = config.getRanges(input_dict['camera']).detach().cpu().numpy()
        XYZ_gt = np.expand_dims(ranges * np.expand_dims(depth, axis=-1), 0)
        valid_mask = np.expand_dims(depth > 1e-4, 0).astype(np.float32)
        camera = input_dict['camera'].detach().cpu().numpy()

        valid_mask = np.expand_dims(cv2.resize(valid_mask[0], (256, 192)), 0)
        XYZ_gt = np.expand_dims(cv2.resize(XYZ_gt[0], (256, 192)), 0)
        detection_masks = np.stack([cv2.resize(detection_masks[c], (256, 192)) for c in range(len(detection_masks))], axis=0)
        XYZ_pred = np.stack([cv2.resize(XYZ_pred[c], (256, 192)) for c in range(len(XYZ_pred))], axis=0)
        locations = [np.array([-0.4, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0.4, 0, 0])]
        angle_pairs = [(np.array([-1, 0, 0, np.pi / 18 * 0]), np.array([0, 0, -1, np.pi / 18])), (np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])), (np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])), (np.array([-1, 0, 0, np.pi / 18 * 0]), np.array([0, 0, 1, np.pi / 18]))]
        num_frames = [25, 10, 25]
        for c in range(len(locations) - 1):
            if c == 2:
                continue
            for frame in range(num_frames[c]):
                ratio = float(frame + 1) / num_frames[c]
                location = locations[c] + (locations[c + 1] - locations[c]) * ratio
                angle_pair = [angle_pairs[c][dim] + (angle_pairs[c + 1][dim] - angle_pairs[c][dim]) * ratio for dim in range(2)]
                pose = np.eye(4)
                pose[:3, :3] = np.matmul(axisAngleToRotationMatrix(angle_pair[0][:3], angle_pair[0][3]), axisAngleToRotationMatrix(angle_pair[1][:3], angle_pair[1][3]))
                pose[:3, 3] = location

                index_offset = sum(num_frames[:c]) + frame
                drawNewViewDepth(options.test_dir + '/' + str(indexOffset) + '_video/' + str(index_offset) + '.png', detection_masks, XYZ_pred, camera, pose)
                
                drawNewViewDepth(options.test_dir + '/' + str(indexOffset) + '_video_gt/' + str(index_offset) + '.png', valid_mask, XYZ_gt, camera, pose)
                continue
            continue
        exit(1)
        pass
    
    if write_ply:
        detection_masks = detection_dict['masks']
        if 'plane_XYZ' not in detection_dict:
            plane_XYZ = planeXYZModule(config.getRanges(input_dict['camera']), detection_dict['detection'][:, 6:9], width=config.IMAGE_MAX_DIM, height=config.IMAGE_MIN_DIM)
            plane_XYZ = plane_XYZ.transpose(1, 2).transpose(0, 1).transpose(2, 3).transpose(1, 2)
            zeros = torch.zeros(int(plane_XYZ.shape[0]), 3, (config.IMAGE_MAX_DIM - config.IMAGE_MIN_DIM) // 2, config.IMAGE_MAX_DIM).cuda()
            plane_XYZ = torch.cat([zeros, plane_XYZ, zeros], dim=2)
            detection_dict['plane_XYZ'] = plane_XYZ
            pass

        print(options.test_dir + '/' + str(indexOffset) + '_model' + suffix + prediction_suffix + '.ply')
        writePLYFileMask(options.test_dir + '/' + str(indexOffset) + '_model' + suffix + prediction_suffix + '.ply', image[80:560], detection_masks[:, 80:560].detach().cpu().numpy(), detection_dict['plane_XYZ'].detach().cpu().numpy().transpose((0, 2, 3, 1))[:, 80:560], write_occlusion='occlusion' in options.suffix)

        pose = np.eye(4)
        pose[:3, :3] = np.matmul(axisAngleToRotationMatrix(np.array([-1, 0, 0]), np.pi / 18), axisAngleToRotationMatrix(np.array([0, -1, 0]), np.pi / 18))
        pose[:3, 3] = np.array([-0.4, 0.3, 0])

        current_dir = os.path.dirname(os.path.realpath(__file__))
        pose_filename = current_dir + '/test/pose_new_view.txt'
        print(pose_filename)
        with open(pose_filename, 'w') as f:
            for row in pose:
                for col in row:
                    f.write(str(col) + '\t')
                    continue
                f.write('\n')
                continue
            f.close()
            pass
        model_filename = current_dir + '/' + options.test_dir + '/' + str(indexOffset) + '_model' + suffix + prediction_suffix + '.ply'
        output_filename = current_dir + '/' + options.test_dir + '/' + str(indexOffset) + '_model' + suffix + prediction_suffix + '.png'
        try:
            os.system('../../../Screenshoter/Screenshoter --model_filename=' + model_filename + ' --output_filename=' + output_filename + ' --pose_filename=' + pose_filename)
        except:
            pass
        pass
    return image_dict


def visualizeBatchDepth(options, config, input_dict, detection_dict, indexOffset=0, prefix='', suffix='', write_ply=False):
    image_dict = {}
    images = input_dict['image'].detach().cpu().numpy().transpose((0, 2, 3, 1))
    images = unmold_image(images, config)
    for batchIndex, image in enumerate(images):
        cv2.imwrite(options.test_dir + '/' + str(indexOffset + batchIndex) + '_image' + suffix + '.png', image)
        continue

    depths = input_dict['depth'].detach().cpu().numpy()
    for batchIndex, depth in enumerate(depths):
        cv2.imwrite(options.test_dir + '/' + str(indexOffset + batchIndex) + '_depth' + suffix + '.png', drawDepthImage(depth))
        continue

    if 'depth_np' in detection_dict:
        for batchIndex, depth in enumerate(detection_dict['depth_np'].detach().cpu().numpy()):
            cv2.imwrite(options.test_dir + '/' + str(indexOffset + batchIndex) + '_depth_pred_np' + suffix + '.png', drawDepthImage(depth))
            continue
        pass
    
    return

def visualizeBatchSingle(options, config, images, image_metas, rpn_rois, depths, dicts, input_dict={}, inference={}, indexOffset=0, prefix='', suffix='', compare_planenet=False):

    image = images[0]
    cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_image' + suffix + '.png', image)

    depth = depths[0]
    cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_depth' + suffix + '.png', drawDepthImage(depth))

    windows = (0, 0, images.shape[1], images.shape[2])
    class_colors = ColorPalette(config.NUM_CLASSES).getColorMap(returnTuples=True)
    instance_colors = ColorPalette(1000).getColorMap(returnTuples=True)

    if 'mask' in input_dict:
        box_image = image.copy()
        boxes = input_dict['bbox'][0].detach().cpu().numpy()
        masks = input_dict['mask'][0].detach().cpu().numpy()

        for box, mask in zip(boxes, masks):
            box = np.round(box).astype(np.int32)
            cv2.rectangle(box_image, (box[1], box[0]), (box[3], box[2]), color=(0, 0, 255), thickness=2)
            continue
        
        segmentation_image = image * 0.0
        for box, mask in zip(boxes, masks):
            box = np.round(box).astype(np.int32)
            mask = cv2.resize(mask, (box[3] - box[1], box[2] - box[0]))
            segmentation_image[box[0]:box[2], box[1]:box[3]] = np.minimum(segmentation_image[box[0]:box[2], box[1]:box[3]] + np.expand_dims(mask, axis=-1) * np.random.randint(255, size=(3, ), dtype=np.int32), 255)
            continue
        cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_detection' + suffix + '.png', segmentation_image.astype(np.uint8))
        pass
    
    for name, result_dict in dicts:

        if len(rpn_rois) > 0:
            detections, keep_indices, ori_rois = detection_layer(config, rpn_rois.unsqueeze(0), result_dict['mrcnn_class'], result_dict['mrcnn_bbox'], result_dict['mrcnn_parameter'], image_metas, return_indices=True)
            box_image = image.copy()
            for instance_index, box in enumerate(detections.detach().cpu().numpy().astype(np.int32)):
                cv2.rectangle(box_image, (box[1], box[0]), (box[3], box[2]), color=class_colors[int(box[4])], thickness=3)
                continue
        else:
            continue
        
        if len(detections) > 0:
            detections[:, :4] = ori_rois

            detections = detections.detach().cpu().numpy()
            mrcnn_mask = result_dict['mrcnn_mask'][keep_indices].detach().cpu().numpy()

            if name == 'gt':
                class_mrcnn_mask = np.zeros(list(mrcnn_mask.shape) + [config.NUM_CLASSES], dtype=np.float32)
                for index, (class_id, mask) in enumerate(zip(detections[:, 4].astype(np.int32), mrcnn_mask)):
                    if config.GLOBAL_MASK:
                        class_mrcnn_mask[index, :, :, 0] = mask
                    else:
                        class_mrcnn_mask[index, :, :, class_id] = mask
                        pass
                    continue
                mrcnn_mask = class_mrcnn_mask
            else:
                mrcnn_mask = mrcnn_mask.transpose((0, 2, 3, 1))
                pass

            box_image = image.copy()
            for instance_index, box in enumerate(detections.astype(np.int32)):
                cv2.rectangle(box_image, (box[1], box[0]), (box[3], box[2]), color=tuple(class_colors[int(box[4])]), thickness=3)
                continue
            
            final_rois, final_class_ids, final_scores, final_masks, final_parameters = unmold_detections(config, detections, mrcnn_mask, image.shape, windows, debug=False)

            result = {
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
                "parameters": final_parameters,
            }


            instance_image, normal_image, depth_image = draw_instances(config, image, depth, result['rois'], result['masks'], result['class_ids'], result['parameters'], result['scores'])
            cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_detection' + suffix + '_' + name + '.png', instance_image)
            cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_depth' + suffix + '_' + name + '.png', depth_image)
        else:
            print('no detections')
            pass
        continue


    if len(inference) > 0:
        instance_image, normal_image, depth_image = draw_instances(config, image, depth, inference['rois'], inference['masks'], inference['class_ids'], inference['parameters'], inference['scores'], draw_mask=True)
        cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_detection' + suffix + '.png', instance_image)

        if compare_planenet:
            print(image.shape, image.min(), image.max())
            pred_dict = detector.detect(image[80:560])
            cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_planenet_segmentation.png', drawSegmentationImage(pred_dict['segmentation'], blackIndex=10))
            cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_planenet_depth.png', drawDepthImage(pred_dict['depth']))
            pass
        pass
    return

def visualizeBatchBoundary(options, config, images, boundary_pred, boundary_gt, indexOffset=0):
    images = (images.detach().cpu().numpy().transpose((0, 2, 3, 1)) + config.MEAN_PIXEL).astype(np.uint8)
    boundary_pred = boundary_pred.detach().cpu().numpy()
    boundary_gt = boundary_gt.detach().cpu().numpy()
    for batchIndex in range(len(images)):
        for name, boundary in [('gt', boundary_gt[batchIndex]), ('pred', boundary_pred[batchIndex])]:
            image = images[batchIndex].copy()
            image[boundary[0] > 0.5] = np.array([255, 0, 0])
            image[boundary[1] > 0.5] = np.array([0, 0, 255])        
            cv2.imwrite(options.test_dir + '/' + str(indexOffset + batchIndex) + '_boundary_' + name + '.png', image)
            continue
        continue
    return

############################################################
#  Visualization
############################################################


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  np.minimum(image[:, :, c] *
                                             (1 - alpha) + alpha * color[c], 255),
                                  image[:, :, c])
    return image




def draw_instances(config, image, depth, boxes, masks, class_ids, parameters,
                      scores=None, title="",
                      figsize=(16, 16), ax=None, draw_mask=False, transform_planes=False, statistics=[], detection_flags={}):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    figsize: (optional) the size of the image.
    """
    ## Number of instances
    N = len(boxes)
    if not N:
        pass
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    ## Generate random colors
    instance_colors = ColorPalette(N).getColorMap(returnTuples=True)
    if len(detection_flags) and False:
        for index in range(N):
            if detection_flags[index] < 0.5:
                instance_colors[index] = (128, 128, 128)
                pass
            continue
        pass

    class_colors = ColorPalette(11).getColorMap(returnTuples=True)
    class_colors[0] = (128, 128, 128)
    
    ## Show area outside image boundaries.
    height, width = image.shape[:2]
    masked_image = image.astype(np.uint8).copy()
    normal_image = np.zeros(image.shape)
    depth_image = depth.copy()

    for i in range(N):

        ## Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        
        ## Label
        class_id = class_ids[i]

        score = scores[i] if scores is not None else None
        x = random.randint(x1, (x1 + x2) // 2)

        ## Mask
        mask = masks[:, :, i]
        masked_image = apply_mask(masked_image.astype(np.float32), mask, instance_colors[i]).astype(np.uint8)
        
        ## Mask Polygon
        ## Pad to ensure proper polygons for masks that touch image edges.
        if draw_mask:
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                ## Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                cv2.polylines(masked_image, np.expand_dims(verts.astype(np.int32), 0), True, color=class_colors[class_id])
                continue

        continue
    
    normal_image = drawNormalImage(normal_image)    
    depth_image = drawDepthImage(depth_image)
    return masked_image.astype(np.uint8), normal_image.astype(np.uint8), depth_image




## Write the reconstruction result to PLY file
def writePLYFileMask(filename, image, masks, plane_XYZ, write_occlusion=False):

    width = image.shape[1]
    height = image.shape[0]
    
    betweenRegionThreshold = 0.1
    nonPlanarRegionThreshold = 0.02
    dotThreshold = np.cos(np.deg2rad(30))

    faces = []
    points = []

    masks = np.round(masks)
    plane_depths = plane_XYZ[:, :, :, 1] * masks + 10 * (1 - masks)
    segmentation = plane_depths.argmin(0)
    for mask_index, (mask, XYZ) in enumerate(zip(masks, plane_XYZ)):
        indices = np.nonzero(mask > 0.5)
        for y, x in zip(indices[0], indices[1]):
            if y == height - 1 or x == width - 1:
                continue
            validNeighborPixels = []
            for neighborPixel in [(x, y + 1), (x + 1, y), (x + 1, y + 1)]:
                if mask[neighborPixel[1], neighborPixel[0]] > 0.5:
                    validNeighborPixels.append(neighborPixel)
                    pass
                continue
            if len(validNeighborPixels) == 3:
                faces.append([len(points) + c for c in range(3)])
                points += [(XYZ[pixel[1], pixel[0]], pixel, segmentation[pixel[1], pixel[0]] == mask_index) for pixel in [(x, y), (x + 1, y + 1), (x + 1, y)]]
                faces.append([len(points) + c for c in range(3)])
                points += [(XYZ[pixel[1], pixel[0]], pixel, segmentation[pixel[1], pixel[0]] == mask_index) for pixel in [(x, y), (x, y + 1), (x + 1, y + 1)]]
            elif len(validNeighborPixels) == 2:
                faces.append([len(points) + c for c in range(3)])
                points += [(XYZ[pixel[1], pixel[0]], pixel, segmentation[pixel[1], pixel[0]] == mask_index) for pixel in [(x, y), (validNeighborPixels[0][0], validNeighborPixels[0][1]), (validNeighborPixels[1][0], validNeighborPixels[1][1])]]
                pass
            continue
        continue

    imageFilename = "textureless"
    with open(filename, 'w') as f:
        header = """ply
format ascii 1.0"""
        header += imageFilename
        header += """
element vertex """
        header += str(len(points))
        header += """
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face """
        header += str(len(faces))
        header += """
property list uchar int vertex_indices
end_header
"""
        f.write(header)
        for point in points:
            X = point[0][0]
            Y = point[0][1]
            Z = point[0][2]
            if not write_occlusion or point[2]:
                color = image[point[1][1], point[1][0]]
            else:
                color = (128, 128, 128)
                pass
            f.write(str(X) + ' ' +    str(Z) + ' ' + str(-Y) + ' ' + str(color[2]) + ' ' + str(color[1]) + ' ' + str(color[0]) + '\n')
            continue

        for face in faces:
            valid = True
            f.write('3 ')                
            for c in face:
                f.write(str(c) + ' ')
                continue
            f.write('\n')
            continue
        f.close()
        pass
    return


def drawNewViewDepth(depth_filename, masks, XYZs, camera, pose):
    faces = []
    width, height = masks.shape[2], masks.shape[1]
    for mask, XYZ in zip(masks, XYZs):
        indices = np.nonzero(mask > 0.5)
        for y, x in zip(indices[0], indices[1]):
            if y == height - 1 or x == width - 1:
                continue
            validNeighborPixels = []
            for neighborPixel in [(x, y + 1), (x + 1, y), (x + 1, y + 1)]:
                if mask[neighborPixel[1], neighborPixel[0]] > 0.5:
                    validNeighborPixels.append(neighborPixel)
                    pass
                continue
            if len(validNeighborPixels) == 3:
                faces.append([XYZ[pixel[1], pixel[0]] for pixel in [(x, y), (x + 1, y + 1), (x + 1, y)]])
                faces.append([XYZ[pixel[1], pixel[0]] for pixel in [(x, y), (x, y + 1), (x + 1, y + 1)]])
            elif len(validNeighborPixels) == 2:
                faces.append([XYZ[pixel[1], pixel[0]] for pixel in [(x, y), (validNeighborPixels[0][0], validNeighborPixels[0][1]), (validNeighborPixels[1][0], validNeighborPixels[1][1])]])
                pass
            continue
        continue
    faces = np.array(faces)
    XYZ = faces.reshape((-1, 3))
    XYZ = np.matmul(np.concatenate([XYZ, np.ones((len(XYZ), 1))], axis=-1), pose.transpose())
    XYZ = XYZ[:, :3] / XYZ[:, 3:]
    points = XYZ[:, :3]
    depth = XYZ[:, 1:2]
    depth = np.clip(depth / 5 * 255, 0, 255).astype(np.uint8)
    colors = cv2.applyColorMap(255 - depth, colormap=cv2.COLORMAP_JET).reshape((-1, 3))

    imageFilename = "textureless"
    filename = 'test/model.ply'
    with open(filename, 'w') as f:
        header = """ply
format ascii 1.0"""
        header += imageFilename
        header += """
element vertex """
        header += str(len(points))
        header += """
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face """
        header += str(len(points) // 3)
        header += """
property list uchar int vertex_indices
end_header
"""
        f.write(header)
        for point, color in zip(points, colors):
            X = point[0]
            Y = point[1]
            Z = point[2]
            f.write(str(X) + ' ' +    str(Z) + ' ' + str(-Y) + ' ' + str(color[2]) + ' ' + str(color[1]) + ' ' + str(color[0]) + '\n')
            continue

        faces = np.arange(len(points)).reshape((-1, 3))
        for face in faces:
            valid = True
            f.write('3 ')                
            for c in face:
                f.write(str(c) + ' ')
                continue
            f.write('\n')
            continue
        f.close()
        pass
    pose = np.eye(4)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    pose_filename = current_dir + '/test/pose_new_view.txt'
    with open(pose_filename, 'w') as f:
        for row in pose:
            for col in row:
                f.write(str(col) + '\t')
                continue
            f.write('\n')
            continue
        f.close()
        pass
    model_filename = current_dir + '/test/model.ply'
    output_filename = current_dir + '/' + depth_filename
    try:
        os.system('../../../Screenshoter/Screenshoter --model_filename=' + model_filename + ' --output_filename=' + output_filename + ' --pose_filename=' + pose_filename)
    except:
        print('depth rendering failed')
        pass
    return

def rotateModel(model_filename, output_folder):
    locations = [np.array([-0.4, 0.3, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0.4, 0.3, 0])]
    angle_pairs = [(np.array([-1, 0, 0, np.pi / 18]), np.array([0, -1, 0, np.pi / 18])), (np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])), (np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])), (np.array([-1, 0, 0, np.pi / 18]), np.array([0, 1, 0, np.pi / 18]))]
    num_frames = [50, 20, 50]
    for c in range(len(locations) - 1):
        for frame in range(num_frames[c]):
            ratio = float(frame + 1) / num_frames[c]
            location = locations[c] + (locations[c + 1] - locations[c]) * ratio
            angle_pair = [angle_pairs[c][dim] + (angle_pairs[c + 1][dim] - angle_pairs[c][dim]) * ratio for dim in range(2)]
            pose = np.eye(4)
            pose[:3, :3] = np.matmul(axisAngleToRotationMatrix(angle_pair[0][:3], angle_pair[0][3]), axisAngleToRotationMatrix(angle_pair[1][:3], angle_pair[1][3]))
            pose[:3, 3] = location
            current_dir = os.path.dirname(os.path.realpath(__file__))
            pose_filename = output_folder + '/%04d'%(sum(num_frames[:c]) + frame) + '.txt'
            with open(pose_filename, 'w') as f:
                for row in pose:
                    for col in row:
                        f.write(str(col) + '\t')
                        continue
                    f.write('\n')
                    continue
                f.close()
                pass
            continue
        continue
    try:
        os.system('../../../Recorder/Recorder --model_filename=' + model_filename + ' --output_folder=' + output_folder + ' --pose_folder=' + output_folder + ' --num_frames=' + str(sum(num_frames)))
    except:
        print('Recording failed')
        pass
    pass

def visualizeGraph(var, params):
    """Visualize the network"""
    from torchviz import make_dot
    return make_dot(var, params)

if __name__ == '__main__':
    pose = np.eye(4)
    
    current_dir = os.path.dirname(os.path.realpath(__file__))
    pose_filename = current_dir + '/test/pose.txt'
    with open(pose_filename, 'w') as f:
        for row in pose:
            for col in row:
                f.write(str(col) + '\t')
                continue
            f.write('\n')
            continue
        f.close()
        pass
    test_dir = 'test/occlusion_debug'
    indexOffset = 33
    model_filename = current_dir + '/test/model.ply'
    output_filename = current_dir + '/' + test_dir + '/' + str(indexOffset) + '_model_0_occlusion.png'
    print('screenshot', output_filename)
    os.system('../../../Screenshoter/Screenshoter --model_filename=' + model_filename + ' --output_filename=' + output_filename + ' --pose_filename=' + pose_filename)
    exit(1)    
