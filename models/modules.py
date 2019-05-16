"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn.functional as F
import time
import numpy as np
from torch import nn
import sys

def unmoldDetections(config, camera, detections, detection_masks, depth_np, unmold_masks=True, debug=False):
    """Reformats the detections of one image from the format of the neural
    network output to a format suitable for use in the rest of the
    application.

    detections: [N, (y1, x1, y2, x2, class_id, score)]
    mrcnn_mask: [N, height, width, num_classes]
    image_shape: [height, width, depth] Original size of the image before resizing
    window: [y1, x1, y2, x2] Box in the image where the real image is
            excluding the padding.

    Returns:
    boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
    class_ids: [N] Integer class IDs for each bounding box
    scores: [N] Float probability scores of the class_id
    masks: [height, width, num_instances] Instance masks
    """
    if config.GLOBAL_MASK:
        masks = detection_masks[torch.arange(len(detection_masks)).cuda().long(), 0, :, :]
    else:
        masks = detection_masks[torch.arange(len(detection_masks)).cuda().long(), detections[:, 4].long(), :, :]
        pass

    final_masks = []
    for detectionIndex in range(len(detections)):
        box = detections[detectionIndex][:4].long()
        if (box[2] - box[0]) * (box[3] - box[1]) <= 0:
            continue
            
        mask = masks[detectionIndex]
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = F.upsample(mask, size=(box[2] - box[0], box[3] - box[1]), mode='bilinear')
        mask = mask.squeeze(0).squeeze(0)

        final_mask = torch.zeros(config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM).cuda()
        final_mask[box[0]:box[2], box[1]:box[3]] = mask
        final_masks.append(final_mask)
        continue
    final_masks = torch.stack(final_masks, dim=0)
    
    if config.NUM_PARAMETER_CHANNELS > 0:
        ## We could potentially predict depth and/or normals for each instance (not being used)
        parameters_array = detection_masks[torch.arange(len(detection_masks)).cuda().long(), -config.NUM_PARAMETER_CHANNELS:, :, :]
        final_parameters_array = []
        for detectionIndex in range(len(detections)):
            box = detections[detectionIndex][:4].long()
            if (box[2] - box[0]) * (box[3] - box[1]) <= 0:
                continue
            parameters = F.upsample(parameters_array[detectionIndex].unsqueeze(0), size=(box[2] - box[0], box[3] - box[1]), mode='bilinear').squeeze(0)
            final_parameters = torch.zeros(config.NUM_PARAMETER_CHANNELS, config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM).cuda()
            final_parameters[:, box[0]:box[2], box[1]:box[3]] = parameters
            final_parameters_array.append(final_parameters)
            continue
        final_parameters = torch.stack(final_parameters_array, dim=0)        
        final_masks = torch.cat([final_masks.unsqueeze(1), final_parameters], dim=1)
        pass

    masks = final_masks

    if 'normal' in config.ANCHOR_TYPE:
        ## Compute offset based normal prediction and depthmap prediction
        ranges = config.getRanges(camera).transpose(1, 2).transpose(0, 1)
        zeros = torch.zeros(3, (config.IMAGE_MAX_DIM - config.IMAGE_MIN_DIM) // 2, config.IMAGE_MAX_DIM).cuda()        
        ranges = torch.cat([zeros, ranges, zeros], dim=1)
        
        if config.NUM_PARAMETER_CHANNELS == 4:
            ## If we predict depthmap and normal map for each instance, we compute normals again (not used)
            masks_cropped = masks[:, 0:1, 80:560]
            mask_sum = masks_cropped.sum(-1).sum(-1)
            plane_normals = (masks[:, 2:5, 80:560] * masks_cropped).sum(-1).sum(-1) / mask_sum
            plane_normals = plane_normals / torch.clamp(torch.norm(plane_normals, dim=-1, keepdim=True), min=1e-4)
            XYZ_np_cropped = (ranges * masks[:, 1:2])[:, :, 80:560]
            offsets = ((plane_normals.view(-1, 3, 1, 1) * XYZ_np_cropped).sum(1, keepdim=True) * masks_cropped).sum(-1).sum(-1) / mask_sum
            plane_parameters = plane_normals * offsets.view((-1, 1))
            masks = masks[:, 0]            
        else:
            if config.NUM_PARAMETER_CHANNELS > 0:
                ## If we predict depthmap independently for each instance, we use the individual depthmap instead of the global depth map (not used)                
                if config.OCCLUSION:
                    XYZ_np = ranges * depth_np                
                    XYZ_np_cropped = XYZ_np[:, 80:560]
                    masks_cropped = masks[:, 1, 80:560]                    
                    masks = masks[:, 0]
                else:
                    XYZ_np_cropped = (ranges * masks[:, 1:2])[:, :, 80:560]
                    masks = masks[:, 0]
                    masks_cropped = masks[:, 80:560]
                    pass
            else:
                ## We use the global depthmap prediction to compute plane offsets
                XYZ_np = ranges * depth_np                
                XYZ_np_cropped = XYZ_np[:, 80:560]
                masks_cropped = masks[:, 80:560]                            
                pass

            if config.FITTING_TYPE % 2 == 1:
                ## We fit all plane parameters using depthmap prediction (not used)
                A = masks_cropped.unsqueeze(1) * XYZ_np_cropped
                b = masks_cropped
                Ab = (A * b.unsqueeze(1)).sum(-1).sum(-1)
                AA = (A.unsqueeze(2) * A.unsqueeze(1)).sum(-1).sum(-1)
                plane_parameters = torch.stack([torch.matmul(torch.inverse(AA[planeIndex]), Ab[planeIndex]) for planeIndex in range(len(AA))], dim=0)
                plane_offsets = torch.norm(plane_parameters, dim=-1, keepdim=True)
                plane_parameters = plane_parameters / torch.clamp(torch.pow(plane_offsets, 2), 1e-4)                
            else:
                ## We compute only plane offset using depthmap prediction                
                plane_parameters = detections[:, 6:9]            
                plane_normals = plane_parameters / torch.clamp(torch.norm(plane_parameters, dim=-1, keepdim=True), 1e-4)
                offsets = ((plane_normals.view(-1, 3, 1, 1) * XYZ_np_cropped).sum(1) * masks_cropped).sum(-1).sum(-1) / torch.clamp(masks_cropped.sum(-1).sum(-1), min=1e-4)
                plane_parameters = plane_normals * offsets.view((-1, 1))
                pass
            pass
        detections = torch.cat([detections[:, :6], plane_parameters], dim=-1)
        pass
    return detections, masks

def planeXYZModule(ranges, planes, width, height, max_depth=10):
    """Compute plane XYZ from plane parameters
    ranges: K^(-1)x
    planes: plane parameters
    
    Returns:
    plane depthmaps
    """
    planeOffsets = torch.norm(planes, dim=-1, keepdim=True)
    planeNormals = planes / torch.clamp(planeOffsets, min=1e-4)

    normalXYZ = torch.matmul(ranges, planeNormals.transpose(0, 1))
    normalXYZ[normalXYZ == 0] = 1e-4
    planeDepths = planeOffsets.squeeze(-1) / normalXYZ
    planeDepths = torch.clamp(planeDepths, min=0, max=max_depth)
    return planeDepths.unsqueeze(-1) * ranges.unsqueeze(2)

def planeDepthsModule(ranges, planes, width, height, max_depth=10):
    """Compute coordinate maps from plane parameters
    ranges: K^(-1)x
    planes: plane parameters
    
    Returns:
    plane coordinate maps
    """
    planeOffsets = torch.norm(planes, dim=-1, keepdim=True)
    planeNormals = planes / torch.clamp(planeOffsets, min=1e-4)

    normalXYZ = torch.matmul(ranges, planeNormals.transpose(0, 1))
    normalXYZ[normalXYZ == 0] = 1e-4
    planeDepths = planeOffsets.squeeze(-1) / normalXYZ
    if max_depth > 0:
        planeDepths = torch.clamp(planeDepths, min=0, max=max_depth)
        pass
    return planeDepths

def warpModuleDepth(config, camera, depth_1, features_2, extrinsics_1, extrinsics_2, width, height):
    """Warp one feature map to another view given camera pose and depth"""
    padding = (width - height) // 2
    XYZ_1 = config.getRanges(camera) * depth_1[padding:-padding].unsqueeze(-1)
    warped_features, valid_mask = warpModuleXYZ(config, camera, XYZ_1.unsqueeze(2), features_2, extrinsics_1, extrinsics_2, width, height)
    return warped_features.squeeze(0), valid_mask

def warpModuleXYZ(config, camera, XYZ_1, features_2, extrinsics_1, extrinsics_2, width, height):
    """Warp one feature map to another view given camera pose and XYZ"""
    XYZ_shape = XYZ_1.shape
    numPlanes = int(XYZ_1.shape[2])

    XYZ_1 = XYZ_1.view((-1, 3))
    XYZ_2 = torch.matmul(torch.matmul(torch.cat([XYZ_1, torch.ones((len(XYZ_1), 1)).cuda()], dim=-1), extrinsics_1.inverse().transpose(0, 1)), extrinsics_2.transpose(0, 1))
    validMask = XYZ_2[:, 1] > 1e-4
    U = (XYZ_2[:, 0] / torch.clamp(XYZ_2[:, 1], min=1e-4) * camera[0] + camera[2]) / camera[4] * 2 - 1
    V = (-XYZ_2[:, 2] / torch.clamp(XYZ_2[:, 1], min=1e-4) * camera[1] + camera[3]) / camera[5] * 2 - 1

    padding = (width - height) // 2
    grids = torch.stack([U, V], dim=-1)

    validMask = (validMask) & (U >= -1) & (U <= 1) & (V >= -1) & (V <= 1)
    warped_features = F.grid_sample(features_2[:, :, padding:-padding], grids.unsqueeze(1).unsqueeze(0))
    numFeatureChannels = int(features_2.shape[1])
    warped_features = warped_features.view((numFeatureChannels, height, width, numPlanes)).transpose(2, 3).transpose(1, 2).transpose(0, 1).contiguous().view((-1, int(features_2.shape[1]), height, width))
    zeros = torch.zeros((numPlanes, numFeatureChannels, (width - height) // 2, width)).cuda()
    warped_features = torch.cat([zeros, warped_features, zeros], dim=2)
    validMask = validMask.view((numPlanes, height, width))
    validMask = torch.cat([zeros[:, 1], validMask.float(), zeros[:, 1]], dim=1)
    return warped_features, validMask


def calcXYZModule(config, camera, detections, masks, depth_np, return_individual=False, debug_type=0):
    """Compute a global coordinate map from plane detections"""
    ranges = config.getRanges(camera)
    ranges_ori = ranges
    zeros = torch.zeros(3, (config.IMAGE_MAX_DIM - config.IMAGE_MIN_DIM) // 2, config.IMAGE_MAX_DIM).cuda()        
    ranges = torch.cat([zeros, ranges.transpose(1, 2).transpose(0, 1), zeros], dim=1)
    XYZ_np = ranges * depth_np

    if len(detections) == 0:
        detection_mask = torch.zeros((config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM)).cuda()
        if return_individual:
            return XYZ_np, detection_mask, []
        else:
            return XYZ_np, detection_mask
        pass
    
    plane_parameters = detections[:, 6:9]
    
    XYZ = torch.ones((3, config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM)).cuda() * 10
    depthMask = torch.zeros((config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM)).cuda()
    planeXYZ = planeXYZModule(ranges_ori, plane_parameters, width=config.IMAGE_MAX_DIM, height=config.IMAGE_MIN_DIM)
    planeXYZ = planeXYZ.transpose(2, 3).transpose(1, 2).transpose(0, 1)
    zeros = torch.zeros(3, (config.IMAGE_MAX_DIM - config.IMAGE_MIN_DIM) // 2, config.IMAGE_MAX_DIM, int(planeXYZ.shape[-1])).cuda()
    planeXYZ = torch.cat([zeros, planeXYZ, zeros], dim=1)

    one_hot = True    
    if one_hot:
        for detectionIndex in range(len(detections)):
            mask = masks[detectionIndex]
            with torch.no_grad():
                mask_binary = torch.round(mask)
                pass
            if config.FITTING_TYPE >= 2:
                if (torch.norm(planeXYZ[:, :, :, detectionIndex] - XYZ_np, dim=0) * mask_binary).sum() / torch.clamp(mask_binary.sum(), min=1e-4) > 0.5:
                    mask_binary = torch.zeros(mask_binary.shape).cuda()
                    pass
                pass
            mask_binary = mask_binary * (planeXYZ[1, :, :, detectionIndex] < XYZ[1]).float()
            XYZ = planeXYZ[:, :, :, detectionIndex] * mask_binary + XYZ * (1 - mask_binary)
            depthMask = torch.max(depthMask, mask)
            continue
        XYZ = XYZ * torch.round(depthMask) + XYZ_np * (1 - torch.round(depthMask))
    else:
        background_mask = torch.clamp(1 - masks.sum(0, keepdim=True), min=0)
        all_masks = torch.cat([background_mask, masks], dim=0)
        all_XYZ = torch.cat([XYZ_np.unsqueeze(-1), planeXYZ], dim=-1)
        XYZ = (all_XYZ.transpose(2, 3).transpose(1, 2) * all_masks).sum(1)
        depthMask = torch.ones(depthMask.shape).cuda()
        pass

    if debug_type == 2:
        XYZ = XYZ_np
        pass

    if return_individual:
        return XYZ, depthMask, planeXYZ.transpose(2, 3).transpose(1, 2).transpose(0, 1)
    return XYZ, depthMask



class ConvBlock(torch.nn.Module):
    """The block consists of a convolution layer, an optional batch normalization layer, and a ReLU layer"""
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, output_padding=0, mode='conv', use_bn=True):
        super(ConvBlock, self).__init__()
        self.use_bn = use_bn
        if mode == 'conv':
            self.conv = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=not self.use_bn)
        elif mode == 'deconv':
            self.conv = torch.nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=not self.use_bn)
        elif mode == 'upsample':
            self.conv = torch.nn.Sequential(torch.nn.Upsample(scale_factor=stride, mode='nearest'), torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=padding, bias=not self.use_bn))
        elif mode == 'conv_3d':
            self.conv = torch.nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=not self.use_bn)
        elif mode == 'deconv_3d':
            self.conv = torch.nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=not self.use_bn)
        else:
            print('conv mode not supported', mode)
            exit(1)
            pass
        if '3d' not in mode:
            self.bn = torch.nn.BatchNorm2d(out_planes)
        else:
            self.bn = torch.nn.BatchNorm3d(out_planes)
            pass
        self.relu = torch.nn.ReLU(inplace=True)
        return
   
    def forward(self, inp):
        if self.use_bn:
            return self.relu(self.bn(self.conv(inp)))
        else:
            return self.relu(self.conv(inp))

class LinearBlock(torch.nn.Module):
    """The block consists of a linear layer and a ReLU layer"""    
    def __init__(self, in_planes, out_planes):
        super(LinearBlock, self).__init__()
        self.linear = torch.nn.Linear(in_planes, out_planes)
        self.relu = torch.nn.ReLU(inplace=True)
        return

    def forward(self, inp):
        return self.relu(self.linear(inp))       


def l2NormLossMask(pred, gt, mask, dim):
    """L2  loss with a mask"""
    return torch.sum(torch.norm(pred - gt, dim=dim) * mask) / torch.clamp(mask.sum(), min=1)

def l2LossMask(pred, gt, mask):
    """MSE with a mask"""    
    return torch.sum(torch.pow(pred - gt, 2) * mask) / torch.clamp(mask.sum(), min=1)

def l1LossMask(pred, gt, mask):
    """L1 loss with a mask"""        
    return torch.sum(torch.abs(pred - gt) * mask) / torch.clamp(mask.sum(), min=1)


def invertDepth(depth, inverse=False):
    """Invert depth or not"""
    if inverse:
        valid_mask = (depth > 1e-4).float()
        depth_inv = 1.0 / torch.clamp(depth, min=1e-4)
        return depth_inv * valid_mask
    else:
        return depth


class PlaneToDepth(torch.nn.Module):
    def __init__(self, normalized_K = True, normalized_flow = True, inverse_depth = True, W = 64, H = 48):

        super(PlaneToDepth, self).__init__()

        self.normalized_K = normalized_K
        self.normalized_flow = normalized_flow
        self.inverse_depth = inverse_depth

        with torch.no_grad():
            self.URANGE = ((torch.arange(W).float() + 0.5) / W).cuda().view((1, -1)).repeat(H, 1)
            self.VRANGE = ((torch.arange(H).float() + 0.5) / H).cuda().view((-1, 1)).repeat(1, W)
            self.ONES = torch.ones((H, W)).cuda()
            pass
        
    def forward(self, intrinsics, plane, return_XYZ=False):

        """
        :param K1: intrinsics of 1st image, 3x3
        :param K2: intrinsics of 2nd image, 3x3
        :param depth: depth map of first image, 1 x height x width
        :param rot: rotation from first to second image, 3
        :param trans: translation from first to second, 3
        :return: normalized flow from 1st image to 2nd image, 2 x height x width
        """

        with torch.no_grad():
            urange = (self.URANGE * intrinsics[4] - intrinsics[2]) / intrinsics[0]
            vrange = (self.VRANGE * intrinsics[5] - intrinsics[3]) / intrinsics[1]
            ranges = torch.stack([urange,
                                  self.ONES,
                                  -vrange], -1)
            pass

        planeOffset = torch.norm(plane, dim=-1)
        planeNormal = plane / torch.clamp(planeOffset.unsqueeze(-1), min=1e-4)
        depth = planeOffset / torch.clamp(torch.sum(ranges.unsqueeze(-2) * planeNormal, dim=-1), min=1e-4)
        depth = torch.clamp(depth, min=0, max=10)

        if self.inverse_depth:
            depth = invertDepth(depth)
        depth = depth.transpose(1, 2).transpose(0, 1)

        if return_XYZ:
            return depth, depth.unsqueeze(-1) * ranges
        return depth        

class PlaneToDepthLayer(torch.nn.Module):

    def __init__(self, normalized_K = False,  normalized_flow = True, inverse_depth = True):

        super(PlaneToDepthLayer, self).__init__()

        self.plane_to_depth = PlaneToDepth(normalized_K = normalized_K,
                                           normalized_flow = normalized_flow,
                                           inverse_depth = inverse_depth)

    def forward(self, intrinsics, plane, mask):

        """
        :param K1:  3x3 if shared_K is True, otherwise K1 is nx3x3
        :param K2:  3x3 if shared_K is True, otherwise K2 is nx3x3
        :param depth: n x 1 x h x w
        :param rot:   n x 3
        :param trans: n x3
        :param shared_K: if True, we share intrinsics for the depth images of the whole batch
        :return: n x 2 x h x w
        """

        batch_size = plane.size(0)

        depths = ()
        for i in range(batch_size):

            depth = self.plane_to_depth(intrinsics[i], plane[i], mask[i])
            depths += (depth, )
        depth = torch.stack(depths, 0)
        return depth    
