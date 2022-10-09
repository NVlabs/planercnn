"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""


import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
import sys
import os
import cv2
import copy
import glob

from models.model import *
from models.refinement_net import RefineModel
from models.modules import *
from datasets.plane_stereo_dataset import PlaneDataset
from datasets.inference_dataset import InferenceDataset
from datasets.nyu_dataset import NYUDataset
from utils import *
from visualize_utils import *
from evaluate_utils import *
from plane_utils import *
from options import parse_args
from config import InferenceConfig


class PlaneRCNNDetector():
    def __init__(self, options, config, modelType, checkpoint_dir=''):
        self.options = options
        self.config = config
        self.modelType = modelType
        self.model = MaskRCNN(config)
        self.model.cuda()
        self.model.eval()

        if modelType == 'basic':
            checkpoint_dir = checkpoint_dir if checkpoint_dir != '' else 'checkpoint/pair_' + options.anchorType
        elif modelType == 'pair':
            checkpoint_dir = checkpoint_dir if checkpoint_dir != '' else 'checkpoint/pair_' + options.anchorType
        elif modelType == 'refine':
            checkpoint_dir = checkpoint_dir if checkpoint_dir != '' else 'checkpoint/instance_' + options.anchorType
        elif modelType == 'refine_single':
            checkpoint_dir = checkpoint_dir if checkpoint_dir != '' else 'checkpoint/refinement_' + options.anchorType
        elif modelType == 'occlusion':
            checkpoint_dir = checkpoint_dir if checkpoint_dir != '' else 'checkpoint/plane_' + options.anchorType
        elif modelType == 'final':
            checkpoint_dir = checkpoint_dir if checkpoint_dir != '' else 'checkpoint/planercnn_' + options.anchorType
            pass

        if options.suffix != '':
            checkpoint_dir += '_' + options.suffix
            pass

        ## Indicates that the refinement network is trained separately        
        separate = modelType == 'refine'

        if not separate:
            if options.startEpoch >= 0:
                self.model.load_state_dict(torch.load(checkpoint_dir + '/checkpoint_' + str(options.startEpoch) + '.pth'))
            else:
                self.model.load_state_dict(torch.load(checkpoint_dir + '/checkpoint.pth'))
                pass
            pass

        if 'refine' in modelType or 'final' in modelType:
            self.refine_model = RefineModel(options)

            self.refine_model.cuda()
            self.refine_model.eval()
            if not separate:
                state_dict = torch.load(checkpoint_dir + '/checkpoint_refine.pth')
                self.refine_model.load_state_dict(state_dict)
                pass
            else:
                self.model.load_state_dict(torch.load('checkpoint/pair_' + options.anchorType + '_pair/checkpoint.pth'))
                self.refine_model.load_state_dict(torch.load('checkpoint/instance_normal_refine_mask_softmax_valid/checkpoint_refine.pth'))
                pass
            pass

        return

    def detect(self, sample):

        input_pair = []
        detection_pair = []
        camera = sample[30][0].cuda()
        for indexOffset in [0, ]:
            images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, gt_parameters, gt_depth, extrinsics, planes, gt_segmentation = sample[indexOffset + 0].cuda(), sample[indexOffset + 1].numpy(), sample[indexOffset + 2].cuda(), sample[indexOffset + 3].cuda(), sample[indexOffset + 4].cuda(), sample[indexOffset + 5].cuda(), sample[indexOffset + 6].cuda(), sample[indexOffset + 7].cuda(), sample[indexOffset + 8].cuda(), sample[indexOffset + 9].cuda(), sample[indexOffset + 10].cuda(), sample[indexOffset + 11].cuda()
            rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, target_parameters, mrcnn_parameters, detections, detection_masks, detection_gt_parameters, detection_gt_masks, rpn_rois, roi_features, roi_indices, depth_np_pred = self.model.predict([images, image_metas, gt_class_ids, gt_boxes, gt_masks, gt_parameters, camera], mode='inference_detection', use_nms=2, use_refinement=True)

            if len(detections) > 0:
                detections, detection_masks = unmoldDetections(self.config, camera, detections, detection_masks, depth_np_pred, debug=False)
                pass

            XYZ_pred, detection_mask, plane_XYZ = calcXYZModule(self.config, camera, detections, detection_masks, depth_np_pred, return_individual=True)
            detection_mask = detection_mask.unsqueeze(0)

            input_pair.append({'image': images, 'depth': gt_depth, 'mask': gt_masks, 'bbox': gt_boxes, 'extrinsics': extrinsics, 'segmentation': gt_segmentation, 'camera': camera})

            if 'nyu_dorn_only' in self.options.dataset:
                XYZ_pred[1:2] = sample[27].cuda()
                pass

            detection_pair.append({'XYZ': XYZ_pred, 'depth': XYZ_pred[1:2], 'mask': detection_mask, 'detection': detections, 'masks': detection_masks, 'depth_np': depth_np_pred, 'plane_XYZ': plane_XYZ})
            continue

        if ('refine' in self.modelType or 'refine' in self.options.suffix):
            pose = sample[26][0].cuda()
            pose = torch.cat([pose[0:3], pose[3:6] * pose[6]], dim=0)
            pose_gt = torch.cat([pose[0:1], -pose[2:3], pose[1:2], pose[3:4], -pose[5:6], pose[4:5]], dim=0).unsqueeze(0)
            camera = camera.unsqueeze(0)

            for c in range(1):
                detection_dict, input_dict = detection_pair[c], input_pair[c]

                new_input_dict = {k: v for k, v in input_dict.items()}
                new_input_dict['image'] = (input_dict['image'] + self.config.MEAN_PIXEL_TENSOR.view((-1, 1, 1))) / 255.0 - 0.5
                new_input_dict['image_2'] = (sample[13].cuda() + self.config.MEAN_PIXEL_TENSOR.view((-1, 1, 1))) / 255.0 - 0.5
                detections = detection_dict['detection']
                detection_masks = detection_dict['masks']
                depth_np = detection_dict['depth_np']
                image = new_input_dict['image']
                image_2 = new_input_dict['image_2']
                depth_gt = new_input_dict['depth'].unsqueeze(1)

                masks_inp = torch.cat([detection_masks.unsqueeze(1), detection_dict['plane_XYZ']], dim=1)

                segmentation = new_input_dict['segmentation']

                detection_masks = torch.nn.functional.interpolate(detection_masks[:, 80:560].unsqueeze(1), size=(192, 256), mode='nearest').squeeze(1)
                image = torch.nn.functional.interpolate(image[:, :, 80:560], size=(192, 256), mode='bilinear')
                image_2 = torch.nn.functional.interpolate(image_2[:, :, 80:560], size=(192, 256), mode='bilinear')
                masks_inp = torch.nn.functional.interpolate(masks_inp[:, :, 80:560], size=(192, 256), mode='bilinear')
                depth_np = torch.nn.functional.interpolate(depth_np[:, 80:560].unsqueeze(1), size=(192, 256), mode='bilinear').squeeze(1)
                plane_depth = torch.nn.functional.interpolate(detection_dict['depth'][:, 80:560].unsqueeze(1), size=(192, 256), mode='bilinear').squeeze(1)
                segmentation = torch.nn.functional.interpolate(segmentation[:, 80:560].unsqueeze(1).float(), size=(192, 256), mode='nearest').squeeze().long()

                new_input_dict['image'] = image
                new_input_dict['image_2'] = image_2

                results = self.refine_model(image, image_2, camera, masks_inp, detection_dict['detection'][:, 6:9], plane_depth, depth_np)

                masks = results[-1]['mask'].squeeze(1)

                all_masks = torch.softmax(masks, dim=0)

                masks_small = all_masks[1:]
                all_masks = torch.nn.functional.interpolate(all_masks.unsqueeze(1), size=(480, 640), mode='bilinear').squeeze(1)
                all_masks = (all_masks.max(0, keepdim=True)[1] == torch.arange(len(all_masks)).cuda().long().view((-1, 1, 1))).float()
                masks = all_masks[1:]
                detection_masks = torch.zeros(detection_dict['masks'].shape).cuda()
                detection_masks[:, 80:560] = masks


                detection_dict['masks'] = detection_masks
                detection_dict['depth_ori'] = detection_dict['depth'].clone()
                detection_dict['mask'][:, 80:560] = (masks.max(0, keepdim=True)[0] > (1 - masks.sum(0, keepdim=True))).float()

                if self.options.modelType == 'fitting':
                    masks_cropped = masks_small
                    ranges = self.config.getRanges(camera).transpose(1, 2).transpose(0, 1)
                    XYZ = torch.nn.functional.interpolate(ranges.unsqueeze(1), size=(192, 256), mode='bilinear').squeeze(1) * results[-1]['depth'].squeeze(1)
                    detection_areas = masks_cropped.sum(-1).sum(-1)
                    A = masks_cropped.unsqueeze(1) * XYZ
                    b = masks_cropped
                    Ab = (A * b.unsqueeze(1)).sum(-1).sum(-1)
                    AA = (A.unsqueeze(2) * A.unsqueeze(1)).sum(-1).sum(-1)
                    plane_parameters = torch.stack([torch.matmul(torch.inverse(AA[planeIndex]), Ab[planeIndex]) if detection_areas[planeIndex] else detection_dict['detection'][planeIndex, 6:9] for planeIndex in range(len(AA))], dim=0)
                    plane_offsets = torch.norm(plane_parameters, dim=-1, keepdim=True)
                    plane_parameters = plane_parameters / torch.clamp(torch.pow(plane_offsets, 2), 1e-4)
                    detection_dict['detection'][:, 6:9] = plane_parameters

                    XYZ_pred, detection_mask, plane_XYZ = calcXYZModule(self.config, camera, detection_dict['detection'], detection_masks, detection_dict['depth'], return_individual=True)
                    detection_dict['depth'] = XYZ_pred[1:2]
                    pass
                continue
            pass
        return detection_pair

class DepthDetector():
    def __init__(self, options, config, modelType, checkpoint_dir=''):
        self.options = options
        self.config = config
        self.modelType = modelType

        self.model = MaskRCNNDepth(config)
        self.model.cuda()
        self.model.eval()

        checkpoint_dir = checkpoint_dir if checkpoint_dir != '' else 'checkpoint/depth_np'
        if options.suffix != '':
            checkpoint_dir += '_' + options.suffix
            pass
        self.model.load_state_dict(torch.load(checkpoint_dir + '/checkpoint.pth'))
        return

    def detect(self, sample):
        detection_pair = []
        camera = sample[30][0].cuda()
        for indexOffset in [0, ]:
            images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, gt_parameters, gt_depth, extrinsics, planes, gt_segmentation = sample[indexOffset + 0].cuda(), sample[indexOffset + 1].numpy(), sample[indexOffset + 2].cuda(), sample[indexOffset + 3].cuda(), sample[indexOffset + 4].cuda(), sample[indexOffset + 5].cuda(), sample[indexOffset + 6].cuda(), sample[indexOffset + 7].cuda(), sample[indexOffset + 8].cuda(), sample[indexOffset + 9].cuda(), sample[indexOffset + 10].cuda(), sample[indexOffset + 11].cuda()

            depth_np_pred = self.model.predict([images, camera], mode='inference_detection', use_nms=2, use_refinement='refinement' in self.options.suffix)

            if depth_np_pred.shape != gt_depth.shape:
                depth_np_pred = torch.nn.functional.interpolate(depth_np_pred.unsqueeze(1), size=(640, 640), mode='bilinear').squeeze(1)
                pass
            detection_pair.append({'depth': depth_np_pred, 'mask': torch.ones(depth_np_pred.shape).cuda()})
            continue
        return detection_pair


class PlaneNetDetector():
    def __init__(self, options, config, checkpoint_dir=''):
        self.options = options
        self.config = config
        sys.path.append('../../existing_methods/')
        from PlaneNet.planenet_inference import PlaneNetDetector
        self.detector = PlaneNetDetector(predictNYU=False)
        return

    def detect(self, sample):

        detection_pair = []
        for indexOffset in [0, ]:
            images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, gt_parameters, gt_depth, extrinsics, planes, gt_segmentation = sample[indexOffset + 0].cuda(), sample[indexOffset + 1].numpy(), sample[indexOffset + 2].cuda(), sample[indexOffset + 3].cuda(), sample[indexOffset + 4].cuda(), sample[indexOffset + 5].cuda(), sample[indexOffset + 6].cuda(), sample[indexOffset + 7].cuda(), sample[indexOffset + 8].cuda(), sample[indexOffset + 9].cuda(), sample[indexOffset + 10].cuda(), sample[indexOffset + 11].cuda()

            image = (images[0].detach().cpu().numpy().transpose((1, 2, 0)) + self.config.MEAN_PIXEL)[80:560]

            pred_dict = self.detector.detect(image)
            segmentation = pred_dict['segmentation']
            segmentation = np.concatenate([np.full((80, 640), fill_value=-1, dtype=np.int32), segmentation, np.full((80, 640), fill_value=-1, dtype=np.int32)], axis=0)

            planes = pred_dict['plane']

            masks = (segmentation == np.arange(len(planes), dtype=np.int32).reshape((-1, 1, 1))).astype(np.float32)
            depth = pred_dict['depth']
            depth = np.concatenate([np.zeros((80, 640), dtype=np.int32), depth, np.zeros((80, 640), dtype=np.int32)], axis=0)
            detections = np.concatenate([np.ones((len(planes), 4)), np.ones((len(planes), 2)), planes], axis=-1)

            detections = torch.from_numpy(detections).float().cuda()
            depth = torch.from_numpy(depth).unsqueeze(0).float().cuda()
            masks = torch.from_numpy(masks).float().cuda()
            detection_pair.append({'depth': depth, 'mask': masks.sum(0, keepdim=True), 'masks': masks, 'detection': detections})
            continue
        return detection_pair


class PlaneRecoverDetector():
    def __init__(self, options, config, checkpoint_dir=''):
        self.options = options
        self.config = config
        sys.path.append('../../existing_methods/')
        from planerecover_ori.inference import PlaneRecoverDetector
        self.detector = PlaneRecoverDetector()
        return

    def detect(self, sample):

        detection_pair = []
        camera = sample[30][0].cuda()
        for indexOffset in [0, ]:
            images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, gt_parameters, gt_depth, extrinsics, planes, gt_segmentation = sample[indexOffset + 0].cuda(), sample[indexOffset + 1].numpy(), sample[indexOffset + 2].cuda(), sample[indexOffset + 3].cuda(), sample[indexOffset + 4].cuda(), sample[indexOffset + 5].cuda(), sample[indexOffset + 6].cuda(), sample[indexOffset + 7].cuda(), sample[indexOffset + 8].cuda(), sample[indexOffset + 9].cuda(), sample[indexOffset + 10].cuda(), sample[indexOffset + 11].cuda()

            image = (images[0].detach().cpu().numpy().transpose((1, 2, 0)) + self.config.MEAN_PIXEL)[80:560]

            pred_dict = self.detector.detect(image)
            segmentation = pred_dict['segmentation']
            segmentation = np.concatenate([np.full((80, 640), fill_value=-1, dtype=np.int32), segmentation, np.full((80, 640), fill_value=-1, dtype=np.int32)], axis=0)

            planes = pred_dict['plane']

            masks = (segmentation == np.arange(len(planes), dtype=np.int32).reshape((-1, 1, 1))).astype(np.float32)
            detections = np.concatenate([np.ones((len(planes), 4)), np.ones((len(planes), 2)), planes], axis=-1)

            detections = torch.from_numpy(detections).float().cuda()
            masks = torch.from_numpy(masks).float().cuda()
            XYZ_pred, detection_mask, plane_XYZ = calcXYZModule(self.config, camera, detections, masks, torch.zeros((1, 640, 640)).cuda(), return_individual=True)
            depth = XYZ_pred[1:2]
            print(planes)
            print(np.unique(segmentation))
            for mask_index, mask in enumerate(masks.detach().cpu().numpy()):
                cv2.imwrite('test/mask_' + str(mask_index) + '.png', drawMaskImage(mask))
                continue
            detection_pair.append({'depth': depth, 'mask': masks.sum(0, keepdim=True), 'masks': masks, 'detection': detections})
            continue
        return detection_pair


class TraditionalDetector():
    def __init__(self, options, config, modelType=''):
        self.options = options
        self.config = config
        self.modelType = modelType
        if 'pred' in modelType:
            sys.path.append('../../')
            from PlaneNet.planenet_inference import PlaneNetDetector
            self.detector = PlaneNetDetector(predictSemantics=True)
            pass
        return

    def detect(self, sample):
        detection_pair = []
        for indexOffset in [0, ]:
            images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, gt_parameters, gt_depth, extrinsics, planes, gt_segmentation, gt_semantics = sample[indexOffset + 0].cuda(), sample[indexOffset + 1].numpy(), sample[indexOffset + 2].cuda(), sample[indexOffset + 3].cuda(), sample[indexOffset + 4].cuda(), sample[indexOffset + 5].cuda(), sample[indexOffset + 6].cuda(), sample[indexOffset + 7].cuda(), sample[indexOffset + 8].cuda(), sample[indexOffset + 9].cuda(), sample[indexOffset + 10].cuda(), sample[indexOffset + 11].cuda(), sample[indexOffset + 12].cuda()

            image = (images[0].detach().cpu().numpy().transpose((1, 2, 0)) + self.config.MEAN_PIXEL)[80:560]

            input_dict = {'image': cv2.resize(image, (256, 192))}

            if 'gt' in self.modelType:
                input_dict['depth'] = cv2.resize(gt_depth[0].detach().cpu().numpy()[80:560], (256, 192))
                semantics = gt_semantics[0].detach().cpu().numpy()[80:560]
                input_dict['semantics'] = cv2.resize(semantics, (256, 192), interpolation=cv2.INTER_NEAREST)
            else:
                pred_dict = self.detector.detect(image)
                input_dict['depth'] = pred_dict['non_plane_depth'].squeeze()
                input_dict['semantics'] = pred_dict['semantics'].squeeze().argmax(-1)
                pass

            camera = sample[30][0].numpy()
            input_dict['info'] = np.array([camera[0], 0, camera[2], 0, 0, camera[1], camera[3], 0, 0, 0, 1, 0, 0, 0, 0, 1, camera[4], camera[5], 1000, 0])
            np.save('test/input_dict.npy', input_dict)
            os.system('rm test/output_dict.npy')
            os.system('python plane_utils.py ' + self.modelType)
            output_dict = np.load('test/output_dict.npy', encoding='latin1')[()]

            segmentation = cv2.resize(output_dict['segmentation'], (640, 480), interpolation=cv2.INTER_NEAREST)
            segmentation = np.concatenate([np.full((80, 640), fill_value=-1, dtype=np.int32), segmentation, np.full((80, 640), fill_value=-1, dtype=np.int32)], axis=0)

            planes = output_dict['plane']
            masks = (segmentation == np.arange(len(planes), dtype=np.int32).reshape((-1, 1, 1))).astype(np.float32)
            plane_depths = calcPlaneDepths(planes, 256, 192, camera, max_depth=10)
            depth = (plane_depths * (np.expand_dims(output_dict['segmentation'], -1) == np.arange(len(planes)))).sum(-1)
            depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_LINEAR)
            depth = np.concatenate([np.zeros((80, 640)), depth, np.zeros((80, 640))], axis=0)
            detections = np.concatenate([np.ones((len(planes), 4)), np.ones((len(planes), 2)), planes], axis=-1)

            detections = torch.from_numpy(detections).float().cuda()
            depth = torch.from_numpy(depth).unsqueeze(0).float().cuda()
            masks = torch.from_numpy(masks).float().cuda()
            detection_pair.append({'depth': depth, 'mask': masks.sum(0, keepdim=True), 'masks': masks, 'detection': detections})
            continue
        return detection_pair


def evaluate(options):
    config = InferenceConfig(options)
    config.FITTING_TYPE = options.numAnchorPlanes

    if options.dataset == '':
        dataset = PlaneDataset(options, config, split='test', random=False, load_semantics=False)
    elif options.dataset == 'occlusion':
        config_dataset = copy.deepcopy(config)
        config_dataset.OCCLUSION = False
        dataset = PlaneDataset(options, config_dataset, split='test', random=False, load_semantics=True)
    elif 'nyu' in options.dataset:
        dataset = NYUDataset(options, config, split='val', random=False)
    elif options.dataset == 'synthia':
        dataset = SynthiaDataset(options, config, split='val', random=False)
    elif options.dataset == 'kitti':
        camera = np.zeros(6)
        camera[0] = 9.842439e+02
        camera[1] = 9.808141e+02
        camera[2] = 6.900000e+02
        camera[3] = 2.331966e+02
        camera[4] = 1242
        camera[5] = 375
        dataset = InferenceDataset(options, config, image_list=glob.glob('../../Data/KITTI/scene_3/*.png'), camera=camera)
    elif options.dataset == '7scene':
        camera = np.zeros(6)
        camera[0] = 519
        camera[1] = 519
        camera[2] = 320
        camera[3] = 240
        camera[4] = 640
        camera[5] = 480
        dataset = InferenceDataset(options, config, image_list=glob.glob('../../Data/SevenScene/scene_3/*.png'), camera=camera)
    elif options.dataset == 'tanktemple':
        camera = np.zeros(6)
        camera[0] = 0.7
        camera[1] = 0.7
        camera[2] = 0.5
        camera[3] = 0.5
        camera[4] = 1
        camera[5] = 1
        dataset = InferenceDataset(options, config, image_list=glob.glob('../../Data/TankAndTemple/scene_4/*.jpg'), camera=camera)
    elif options.dataset == 'make3d':
        camera = np.zeros(6)
        camera[0] = 0.7
        camera[1] = 0.7
        camera[2] = 0.5
        camera[3] = 0.5
        camera[4] = 1
        camera[5] = 1
        dataset = InferenceDataset(options, config, image_list=glob.glob('../../Data/Make3D/*.jpg'), camera=camera)
    elif options.dataset == 'popup':
        camera = np.zeros(6)
        camera[0] = 0.7
        camera[1] = 0.7
        camera[2] = 0.5
        camera[3] = 0.5
        camera[4] = 1
        camera[5] = 1
        dataset = InferenceDataset(options, config, image_list=glob.glob('../../Data/PhotoPopup/*.jpg'), camera=camera)
    elif options.dataset == 'cross' or options.dataset == 'cross_2':
        image_list = ['test/cross_dataset/' + str(c) + '_image.png' for c in range(12)]
        cameras = []
        camera = np.zeros(6)
        camera[0] = 587
        camera[1] = 587
        camera[2] = 320
        camera[3] = 240
        camera[4] = 640
        camera[5] = 480
        for c in range(4):
            cameras.append(camera)
            continue
        camera_kitti = np.zeros(6)
        camera_kitti[0] = 9.842439e+02
        camera_kitti[1] = 9.808141e+02
        camera_kitti[2] = 6.900000e+02
        camera_kitti[3] = 2.331966e+02
        camera_kitti[4] = 1242.0
        camera_kitti[5] = 375.0
        for c in range(2):
            cameras.append(camera_kitti)
            continue
        camera_synthia = np.zeros(6)
        camera_synthia[0] = 133.185088
        camera_synthia[1] = 134.587036
        camera_synthia[2] = 160.000000
        camera_synthia[3] = 96.000000
        camera_synthia[4] = 320
        camera_synthia[5] = 192
        for c in range(2):
            cameras.append(camera_synthia)
            continue
        camera_tanktemple = np.zeros(6)
        camera_tanktemple[0] = 0.7
        camera_tanktemple[1] = 0.7
        camera_tanktemple[2] = 0.5
        camera_tanktemple[3] = 0.5
        camera_tanktemple[4] = 1
        camera_tanktemple[5] = 1
        for c in range(2):
            cameras.append(camera_tanktemple)
            continue
        for c in range(2):
            cameras.append(camera)
            continue
        dataset = InferenceDataset(options, config, image_list=image_list, camera=cameras)
    elif options.dataset == 'selected':
        image_list = glob.glob('test/selected_images/*_image_0.png')
        image_list = [filename for filename in image_list if '63_image' not in filename and '77_image' not in filename] + [filename for filename in image_list if '63_image' in filename or '77_image' in filename]
        camera = np.zeros(6)
        camera[0] = 587
        camera[1] = 587
        camera[2] = 320
        camera[3] = 240
        camera[4] = 640
        camera[5] = 480
        dataset = InferenceDataset(options, config, image_list=image_list, camera=camera)
    elif options.dataset == 'comparison':
        image_list = ['test/comparison/' + str(index) + '_image_0.png' for index in [65, 11, 24]]
        camera = np.zeros(6)
        camera[0] = 587
        camera[1] = 587
        camera[2] = 320
        camera[3] = 240
        camera[4] = 640
        camera[5] = 480
        dataset = InferenceDataset(options, config, image_list=image_list, camera=camera)
    elif 'inference' in options.dataset:
        image_list = glob.glob(options.customDataFolder + '/*.png') + glob.glob(options.customDataFolder + '/*.jpg')
        if os.path.exists(options.customDataFolder + '/camera.txt'):
            camera = np.zeros(6)
            with open(options.customDataFolder + '/camera.txt', 'r') as f:
                for line in f:
                    values = [float(token.strip()) for token in line.split(' ') if token.strip() != '']
                    for c in range(6):
                        camera[c] = values[c]
                        continue
                    break
                pass
        else:
            camera = [filename.replace('.png', '.txt').replace('.jpg', '.txt') for filename in image_list]
            pass
        dataset = InferenceDataset(options, config, image_list=image_list, camera=camera)
        pass

    print('the number of images', len(dataset))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    epoch_losses = []
    data_iterator = tqdm(dataloader, total=len(dataset))

    specified_suffix = options.suffix
    with torch.no_grad():
        detectors = []
        for method in options.methods:
            if method == 'w':
                options.suffix = 'pair_' + specified_suffix if specified_suffix != '' else 'pair'
                detectors.append(('warping', PlaneRCNNDetector(options, config, modelType='pair')))
            elif method == 'b':
                options.suffix = specified_suffix if specified_suffix != '' else ''
                detectors.append(('basic', PlaneRCNNDetector(options, config, modelType='pair')))
            elif method == 'o':
                options.suffix = 'occlusion_' + specified_suffix if specified_suffix != '' else 'occlusion'
                detectors.append(('occlusion', PlaneRCNNDetector(options, config, modelType='occlusion')))
            elif method == 'p':
                detectors.append(('planenet', PlaneNetDetector(options, config)))
            elif method == 'e':
                detectors.append(('planerecover', PlaneRecoverDetector(options, config)))
            elif method == 't':
                if 'gt' in options.suffix:
                    detectors.append(('manhattan_gt', TraditionalDetector(options, config, 'manhattan_gt')))
                else:
                    detectors.append(('manhattan_pred', TraditionalDetector(options, config, 'manhattan_pred')))
                    pass
            elif method == 'n':
                options.suffix = specified_suffix if specified_suffix != '' else ''
                detectors.append(('non_planar', DepthDetector(options, config, modelType='np')))
            elif method == 'r':
                options.suffix = specified_suffix if specified_suffix != '' else ''
                detectors.append(('refine', PlaneRCNNDetector(options, config, modelType='refine')))
            elif method == 's':
                options.suffix = specified_suffix if specified_suffix != '' else ''
                detectors.append(('refine_single', PlaneRCNNDetector(options, config, modelType='refine_single')))
            elif method == 'f':
                options.suffix = specified_suffix if specified_suffix != '' else ''
                detectors.append(('final', PlaneRCNNDetector(options, config, modelType='final')))
                pass
            continue
        pass

    if not options.debug:
        for method_name in [detector[0] for detector in detectors]:
            os.system('rm ' + options.test_dir + '/*_' + method_name + '.png')
            continue
        pass

    all_statistics = []
    for name, detector in detectors:
        statistics = [[], [], [], []]
        for sampleIndex, sample in enumerate(data_iterator):
            if options.testingIndex >= 0 and sampleIndex != options.testingIndex:
                if sampleIndex > options.testingIndex:
                    break
                continue
            input_pair = []
            camera = sample[30][0].cuda()
            for indexOffset in [0, ]:
                images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, gt_parameters, gt_depth, extrinsics, planes, gt_segmentation = sample[indexOffset + 0].cuda(), sample[indexOffset + 1].numpy(), sample[indexOffset + 2].cuda(), sample[indexOffset + 3].cuda(), sample[indexOffset + 4].cuda(), sample[indexOffset + 5].cuda(), sample[indexOffset + 6].cuda(), sample[indexOffset + 7].cuda(), sample[indexOffset + 8].cuda(), sample[indexOffset + 9].cuda(), sample[indexOffset + 10].cuda(), sample[indexOffset + 11].cuda()

                masks = (gt_segmentation == torch.arange(gt_segmentation.max() + 1).cuda().view(-1, 1, 1)).float()
                input_pair.append({'image': images, 'depth': gt_depth, 'bbox': gt_boxes, 'extrinsics': extrinsics, 'segmentation': gt_segmentation, 'camera': camera, 'plane': planes[0], 'masks': masks, 'mask': gt_masks})
                continue

            if sampleIndex >= options.numTestingImages:
                break

            with torch.no_grad():
                detection_pair = detector.detect(sample)
                pass

            if options.dataset == 'rob':
                depth = detection_pair[0]['depth'].squeeze().detach().cpu().numpy()
                os.system('rm ' + image_list[sampleIndex].replace('color', 'depth'))
                depth_rounded = np.round(depth * 256)
                depth_rounded[np.logical_or(depth_rounded < 0, depth_rounded >= 256 * 256)] = 0
                cv2.imwrite(image_list[sampleIndex].replace('color', 'depth').replace('jpg', 'png'), depth_rounded.astype(np.uint16))
                continue


            if 'inference' not in options.dataset:
                for c in range(len(input_pair)):
                    evaluateBatchDetection(options, config, input_pair[c], detection_pair[c], statistics=statistics, printInfo=options.debug, evaluate_plane=options.dataset == '')
                    continue
            else:
                for c in range(len(detection_pair)):
                    np.save(options.test_dir + '/' + str(sampleIndex % 500) + '_plane_parameters_' + str(c) + '.npy', detection_pair[c]['detection'][:, 6:9].cpu())
                    np.save(options.test_dir + '/' + str(sampleIndex % 500) + '_plane_masks_' + str(c) + '.npy', detection_pair[c]['masks'][:, 80:560].cpu())
                    continue
                pass
                            
            if sampleIndex < 30 or options.debug or options.dataset != '':
                visualizeBatchPair(options, config, input_pair, detection_pair, indexOffset=sampleIndex % 500, suffix='_' + name + options.modelType, write_ply=options.testingIndex >= 0, write_new_view=options.testingIndex >= 0 and 'occlusion' in options.suffix)
                pass
            if sampleIndex >= options.numTestingImages:
                break
            continue
        if 'inference' not in options.dataset:
            options.keyname = name
            printStatisticsDetection(options, statistics)
            all_statistics.append(statistics)
            pass
        continue
    if 'inference' not in options.dataset:
        if options.debug and len(detectors) > 1:
            all_statistics = np.concatenate([np.arange(len(all_statistics[0][0])).reshape((-1, 1)), ] + [np.array(statistics[3]) for statistics in all_statistics], axis=-1)
            print(all_statistics.astype(np.int32))
            pass
        if options.testingIndex == -1:
            np.save('logs/all_statistics.npy', all_statistics)
            pass
        pass
    return

if __name__ == '__main__':
    args = parse_args()

    if args.dataset == '':
        args.keyname = 'evaluate'
    else:
        args.keyname = args.dataset
        pass
    args.test_dir = 'test/' + args.keyname

    if args.testingIndex >= 0:
        args.debug = True
        pass
    if args.debug:
        args.test_dir += '_debug'
        args.printInfo = True
        pass

    ## Write html for visualization
    if False:
        if False:
            info_list = ['image_0', 'segmentation_0', 'segmentation_0_warping', 'depth_0', 'depth_0_warping']
            writeHTML(args.test_dir, info_list, numImages=100, convertToImage=False, filename='index', image_width=256)
            pass
        if False:
            info_list = ['image_0', 'segmentation_0', 'detection_0_planenet', 'detection_0_warping', 'detection_0_refine']
            writeHTML(args.test_dir, info_list, numImages=20, convertToImage=True, filename='comparison_segmentation')
            pass
        if False:
            info_list = ['image_0', 'segmentation_0', 'segmentation_0_manhattan_gt', 'segmentation_0_planenet', 'segmentation_0_warping']
            writeHTML(args.test_dir, info_list, numImages=30, convertToImage=False, filename='comparison_segmentation')
            pass
        exit(1)
        pass

    if not os.path.exists(args.test_dir):
        os.system("mkdir -p %s"%args.test_dir)
        pass

    if args.debug and args.dataset == '':
        os.system('rm ' + args.test_dir + '/*')
        pass

    evaluate(args)
