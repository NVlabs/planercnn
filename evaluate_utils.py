"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import cv2
import numpy as np
from utils import *
from models.modules import *


def evaluateDepths(predDepths, gtDepths, printInfo=False):
    """Evaluate depth reconstruction accuracy"""
    
    masks = gtDepths > 1e-4
    
    numPixels = float(masks.sum())
    
    rmse = np.sqrt((pow(predDepths - gtDepths, 2) * masks).sum() / numPixels)
    rmse_log = np.sqrt((pow(np.log(np.maximum(predDepths, 1e-4)) - np.log(np.maximum(gtDepths, 1e-4)), 2) * masks).sum() / numPixels)
    log10 = (np.abs(np.log10(np.maximum(predDepths, 1e-4)) - np.log10(np.maximum(gtDepths, 1e-4))) * masks).sum() / numPixels
    rel = (np.abs(predDepths - gtDepths) / np.maximum(gtDepths, 1e-4) * masks).sum() / numPixels
    rel_sqr = (pow(predDepths - gtDepths, 2) / np.maximum(gtDepths, 1e-4) * masks).sum() / numPixels    
    deltas = np.maximum(predDepths / np.maximum(gtDepths, 1e-4), gtDepths / np.maximum(predDepths, 1e-4)) + (1 - masks.astype(np.float32)) * 10000
    accuracy_1 = (deltas < 1.25).sum() / numPixels
    accuracy_2 = (deltas < pow(1.25, 2)).sum() / numPixels
    accuracy_3 = (deltas < pow(1.25, 3)).sum() / numPixels
    if printInfo:
        print(('depth statistics', rel, rel_sqr, log10, rmse, rmse_log, accuracy_1, accuracy_2, accuracy_3))
        pass
    return [rel, rel_sqr, log10, rmse, rmse_log, accuracy_1, accuracy_2, accuracy_3]



def evaluatePlanesTensor(input_dict, detection_dict, printInfo=False, use_gpu=True):
    """Evaluate plane detection accuracy in terms of Average Precision"""
    if use_gpu:
        masks_pred, masks_gt, depth_pred, depth_gt = detection_dict['masks'], input_dict['masks'], detection_dict['depth'], input_dict['depth']
    else:
        masks_pred, masks_gt, depth_pred, depth_gt = detection_dict['masks'].cpu(), input_dict['masks'].cpu(), detection_dict['depth'].cpu(), input_dict['depth'].cpu()
        pass

    
    masks_pred = torch.round(masks_pred)
    
    plane_areas = masks_gt.sum(dim=1).sum(dim=1)
    masks_intersection = (masks_gt.unsqueeze(1) * (masks_pred.unsqueeze(0))).float()
    intersection_areas = masks_intersection.sum(2).sum(2)

    depth_diff = torch.abs(depth_gt - depth_pred)
    depth_diff[depth_gt < 1e-4] = 0

    depths_diff = (depth_diff * masks_intersection).sum(2).sum(2) / torch.clamp(intersection_areas, min=1e-4)
    depths_diff[intersection_areas < 1e-4] = 1000000
    
    union = ((masks_gt.unsqueeze(1) + masks_pred.unsqueeze(0)) > 0.5).float().sum(2).sum(2)
    plane_IOUs = intersection_areas / torch.clamp(union, min=1e-4)

    plane_IOUs = plane_IOUs.detach().cpu().numpy()
    depths_diff = depths_diff.detach().cpu().numpy()
    plane_areas = plane_areas.detach().cpu().numpy()
    intersection_areas = intersection_areas.detach().cpu().numpy()

    num_plane_pixels = plane_areas.sum()
        
    pixel_curves = []
    plane_curves = []

    for IOU_threshold in [0.5, ]:
        IOU_mask = (plane_IOUs > IOU_threshold).astype(np.float32)
        min_diff = np.min(depths_diff * IOU_mask + 1e6 * (1 - IOU_mask), axis=1)
        stride = 0.05
        plane_recall = []
        pixel_recall = []
        for step in range(21):
            diff_threshold = step * stride
            pixel_recall.append(np.minimum((intersection_areas * ((depths_diff <= diff_threshold).astype(np.float32) * IOU_mask)).sum(1), plane_areas).sum() / num_plane_pixels)
            
            plane_recall.append(float((min_diff <= diff_threshold).sum()) / len(masks_gt))
            continue
        pixel_curves.append(pixel_recall)
        plane_curves.append(plane_recall)
        continue

    APs = []
    for diff_threshold in [0.2, 0.3, 0.6, 0.9]:
        correct_mask = np.minimum((depths_diff < diff_threshold), (plane_IOUs > 0.5))
        match_mask = np.zeros(len(correct_mask), dtype=np.bool)
        recalls = []
        precisions = []
        num_predictions = correct_mask.shape[-1]
        num_targets = (plane_areas > 0).sum()
        for rank in range(num_predictions):
            match_mask = np.maximum(match_mask, correct_mask[:, rank])
            num_matches = match_mask.sum()
            precisions.append(float(num_matches) / (rank + 1))
            recalls.append(float(num_matches) / num_targets)
            continue
        max_precision = 0.0
        prev_recall = 1.0
        AP = 0.0
        for recall, precision in zip(recalls[::-1], precisions[::-1]):
            AP += (prev_recall - recall) * max_precision
            max_precision = max(max_precision, precision)
            prev_recall = recall
            continue
        AP += prev_recall * max_precision
        APs.append(AP)
        continue    

    detection_dict['flag'] = correct_mask.max(0)
    input_dict['flag'] = correct_mask.max(1)
    
    if printInfo:
        print('plane statistics', correct_mask.max(-1).sum(), num_targets, num_predictions)
        pass
    return APs + plane_curves[0] + pixel_curves[0]

def evaluatePlaneDepth(config, input_dict, detection_dict, printInfo=False):
    masks_gt, depth_pred, depth_gt = input_dict['masks'], detection_dict['depth'], input_dict['depth']
    masks_cropped = masks_gt[:, 80:560]
    ranges = config.getRanges(input_dict['camera']).transpose(1, 2).transpose(0, 1)
    plane_parameters_array = []
    for depth in [depth_pred, depth_gt]:
        XYZ = ranges * depth[:, 80:560]
        A = masks_cropped.unsqueeze(1) * XYZ
        b = masks_cropped
        Ab = (A * b.unsqueeze(1)).sum(-1).sum(-1)
        AA = (A.unsqueeze(2) * A.unsqueeze(1)).sum(-1).sum(-1)
        plane_parameters = torch.stack([torch.matmul(torch.inverse(AA[planeIndex]), Ab[planeIndex]) for planeIndex in range(len(AA))], dim=0)
        plane_offsets = torch.norm(plane_parameters, dim=-1, keepdim=True)
        plane_parameters = plane_parameters / torch.clamp(torch.pow(plane_offsets, 2), 1e-4)
        plane_parameters_array.append(plane_parameters)
        continue

    plane_diff = torch.norm(plane_parameters_array[0] - plane_parameters_array[1], dim=-1)
    plane_areas = masks_gt.sum(-1).sum(-1)

    statistics = [plane_diff.mean(), (plane_diff * plane_areas).sum() / plane_areas.sum()]
    if printInfo:
        print('plane statistics', statistics)
        pass    
    return statistics

def evaluateMask(predMasks, gtMasks, printInfo=False):
    predMasks = predMasks[:, 80:560]
    gtMasks = gtMasks[:, 80:560]
    intersection = np.minimum(predMasks, gtMasks).sum()
    info = [intersection / max(predMasks.sum(), 1), intersection / max(gtMasks.sum(), 1)]
    if printInfo:
        print('mask statistics', info)
        pass
    return info

def evaluateMasksTensor(predMasks, gtMasks, valid_mask, printInfo=False):
    gtMasks = torch.cat([gtMasks, torch.clamp(1 - gtMasks.sum(0, keepdim=True), min=0)], dim=0)
    predMasks = torch.cat([predMasks, torch.clamp(1 - predMasks.sum(0, keepdim=True), min=0)], dim=0)
    intersection = (gtMasks.unsqueeze(1) * predMasks * valid_mask).sum(-1).sum(-1).float()
    union = (torch.max(gtMasks.unsqueeze(1), predMasks) * valid_mask).sum(-1).sum(-1).float()    

    N = intersection.sum()
    
    RI = 1 - ((intersection.sum(0).pow(2).sum() + intersection.sum(1).pow(2).sum()) / 2 - intersection.pow(2).sum()) / (N * (N - 1) / 2)
    joint = intersection / N
    marginal_2 = joint.sum(0)
    marginal_1 = joint.sum(1)
    H_1 = (-marginal_1 * torch.log2(marginal_1 + (marginal_1 == 0).float())).sum()
    H_2 = (-marginal_2 * torch.log2(marginal_2 + (marginal_2 == 0).float())).sum()

    B = (marginal_1.unsqueeze(-1) * marginal_2)
    log2_quotient = torch.log2(torch.clamp(joint, 1e-8) / torch.clamp(B, 1e-8)) * (torch.min(joint, B) > 1e-8).float()
    MI = (joint * log2_quotient).sum()
    voi = H_1 + H_2 - 2 * MI

    IOU = intersection / torch.clamp(union, min=1)
    SC = ((IOU.max(-1)[0] * torch.clamp((gtMasks * valid_mask).sum(-1).sum(-1), min=1e-4)).sum() / N + (IOU.max(0)[0] * torch.clamp((predMasks * valid_mask).sum(-1).sum(-1), min=1e-4)).sum() / N) / 2
    info = [RI, voi, SC]
    if printInfo:
        print('mask statistics', info)
        pass
    return info

def evaluateBatchDeMoN(options, gt_dict, pred_dict, statistics = [[], [], []]):
    for batchIndex in range(len(gt_dict['depth'])):
        statistics[0].append(evaluateDepthRelative(pred_dict['depth'][batchIndex], gt_dict['depth'][batchIndex], np.linalg.norm(gt_dict['translation'][batchIndex])))
        statistics[1].append(evaluateRotation(pred_dict['rotation'][batchIndex], gt_dict['rotation'][batchIndex]))
        statistics[2].append(evaluateTranslation(pred_dict['translation'][batchIndex], gt_dict['translation'][batchIndex]))
        continue
    return


def evaluateBatchDetection(options, config, input_dict, detection_dict, statistics=[[], [], [], []], debug_dict={}, printInfo=False, evaluate_plane=False):
    if 'depth' in debug_dict:
        planes = fitPlanesModule(config, debug_dict['depth'][80:560], detection_dict['masks'][:, 80:560])
        detections = detection_dict['detection']
        detections = torch.cat([detections[:, :6], planes], dim=-1)
        depth, detection_mask = calcDepthModule(config, detections, detection_dict['masks'])

        detection_dict['depth'] = depth.unsqueeze(0)
        pass

    valid_mask = input_dict['depth'] > 1e-4
    depth_gt = input_dict['depth']

    depth_pred = detection_dict['depth']
    detection_mask = detection_dict['mask'] > 0.5
    plane_mask_gt = input_dict['segmentation'] >= 0
    plane_mask_pred = detection_mask

    padding = 0
    depth_gt = depth_gt[:, 80:560]
    depth_pred = depth_pred[:, 80:560]        

    nyu_mask = torch.zeros((1, 640, 640)).cuda()
    nyu_mask[:, 80 + 44:80 + 471, 40:601] = 1
    nyu_mask = nyu_mask > -0.5
    for c, plane_mask in enumerate([nyu_mask]):
        valid_mask_depth = valid_mask * plane_mask
        if padding > 0:
            valid_mask_depth = valid_mask_depth[:, 80 + padding:560 - padding, padding:-padding]
        else:
            valid_mask_depth = valid_mask_depth[:, 80:560]
            pass

        if options.debug:
            print('\nmask', c)
            pass
        depth_statistics = evaluateDepths(depth_pred[valid_mask_depth].detach().cpu().numpy(), depth_gt[valid_mask_depth].detach().cpu().numpy(), printInfo=printInfo)
        statistics[c].append(depth_statistics[:5])
        continue

    statistics[1].append([0, ])    
    
    if options.debug:
        if 'depth_np' in detection_dict:
            depth_pred = detection_dict['depth_np']
            if padding > 0:
                depth_pred = depth_pred[:, 80 + padding:560 - padding, padding:-padding]
            else:
                depth_pred = depth_pred[:, 80:560]
                pass
            print('\nnon planar')
            evaluateDepths(depth_pred[valid_mask_depth].detach().cpu().numpy(), depth_gt[valid_mask_depth].detach().cpu().numpy(), printInfo=True)
            pass
        if 'depth_ori' in detection_dict:
            depth_pred = detection_dict['depth_ori']
            if padding > 0:
                depth_pred = depth_pred[:, 80 + padding:560 - padding, padding:-padding]
            else:
                depth_pred = depth_pred[:, 80:560]
                pass
            print('\noriginal')
            evaluateDepths(depth_pred[valid_mask_depth].detach().cpu().numpy(), depth_gt[valid_mask_depth].detach().cpu().numpy(), printInfo=True)
            pass
        pass
        
    statistics[2].append(evaluateMasksTensor(torch.round(detection_dict['masks']).cpu(), input_dict['masks'].float().cpu(), valid_mask.float().cpu(), printInfo=printInfo))

    if 'masks' in detection_dict and evaluate_plane:
        plane_statistics = evaluatePlanesTensor(input_dict, detection_dict, printInfo=printInfo)
        statistics[3].append([plane_statistics[c] for c in [1, 2, 3]])
        pass
    return

def printStatisticsDetection(options, statistics):
    if not os.path.exists('logs'):
        os.system("mkdir -p logs")
        pass    
    if not os.path.exists('logs/global.txt'):
        open_type = 'w'
    else:
        open_type = 'a'
        pass
    with open('logs/global.txt', open_type) as f:
        values = np.array(statistics[0]).mean(0).tolist() + np.array(statistics[1]).mean(0).tolist() + np.array(statistics[2]).mean(0).tolist()

        if len(statistics[3]) > 0:
            values += np.array(statistics[3]).mean(0).tolist()
            pass
        name = options.keyname + '_' + options.anchorType
        if options.suffix != '':
            name += '_' + options.suffix
            pass
        if options.numAnchorPlanes > 0:
            name += '_' + str(options.numAnchorPlanes)
            pass
        if options.startEpoch >= 0:
            name += '_' + str(options.startEpoch)
            pass
        if options.modelType != '':
            name += '_' + options.modelType
            pass
        
        line = options.dataset + ': ' + name + ' statistics:'
        for v in values:
            line += ' %0.3f'%v
            continue
        print('\nstatistics', line)
        line += '\n'
        f.write(line)
        f.close()
        pass
    return

def plotCurves(filename='test/curves.png', xlabel='depth threshold', ylabel='per plane recall %', title='', methods=['manhattan_pred', 'manhattan_gt', 'planenet_normal', 'refine_normal_refine']):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.gca()
    colors = []
    markers = []
    sizes = []
    
    colors.append('blue')
    colors.append('red')
    colors.append('orange')
    colors.append('purple')    
    colors.append('brown')

    for _ in range(len(methods)):
        markers.append('')
        continue
    markers[1] = 'o'
    for _ in range(len(methods)):
        sizes.append(1)
        continue
    sizes[-1] = 2

    ordering = range(len(methods))
    final_labels = ['Manhattan + inferred depth', 'Manhattan + gt depth', 'PlaneNet', 'Ours']
    xs = (np.arange(21) * 0.05).tolist()
    ys = {method: [] for method in methods}
    with open('logs/global.txt', 'r') as f:
        for line in f:
            tokens = line.split(' ')
            method = tokens[1].strip()
            if len(tokens) > 30 and method in ys and tokens[0].strip() != 'nyu:':
                ys[method] = [float(v.strip()) for v in tokens[-21:]]
                pass
            continue
        pass
    ys = [ys[method] for method in methods]
    for order in ordering:
        plt.plot(xs, ys[order], figure=fig, label=final_labels[order], color=colors[order], marker=markers[order], linewidth=sizes[order])
        continue
    plt.legend(loc='upper right', bbox_to_anchor=(0.5, 1.05), ncol=1, fancybox=True, shadow=True, handletextpad=0.1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel + ' %')
    ax.set_yticklabels(np.arange(0, 51, 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.xlim((xs[0], xs[-1] + 0.01))
    plt.ylim((0, 0.5))
    plt.tight_layout(w_pad=0.4)
    plt.savefig(filename)
    return


def writeTable(filename='logs/table.txt', methods={'planenet_normal': 'PlaneNet', 'warping_normal_pair': 'Ours', 'basic_normal_backup': 'Ours (w/o warping loss)', 'warping_normal_none_pair': 'Ours (w/o normal anchors', 'warping_joint_pair': 'Ours (w/o depth map)'}, cols=[20, 19, 21, 32, 38, 44], dataset=''):
    """Write the comparison table (Table 2)"""
    method_statistics = {}
    with open('logs/global.txt', 'r') as f:
        for line in f:
            tokens = line.split(' ')
            method = tokens[1].strip()
            if len(tokens) > max(cols) and method in methods and tokens[0].strip()[:-1] == dataset:
                method_statistics[method] = [float(tokens[c].strip()) for c in cols]
                pass
            continue
        pass
    with open(filename, 'w') as f:
        for k, values in method_statistics.items():
            f.write(methods[k])
            for v in values:
                f.write(' & %0.3f'%v)
                continue
            f.write(' \\\\\n')
            continue
        pass
    return
