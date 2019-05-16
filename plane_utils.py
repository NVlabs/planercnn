"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import numpy as np
from utils import *

def getCameraFromInfo(info):
    camera = {}
    camera['fx'] = info[0]
    camera['fy'] = info[5]
    camera['cx'] = info[2]
    camera['cy'] = info[6]
    camera['width'] = info[16]
    camera['height'] = info[17]
    camera['depth_shift'] = info[18]    
    return camera

def one_hot(values, depth):
    maxInds = values.reshape(-1)
    results = np.zeros([maxInds.shape[0], depth])
    results[np.arange(maxInds.shape[0]), maxInds] = 1
    results = results.reshape(list(values.shape) + [depth])
    return results

def normalize(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-4)

def calcNormal(depth, info):

    height = depth.shape[0]
    width = depth.shape[1]

    camera = getCameraFromInfo(info)
    urange = (np.arange(width, dtype=np.float32) / (width) * (camera['width']) - camera['cx']) / camera['fx']
    urange = urange.reshape(1, -1).repeat(height, 0)
    vrange = (np.arange(height, dtype=np.float32) / (height) * (camera['height']) - camera['cy']) / camera['fy']
    vrange = vrange.reshape(-1, 1).repeat(width, 1)
    
    X = depth * urange
    Y = depth
    Z = -depth * vrange

    points = np.stack([X, Y, Z], axis=2).reshape(-1, 3)

    if width > 300:
        grids = np.array([-9, -6, -3, -1, 0, 1, 3, 6, 9])
    else:
        grids = np.array([-5, -3, -1, 0, 1, 3, 5])

    normals = []
    for index in range(width * height):
        us = index % width + grids
        us = us[np.logical_and(us >= 0, us < width)]
        vs = index // width + grids
        vs = vs[np.logical_and(vs >= 0, vs < height)]
        indices = (np.expand_dims(vs, -1) * width + np.expand_dims(us, 0)).reshape(-1).astype(np.int32)
        planePoints = points[indices]
        planePoints = planePoints[np.linalg.norm(planePoints, axis=-1) > 1e-4]

        planePoints = planePoints[np.abs(planePoints[:, 1] - points[index][1]) < 0.05]
            
        try:
            plane = fitPlane(planePoints)
            normals.append(-plane / np.maximum(np.linalg.norm(plane), 1e-4))
        except:
            if len(normals) > 0:
                normals.append(normals[-1])
            else:
                normals.append([0, -1, 0])
                pass
            pass
        continue
    normal = np.array(normals).reshape((height, width, 3))
    return normal

def getSegmentationsGraphCut(planes, image, depth, normal, semantics, info, parameters={}):
    
    height = depth.shape[0]
    width = depth.shape[1]

    numPlanes = planes.shape[0]

    camera = getCameraFromInfo(info)
    urange = (np.arange(width, dtype=np.float32) / (width) * (camera['width']) - camera['cx']) / camera['fx']
    urange = urange.reshape(1, -1).repeat(height, 0)
    vrange = (np.arange(height, dtype=np.float32) / (height) * (camera['height']) - camera['cy']) / camera['fy']
    vrange = vrange.reshape(-1, 1).repeat(width, 1)
    
    X = depth * urange
    Y = depth
    Z = -depth * vrange

    points = np.stack([X, Y, Z], axis=2)
    planes = planes[:numPlanes]
    planesD = np.linalg.norm(planes, axis=1, keepdims=True)
    planeNormals = planes / np.maximum(planesD, 1e-4)

    if 'distanceCostThreshold' in parameters:
        distanceCostThreshold = parameters['distanceCostThreshold']
    else:
        distanceCostThreshold = 0.05
        pass

    distanceCost = np.abs(np.tensordot(points, planeNormals, axes=([2, 1])) - np.reshape(planesD, [1, 1, -1])) / distanceCostThreshold
    distanceCost = np.concatenate([distanceCost, np.ones((height, width, 1))], axis=2)

    normalCostThreshold = 1 - np.cos(np.deg2rad(30))
    normalCost = (1 - np.abs(np.tensordot(normal, planeNormals, axes=([2, 1])))) / normalCostThreshold
    normalCost = np.concatenate([normalCost, np.ones((height, width, 1))], axis=2)


    unaryCost = distanceCost + normalCost
    unaryCost *= np.expand_dims((depth > 1e-4).astype(np.float32), -1)
    unaries = -unaryCost.reshape((-1, numPlanes + 1))

    cv2.imwrite('test/distance_cost.png', drawSegmentationImage(-distanceCost.reshape((height, width, -1)), unaryCost.shape[-1] - 1))
    cv2.imwrite('test/normal_cost.png', drawSegmentationImage(-normalCost.reshape((height, width, -1)), unaryCost.shape[-1] - 1))
    cv2.imwrite('test/unary_cost.png', drawSegmentationImage(-unaryCost.reshape((height, width, -1)), blackIndex=unaryCost.shape[-1] - 1))    
    cv2.imwrite('test/segmentation.png', drawSegmentationImage(unaries.reshape((height, width, -1)), blackIndex=numPlanes))


    nodes = np.arange(height * width).reshape((height, width))

    image = image.astype(np.float32)
    colors = image.reshape((-1, 3))    

    deltas = [(0, 1), (1, 0), (-1, 1), (1, 1)]    
    
    intensityDifferenceSum = 0.0
    intensityDifferenceCount = 0
    for delta in deltas:
        deltaX = delta[0]
        deltaY = delta[1]
        partial_nodes = nodes[max(-deltaY, 0):min(height - deltaY, height), max(-deltaX, 0):min(width - deltaX, width)].reshape(-1)
        intensityDifferenceSum += np.sum(pow(colors[partial_nodes] - colors[partial_nodes + (deltaY * width + deltaX)], 2))
        intensityDifferenceCount += partial_nodes.shape[0]
        continue
    intensityDifference = intensityDifferenceSum / intensityDifferenceCount

    
    edges = []
    edges_features = []
    pairwise_matrix = 1 - np.diag(np.ones(numPlanes + 1))

    for delta in deltas:
        deltaX = delta[0]
        deltaY = delta[1]
        partial_nodes = nodes[max(-deltaY, 0):min(height - deltaY, height), max(-deltaX, 0):min(width - deltaX, width)].reshape(-1)
        edges.append(np.stack([partial_nodes, partial_nodes + (deltaY * width + deltaX)], axis=1))

        colorDiff = np.sum(pow(colors[partial_nodes] - colors[partial_nodes + (deltaY * width + deltaX)], 2), axis=1)
        
        pairwise_cost = np.expand_dims(pairwise_matrix, 0) * np.reshape(1 + 45 * np.exp(-colorDiff / intensityDifference), [-1, 1, 1])
        edges_features.append(pairwise_cost)
        continue

    edges = np.concatenate(edges, axis=0)
    edges_features = np.concatenate(edges_features, axis=0)

    if 'smoothnessWeight' in parameters:
        smoothnessWeight = parameters['smoothnessWeight']
    else:
        smoothnessWeight = 0.02
        pass

    print('start')
    refined_segmentation = inference_ogm(unaries, -edges_features * smoothnessWeight, edges, return_energy=False, alg='alphaexp')
    print('done')
    refined_segmentation = refined_segmentation.reshape([height, width])

    if 'semantics' in parameters and parameters['semantics']:
        for semanticIndex in np.unique(semantics):
            mask = semantics == semanticIndex
            segmentInds = refined_segmentation[mask]
            uniqueSegments, counts = np.unique(segmentInds, return_counts=True)
            for index, count in enumerate(counts):
                if count > segmentInds.shape[0] * 0.9:
                    refined_segmentation[mask] = uniqueSegments[index]
                    pass
                continue
            continue
        pass
    
    return refined_segmentation

def fitPlanesNYU(image, depth, normal, semantics, info, numOutputPlanes=20, local=-1, parameters={}):
    camera = getCameraFromInfo(info)
    width = depth.shape[1]
    height = depth.shape[0]

    urange = (np.arange(width, dtype=np.float32) / width * camera['width'] - camera['cx']) / camera['fx']
    urange = urange.reshape(1, -1).repeat(height, 0)
    vrange = (np.arange(height, dtype=np.float32) / height * camera['height'] - camera['cy']) / camera['fy']
    vrange = vrange.reshape(-1, 1).repeat(width, 1)
    X = depth * urange
    Y = depth
    Z = -depth * vrange

    XYZ = np.stack([X, Y, Z], axis=2).reshape(-1, 3)
    planes = []
    planeMasks = []
    invalidDepthMask = depth < 1e-4

    if 'planeAreaThreshold' in parameters:
        planeAreaThreshold = parameters['planeAreaThreshold']
    else:
        planeAreaThreshold = 500
        pass
    if 'distanceThreshold' in parameters:
        distanceThreshold = parameters['distanceThreshold']
    else:
        distanceThreshold = 0.05
        pass
    if 'local' in parameters:
        local = parameters['local']
    else:
        local = 0.2
        pass
    
    for y in range(5, height, 10):
        for x in range(5, width, 10):
            if invalidDepthMask[y][x]:
                continue
            sampledPoint = XYZ[y * width + x]
            sampledPoints = XYZ[np.linalg.norm(XYZ - sampledPoint, 2, 1) < local]
            if sampledPoints.shape[0] < 3:
                continue
            sampledPoints = sampledPoints[np.random.choice(np.arange(sampledPoints.shape[0]), size=(3))]
            
            try:
                plane = fitPlane(sampledPoints)
                pass
            except:
                continue

            diff = np.abs(np.matmul(XYZ, plane) - np.ones(XYZ.shape[0])) / np.linalg.norm(plane)
            inlierIndices = diff < distanceThreshold
            if np.sum(inlierIndices) < planeAreaThreshold:
                continue
            
            planes.append(plane)
            planeMasks.append(inlierIndices.reshape((height, width)))
            continue
        continue
    
    planes = np.array(planes)
    planeList = zip(planes, planeMasks)
    planeList = sorted(planeList, key=lambda x:-x[1].sum())
    planes, planeMasks = zip(*planeList)

    
    invalidMask = np.zeros((height, width), np.bool)
    validPlanes = []
    validPlaneMasks = []
    
    for planeIndex, plane in enumerate(planes):
        planeMask = planeMasks[planeIndex]
        if np.logical_and(planeMask, invalidMask).sum() > planeMask.sum() * 0.5:
            continue

        validPlanes.append(plane)
        validPlaneMasks.append(planeMask)
        invalidMask = np.logical_or(invalidMask, planeMask)
        continue
    planes = np.array(validPlanes)
    planesD = 1 / np.maximum(np.linalg.norm(planes, 2, 1, keepdims=True), 1e-4)
    planes *= pow(planesD, 2)
    
    planeMasks = np.stack(validPlaneMasks, axis=2)
    
    cv2.imwrite('test/depth.png', drawDepthImage(depth))
    for planeIndex in range(planes.shape[0]):
        cv2.imwrite('test/mask_' + str(planeIndex) + '.png', drawMaskImage(planeMasks[:, :, planeIndex]))
        continue

    print('number of planes: ' + str(planes.shape[0]))
    
    planeSegmentation = getSegmentationsGraphCut(planes, image, depth, normal, semantics, info, parameters=parameters)

    cv2.imwrite('test/segmentation_refined.png', drawSegmentationImage(planeSegmentation, blackIndex=planes.shape[0]))
    
    return planes, planeSegmentation

def readProposalInfo(info, proposals):
    numProposals = proposals.shape[-1]
    outputShape = list(info.shape)
    outputShape[-1] = numProposals
    info = info.reshape([-1, info.shape[-1]])
    proposals = proposals.reshape([-1, proposals.shape[-1]])
    proposalInfo = []

    for proposal in xrange(numProposals):
        proposalInfo.append(info[np.arange(info.shape[0]), proposals[:, proposal]])
        continue
    proposalInfo = np.stack(proposalInfo, axis=1).reshape(outputShape)
    return proposalInfo

def fitPlanesManhattan(image, depth, normal, info, numOutputPlanes=20, imageIndex=-1, parameters={}):
    if 'meanshift' in parameters and parameters['meanshift'] > 0:
        import sklearn.cluster
        meanshift = sklearn.cluster.MeanShift(parameters['meanshift'])
        pass

    
    height = depth.shape[0]
    width = depth.shape[1]

    camera = getCameraFromInfo(info)
    urange = (np.arange(width, dtype=np.float32) / (width) * (camera['width']) - camera['cx']) / camera['fx']
    urange = urange.reshape(1, -1).repeat(height, 0)
    vrange = (np.arange(height, dtype=np.float32) / (height) * (camera['height']) - camera['cy']) / camera['fy']
    vrange = vrange.reshape(-1, 1).repeat(width, 1)
    
    X = depth * urange
    Y = depth
    Z = -depth * vrange


    normals = normal.reshape((-1, 3))
    normals = normals / np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-4)

    validMask = np.logical_and(np.linalg.norm(normals, axis=-1) > 1e-4, depth.reshape(-1) > 1e-4)
    
    valid_normals = normals[validMask]

    
    points = np.stack([X, Y, Z], axis=2).reshape(-1, 3)
    valid_points = points[validMask]

    polarAngles = np.arange(16) * np.pi / 2 / 16
    azimuthalAngles = np.arange(64) * np.pi * 2 / 64
    polarAngles = np.expand_dims(polarAngles, -1)
    azimuthalAngles = np.expand_dims(azimuthalAngles, 0)

    normalBins = np.stack([np.sin(polarAngles) * np.cos(azimuthalAngles), np.tile(np.cos(polarAngles), [1, azimuthalAngles.shape[1]]), -np.sin(polarAngles) * np.sin(azimuthalAngles)], axis=2)
    normalBins = np.reshape(normalBins, [-1, 3])
    numBins = normalBins.shape[0]
    
    
    normalDiff = np.tensordot(valid_normals, normalBins, axes=([1], [1]))
    normalDiffSign = np.sign(normalDiff)
    normalDiff = np.maximum(normalDiff, -normalDiff)
    normalMask = one_hot(np.argmax(normalDiff, axis=-1), numBins)
    bins = normalMask.sum(0)
    np.expand_dims(valid_normals, 1) * np.expand_dims(normalMask, -1)

    maxNormals = np.expand_dims(valid_normals, 1) * np.expand_dims(normalMask, -1)
    maxNormals *= np.expand_dims(normalDiffSign, -1)
    averageNormals = maxNormals.sum(0) / np.maximum(np.expand_dims(bins, -1), 1e-4)
    averageNormals /= np.maximum(np.linalg.norm(averageNormals, axis=-1, keepdims=True), 1e-4)
    dominantNormal_1 = averageNormals[np.argmax(bins)]

    dotThreshold_1 = np.cos(np.deg2rad(100))
    dotThreshold_2 = np.cos(np.deg2rad(80))
    
    dot_1 = np.tensordot(normalBins, dominantNormal_1, axes=([1], [0]))
    bins[np.logical_or(dot_1 < dotThreshold_1, dot_1 > dotThreshold_2)] = 0
    dominantNormal_2 = averageNormals[np.argmax(bins)]
    dot_2 = np.tensordot(normalBins, dominantNormal_2, axes=([1], [0]))
    bins[np.logical_or(dot_2 < dotThreshold_1, dot_2 > dotThreshold_2)] = 0
    
    dominantNormal_3 = averageNormals[np.argmax(bins)]


    dominantNormals = np.stack([dominantNormal_1, dominantNormal_2, dominantNormal_3], axis=0)

    dominantNormalImage = np.abs(np.matmul(normal, dominantNormals.transpose()))
    
    planeHypothesisAreaThreshold = width * height * 0.01

    
    planes = []
    
    if 'offsetGap' in parameters:
        offsetGap = parameters['offsetGap']
    else:
        offsetGap = 0.1
        pass
    for dominantNormal in dominantNormals:
        offsets = np.tensordot(valid_points, dominantNormal, axes=([1], [0]))

        if 'meanshift' in parameters and parameters['meanshift'] > 0:
            sampleInds = np.arange(offsets.shape[0])
            np.random.shuffle(sampleInds)
            meanshift.fit(np.expand_dims(offsets[sampleInds[:int(offsets.shape[0] * 0.02)]], -1))
            for offset in meanshift.cluster_centers_:
                planes.append(dominantNormal * offset)
                continue
            
        offset = offsets.min()
        maxOffset = offsets.max()
        while offset < maxOffset:
            planeMask = np.logical_and(offsets >= offset, offsets < offset + offsetGap)
            segmentOffsets = offsets[np.logical_and(offsets >= offset, offsets < offset + offsetGap)]
            if segmentOffsets.shape[0] < planeHypothesisAreaThreshold:
                offset += offsetGap
                continue
            planeD = segmentOffsets.mean()
            planes.append(dominantNormal * planeD)
            offset = planeD + offsetGap
            continue
        continue
    
    if len(planes) == 0:
        return np.array([]), np.zeros(segmentation.shape).astype(np.int32)
    
    planes = np.array(planes)
    print('number of planes ', planes.shape[0])

    vanishingPoints = np.stack([dominantNormals[:, 0] / np.maximum(dominantNormals[:, 1], 1e-4) * info[0] + info[2], -dominantNormals[:, 2] / np.maximum(dominantNormals[:, 1], 1e-4) * info[5] + info[6]], axis=1)
    vanishingPoints[:, 0] *= width / info[16]
    vanishingPoints[:, 1] *= height / info[17]

    indices = np.arange(width * height, dtype=np.int32)
    uv = np.stack([indices % width, indices // width], axis=1)
    colors = image.reshape((-1, 3))
    windowW = 9
    windowH = 3
    dominantLineMaps = []
    for vanishingPointIndex, vanishingPoint in enumerate(vanishingPoints):
        horizontalDirection = uv - np.expand_dims(vanishingPoint, 0)
        horizontalDirection = horizontalDirection / np.maximum(np.linalg.norm(horizontalDirection, axis=1, keepdims=True), 1e-4)
        verticalDirection = np.stack([horizontalDirection[:, 1], -horizontalDirection[:, 0]], axis=1)

        colorDiffs = []
        for directionIndex, direction in enumerate([horizontalDirection, verticalDirection]):
            neighbors = uv + direction
            neighborsX = neighbors[:, 0]
            neighborsY = neighbors[:, 1]
            neighborsMinX = np.maximum(np.minimum(np.floor(neighborsX).astype(np.int32), width - 1), 0)
            neighborsMaxX = np.maximum(np.minimum(np.ceil(neighborsX).astype(np.int32), width - 1), 0)
            neighborsMinY = np.maximum(np.minimum(np.floor(neighborsY).astype(np.int32), height - 1), 0)
            neighborsMaxY = np.maximum(np.minimum(np.ceil(neighborsY).astype(np.int32), height - 1), 0)
            indices_1 = neighborsMinY * width + neighborsMinX
            indices_2 = neighborsMaxY * width + neighborsMinX
            indices_3 = neighborsMinY * width + neighborsMaxX            
            indices_4 = neighborsMaxY * width + neighborsMaxX
            areas_1 = (neighborsMaxX - neighborsX) * (neighborsMaxY - neighborsY)
            areas_2 = (neighborsMaxX - neighborsX) * (neighborsY - neighborsMinY)
            areas_3 = (neighborsX - neighborsMinX) * (neighborsMaxY - neighborsY)
            areas_4 = (neighborsX - neighborsMinX) * (neighborsY - neighborsMinY)

            neighborsColor = colors[indices_1] * np.expand_dims(areas_1, -1) + colors[indices_2] * np.expand_dims(areas_2, -1) + colors[indices_3] * np.expand_dims(areas_3, -1) + colors[indices_4] * np.expand_dims(areas_4, -1)
            colorDiff = np.linalg.norm(neighborsColor - colors, axis=-1)

            colorDiffs.append(colorDiff)
            continue
        colorDiffs = np.stack(colorDiffs, 1)

        deltaUs, deltaVs = np.meshgrid(np.arange(windowW) - (windowW - 1) / 2, np.arange(windowH) - (windowH - 1) / 2)
        deltas = deltaUs.reshape((1, -1, 1)) * np.expand_dims(horizontalDirection, axis=1) + deltaVs.reshape((1, -1, 1)) * np.expand_dims(verticalDirection, axis=1)
        
        windowIndices = np.expand_dims(uv, 1) - deltas
        windowIndices = (np.minimum(np.maximum(np.round(windowIndices[:, :, 1]), 0), height - 1) * width + np.minimum(np.maximum(np.round(windowIndices[:, :, 0]), 0), width - 1)).astype(np.int32)
        
        dominantLineMap = []

        for pixels in windowIndices:
            gradientSums = colorDiffs[pixels].sum(0)
            dominantLineMap.append(gradientSums[1] / max(gradientSums[0], 1e-4))
            continue
        dominantLineMaps.append(np.array(dominantLineMap).reshape((height, width)))
        continue
    dominantLineMaps = np.stack(dominantLineMaps, axis=2)
    if 'dominantLineThreshold' in parameters:
        dominantLineThreshold = parameters['dominantLineThreshold']
    else:
        dominantLineThreshold = 3
        pass

    smoothnessWeightMask = dominantLineMaps.max(2) > dominantLineThreshold
    
    planesD = np.linalg.norm(planes, axis=1, keepdims=True)
    planeNormals = planes / np.maximum(planesD, 1e-4)


    if 'distanceCostThreshold' in parameters:
        distanceCostThreshold = parameters['distanceCostThreshold']
    else:
        distanceCostThreshold = 0.05
        pass
    
    distanceCost = np.abs(np.tensordot(points, planeNormals, axes=([1, 1])) - np.reshape(planesD, [1, -1])) / distanceCostThreshold

    normalCost = 0
    normalCostThreshold = 1 - np.cos(np.deg2rad(30))        
    normalCost = (1 - np.abs(np.tensordot(normals, planeNormals, axes=([1, 1])))) / normalCostThreshold
    
    unaryCost = distanceCost + normalCost
    unaryCost *= np.expand_dims(validMask.astype(np.float32), -1)
    unaries = unaryCost.reshape((width * height, -1))

    if False:
        cv2.imwrite('test/dominant_normal.png', drawMaskImage(dominantNormalImage))
        
        if imageIndex >= 0:
            cv2.imwrite('test/' + str(imageIndex) + '_dominant_lines.png', drawMaskImage(dominantLineMaps / dominantLineThreshold))
        else:
            cv2.imwrite('test/dominant_lines.png', drawMaskImage(dominantLineMaps / dominantLineThreshold))
            pass
        cv2.imwrite('test/dominant_lines_mask.png', drawMaskImage(smoothnessWeightMask))            
        cv2.imwrite('test/distance_cost.png', drawSegmentationImage(-distanceCost.reshape((height, width, -1)), unaryCost.shape[-1] - 1))
        cv2.imwrite('test/normal_cost.png', drawSegmentationImage(-normalCost.reshape((height, width, -1)), unaryCost.shape[-1] - 1))
        cv2.imwrite('test/unary_cost.png', drawSegmentationImage(-unaryCost.reshape((height, width, -1)), blackIndex=unaryCost.shape[-1] - 1))
        cv2.imwrite('test/segmentation.png', drawSegmentationImage(-unaries.reshape((height, width, -1)), blackIndex=unaries.shape[-1]))
        pass

    if 'numProposals' in parameters:
        numProposals = parameters['numProposals']
    else:
        numProposals = 3
        pass
    numProposals = min(numProposals, unaries.shape[-1] - 1)
    proposals = np.argpartition(unaries, numProposals)[:, :numProposals]
    proposals[np.logical_not(validMask)] = 0
    
    unaries = -readProposalInfo(unaries, proposals).reshape((-1, numProposals))
    
    nodes = np.arange(height * width).reshape((height, width))

    deltas = [(0, 1), (1, 0)]
    
    edges = []
    edges_features = []
    smoothnessWeights = 1 - 0.99 * smoothnessWeightMask.astype(np.float32)
    
    for delta in deltas:
        deltaX = delta[0]
        deltaY = delta[1]
        partial_nodes = nodes[max(-deltaY, 0):min(height - deltaY, height), max(-deltaX, 0):min(width - deltaX, width)].reshape(-1)
        edges.append(np.stack([partial_nodes, partial_nodes + (deltaY * width + deltaX)], axis=1))

        labelDiff = (np.expand_dims(proposals[partial_nodes], -1) != np.expand_dims(proposals[partial_nodes + (deltaY * width + deltaX)], 1)).astype(np.float32)
        edges_features.append(labelDiff * smoothnessWeights.reshape((width * height, -1))[partial_nodes].reshape(-1, 1, 1))
        continue

    edges = np.concatenate(edges, axis=0)
    edges_features = np.concatenate(edges_features, axis=0)


    if 'smoothnessWeight' in parameters:
        smoothnessWeight = parameters['smoothnessWeight']
    else:
        smoothnessWeight = 40
        pass

    print('start')
    refined_segmentation = inference_ogm(unaries, -edges_features * smoothnessWeight, edges, return_energy=False, alg='trw')
    print('done')
    
    refined_segmentation = refined_segmentation.reshape([height, width, 1])
    refined_segmentation = readProposalInfo(proposals, refined_segmentation)
    planeSegmentation = refined_segmentation.reshape([height, width])

    planeSegmentation[np.logical_not(validMask.reshape((height, width)))] = planes.shape[0]

    cv2.imwrite('test/segmentation_refined.png', drawSegmentationImage(planeSegmentation))
    
    return planes, planeSegmentation

def calcVanishingPoint(lines):
    points = lines[:, :2]
    normals = lines[:, 2:4] - lines[:, :2]
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-4)
    normals = np.stack([normals[:, 1], -normals[:, 0]], axis=1)
    normalPointDot = (normals * points).sum(1)

    if lines.shape[0] == 2:
        VP = np.linalg.solve(normals, normalPointDot)
    else:
        VP = np.linalg.lstsq(normals, normalPointDot)[0]
        pass
    
    return VP
    
def calcVanishingPoints(allLines, numVPs):
    distanceThreshold = np.sin(np.deg2rad(5))
    lines = allLines.copy()
    VPs = []
    VPLines = []
    for VPIndex in xrange(numVPs):
        points = lines[:, :2]
        lengths = np.linalg.norm(lines[:, 2:4] - lines[:, :2], axis=-1)
        normals = lines[:, 2:4] - lines[:, :2]
        normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-4)
        normals = np.stack([normals[:, 1], -normals[:, 0]], axis=1)
        maxNumInliers = 0
        bestVP = np.zeros(2)
        for _ in xrange(min(pow(lines.shape[0], 2), 100)):
            sampledInds = np.random.choice(lines.shape[0], 2)
            if sampledInds[0] == sampledInds[1]:
                continue
            sampledLines = lines[sampledInds]
            try:
                VP = calcVanishingPoint(sampledLines)
            except:
                continue

            inliers = np.abs(((np.expand_dims(VP, 0) - points) * normals).sum(-1)) / np.linalg.norm(np.expand_dims(VP, 0) - points, axis=-1) < distanceThreshold

            numInliers = lengths[inliers].sum()
            if numInliers > maxNumInliers:
                maxNumInliers = numInliers
                bestVP = VP
                bestVPInliers = inliers
                pass
            continue
        if maxNumInliers > 0:
            inlierLines = lines[bestVPInliers]
            VP = calcVanishingPoint(inlierLines)
            VPs.append(VP)

            VPLines.append(inlierLines)
            lines = lines[np.logical_not(bestVPInliers)]
            pass
        continue
    VPs = np.stack(VPs, axis=0)
    return VPs, VPLines, lines

def fitPlanesPiecewise(image, depth, normal, info, numOutputPlanes=20, imageIndex=1, parameters={}):
    if 'meanshift' in parameters and parameters['meanshift'] > 0:
        import sklearn.cluster
        meanshift = sklearn.cluster.MeanShift(parameters['meanshift'])
        pass
    
    from pylsd import lsd
    
    height = depth.shape[0]
    width = depth.shape[1]

    camera = getCameraFromInfo(info)
    urange = (np.arange(width, dtype=np.float32) / (width) * (camera['width']) - camera['cx']) / camera['fx']
    urange = urange.reshape(1, -1).repeat(height, 0)
    vrange = (np.arange(height, dtype=np.float32) / (height) * (camera['height']) - camera['cy']) / camera['fy']
    vrange = vrange.reshape(-1, 1).repeat(width, 1)
    
    X = depth * urange
    Y = depth
    Z = -depth * vrange


    normals = normal.reshape((-1, 3))
    normals = normals / np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-4)
    validMask = np.logical_and(np.linalg.norm(normals, axis=-1) > 1e-4, depth.reshape(-1) > 1e-4)
    
    points = np.stack([X, Y, Z], axis=2).reshape(-1, 3)
    valid_points = points[validMask]
    
    lines = lsd(image.mean(2))

    lineImage = image.copy()
    for line in lines:
        cv2.line(lineImage, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), (0, 0, 255), int(np.ceil(line[4] / 2)))
        continue
    cv2.imwrite('test/lines.png', lineImage)

    numVPs = 3
    VPs, VPLines, remainingLines = calcVanishingPoints(lines, numVPs=numVPs)

    lineImage = image.copy()    
    for VPIndex, lines in enumerate(VPLines):
        for line in lines:
            cv2.line(lineImage, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), ((VPIndex == 0) * 255, (VPIndex == 1) * 255, (VPIndex == 2) * 255), int(np.ceil(line[4] / 2)))
            continue
        continue
    cv2.imwrite('test/lines_vp.png', lineImage)    

    dominantNormals = np.stack([(VPs[:, 0] * info[16] / width - info[2]) / info[0], np.ones(numVPs), -(VPs[:, 1] * info[17] / height - info[6]) / info[5]], axis=1)
    dominantNormals /= np.maximum(np.linalg.norm(dominantNormals, axis=1, keepdims=True), 1e-4)

    dotThreshold = np.cos(np.deg2rad(20))
    for normalIndex, crossNormals in enumerate([[1, 2], [2, 0], [0, 1]]):
        normal = np.cross(dominantNormals[crossNormals[0]], dominantNormals[crossNormals[1]])
        normal = normalize(normal)
        if np.dot(normal, dominantNormals[normalIndex]) < dotThreshold:
            dominantNormals = np.concatenate([dominantNormals, np.expand_dims(normal, 0)], axis=0)
            pass
        continue

    print(VPs)
    print(dominantNormals)
    
    dominantNormalImage = np.abs(np.matmul(normal, dominantNormals.transpose()))
    cv2.imwrite('test/dominant_normal.png', drawMaskImage(dominantNormalImage))
    
    planeHypothesisAreaThreshold = width * height * 0.01
    
    planes = []
    vpPlaneIndices = []
    if 'offsetGap' in parameters:
        offsetGap = parameters['offsetGap']
    else:
        offsetGap = 0.1
        pass
    planeIndexOffset = 0

    for dominantNormal in dominantNormals:
        if np.linalg.norm(dominantNormal) < 1e-4:
            continue
        offsets = np.tensordot(valid_points, dominantNormal, axes=([1], [0]))

        if 'meanshift' in parameters and parameters['meanshift'] > 0:
            sampleInds = np.arange(offsets.shape[0])
            np.random.shuffle(sampleInds)
            meanshift.fit(np.expand_dims(offsets[sampleInds[:int(offsets.shape[0] * 0.02)]], -1))
            for offset in meanshift.cluster_centers_:
                planes.append(dominantNormal * offset)
                continue
        else:
            offset = offsets.min()
            maxOffset = offsets.max()
            while offset < maxOffset:
                planeMask = np.logical_and(offsets >= offset, offsets < offset + offsetGap)
                segmentOffsets = offsets[np.logical_and(offsets >= offset, offsets < offset + offsetGap)]
                if segmentOffsets.shape[0] < planeHypothesisAreaThreshold:
                    offset += offsetGap
                    continue
                planeD = segmentOffsets.mean()
                planes.append(dominantNormal * planeD)
                offset = planeD + offsetGap

                continue
            pass
        

        vpPlaneIndices.append(np.arange(planeIndexOffset, len(planes)))
        planeIndexOffset = len(planes)
        continue

    if len(planes) == 0:
        return np.array([]), np.zeros(segmentation.shape).astype(np.int32)    
    planes = np.array(planes)

    
    
    planesD = np.linalg.norm(planes, axis=1, keepdims=True)
    planeNormals = planes / np.maximum(planesD, 1e-4)

    if 'distanceCostThreshold' in parameters:
        distanceCostThreshold = parameters['distanceCostThreshold']
    else:
        distanceCostThreshold = 0.05
        pass


    distanceCost = np.abs(np.tensordot(points, planeNormals, axes=([1, 1])) - np.reshape(planesD, [1, -1])) / distanceCostThreshold

    normalCostThreshold = 1 - np.cos(np.deg2rad(30))        
    normalCost = (1 - np.abs(np.tensordot(normals, planeNormals, axes=([1, 1])))) / normalCostThreshold

    if 'normalWeight' in parameters:
        normalWeight = parameters['normalWeight']
    else:
        normalWeight = 1
        pass
    
    unaryCost = distanceCost + normalCost * normalWeight
    unaryCost *= np.expand_dims(validMask.astype(np.float32), -1)    
    unaries = unaryCost.reshape((width * height, -1))
    
    
    print('number of planes ', planes.shape[0])
    cv2.imwrite('test/distance_cost.png', drawSegmentationImage(-distanceCost.reshape((height, width, -1)), unaryCost.shape[-1] - 1))

    cv2.imwrite('test/normal_cost.png', drawSegmentationImage(-normalCost.reshape((height, width, -1)), unaryCost.shape[-1] - 1))

    cv2.imwrite('test/unary_cost.png', drawSegmentationImage(-unaryCost.reshape((height, width, -1)), blackIndex=unaryCost.shape[-1] - 1))

    cv2.imwrite('test/segmentation.png', drawSegmentationImage(-unaries.reshape((height, width, -1)), blackIndex=unaries.shape[-1]))
    

    if 'numProposals' in parameters:
        numProposals = parameters['numProposals']
    else:
        numProposals = 3
        pass

    numProposals = min(numProposals, unaries.shape[-1] - 1)
    
    proposals = np.argpartition(unaries, numProposals)[:, :numProposals]
    unaries = -readProposalInfo(unaries, proposals).reshape((-1, numProposals))
    
    nodes = np.arange(height * width).reshape((height, width))

    deltas = [(0, 1), (1, 0)]
    
    edges = []
    edges_features = []
            
                
    for delta in deltas:
        deltaX = delta[0]
        deltaY = delta[1]
        partial_nodes = nodes[max(-deltaY, 0):min(height - deltaY, height), max(-deltaX, 0):min(width - deltaX, width)].reshape(-1)
        edges.append(np.stack([partial_nodes, partial_nodes + (deltaY * width + deltaX)], axis=1))

        labelDiff = (np.expand_dims(proposals[partial_nodes], -1) != np.expand_dims(proposals[partial_nodes + (deltaY * width + deltaX)], 1)).astype(np.float32)

        
        edges_features.append(labelDiff)
        continue

    edges = np.concatenate(edges, axis=0)
    edges_features = np.concatenate(edges_features, axis=0)


    if 'edgeWeights' in parameters:
        edgeWeights = parameters['edgeWeights']
    else:
        edgeWeights = [0.5, 0.6, 0.6]
        pass    
    
    lineSets = np.zeros((height * width, 3))
    creaseLines = np.expand_dims(np.stack([planeNormals[:, 0] / info[0], planeNormals[:, 1], -planeNormals[:, 2] / info[5]], axis=1), 1) * planesD.reshape((1, -1, 1))
    creaseLines = creaseLines - np.transpose(creaseLines, [1, 0, 2])    
    for planeIndex_1 in xrange(planes.shape[0]):
        for planeIndex_2 in xrange(planeIndex_1 + 1, planes.shape[0]):
            creaseLine = creaseLines[planeIndex_1, planeIndex_2]
            if abs(creaseLine[0]) > abs(creaseLine[2]):
                vs = np.arange(height)
                us = -(creaseLine[1] + (vs - info[6]) * creaseLine[2]) / creaseLine[0] + info[2]
                minUs = np.floor(us).astype(np.int32)
                maxUs = minUs + 1
                validIndicesMask = np.logical_and(minUs >= 0, maxUs < width)
                if validIndicesMask.sum() == 0:
                    continue
                vs = vs[validIndicesMask]
                minUs = minUs[validIndicesMask]
                maxUs = maxUs[validIndicesMask]
                edgeIndices = (height - 1) * width + (vs * (width - 1) + minUs)
                for index, edgeIndex in enumerate(edgeIndices):
                    pixel_1 = vs[index] * width + minUs[index]
                    pixel_2 = vs[index] * width + maxUs[index]
                    proposals_1 = proposals[pixel_1]
                    proposals_2 = proposals[pixel_2]                    
                    if planeIndex_1 in proposals_1 and planeIndex_2 in proposals_2:
                        proposalIndex_1 = np.where(proposals_1 == planeIndex_1)[0][0]
                        proposalIndex_2 = np.where(proposals_2 == planeIndex_2)[0][0]
                        edges_features[edgeIndex, proposalIndex_1, proposalIndex_2] *= edgeWeights[0]
                        pass
                    if planeIndex_2 in proposals_1 and planeIndex_1 in proposals_2:
                        proposalIndex_1 = np.where(proposals_1 == planeIndex_2)[0][0]
                        proposalIndex_2 = np.where(proposals_2 == planeIndex_1)[0][0]
                        edges_features[edgeIndex, proposalIndex_1, proposalIndex_2] *= edgeWeights[0]
                        pass
                    continue

                lineSets[vs * width + minUs, 0] = 1
                lineSets[vs * width + maxUs, 0] = 1
            else:
                us = np.arange(width)
                vs = -(creaseLine[1] + (us - info[2]) * creaseLine[0]) / creaseLine[2] + info[6]
                minVs = np.floor(vs).astype(np.int32)
                maxVs = minVs + 1
                validIndicesMask = np.logical_and(minVs >= 0, maxVs < height)
                if validIndicesMask.sum() == 0:
                    continue                
                us = us[validIndicesMask]
                minVs = minVs[validIndicesMask]
                maxVs = maxVs[validIndicesMask]                
                edgeIndices = (minVs * width + us)
                for index, edgeIndex in enumerate(edgeIndices):
                    pixel_1 = minVs[index] * width + us[index]
                    pixel_2 = maxVs[index] * width + us[index]
                    proposals_1 = proposals[pixel_1]
                    proposals_2 = proposals[pixel_2]                    
                    if planeIndex_1 in proposals_1 and planeIndex_2 in proposals_2:
                        proposalIndex_1 = np.where(proposals_1 == planeIndex_1)[0][0]
                        proposalIndex_2 = np.where(proposals_2 == planeIndex_2)[0][0]
                        edges_features[edgeIndex, proposalIndex_1, proposalIndex_2] *= edgeWeights[0]
                        pass
                    if planeIndex_2 in proposals_1 and planeIndex_1 in proposals_2:
                        proposalIndex_1 = np.where(proposals_1 == planeIndex_2)[0][0]
                        proposalIndex_2 = np.where(proposals_2 == planeIndex_1)[0][0]
                        edges_features[edgeIndex, proposalIndex_1, proposalIndex_2] *= edgeWeights[0]
                        pass
                    continue
                lineSets[minVs * width + us, 0] = 1
                lineSets[maxVs * width + us, 0] = 1                
                pass
            continue
        continue

    planeDepths = calcPlaneDepths(planes, width, height, np.array([info[0], info[5], info[2], info[6], info[16], info[17], 0, 0, 0, 0])).reshape((height * width, -1))
    planeDepths = readProposalInfo(planeDepths, proposals).reshape((-1, numProposals))

    planeHorizontalVPMask = np.ones((planes.shape[0], 3), dtype=np.bool)
    for VPIndex, planeIndices in enumerate(vpPlaneIndices):
        planeHorizontalVPMask[planeIndices] = False
        continue

    
    for VPIndex, lines in enumerate(VPLines):
        lp = lines[:, :2]
        ln = lines[:, 2:4] - lines[:, :2]
        ln /= np.maximum(np.linalg.norm(ln, axis=-1, keepdims=True), 1e-4)
        ln = np.stack([ln[:, 1], -ln[:, 0]], axis=1)
        lnp = (ln * lp).sum(1, keepdims=True)
        occlusionLines = np.concatenate([ln, lnp], axis=1)
        for occlusionLine in occlusionLines:
            if abs(occlusionLine[0]) > abs(occlusionLine[1]):
                vs = np.arange(height)
                us = (occlusionLine[2] - vs * occlusionLine[1]) / occlusionLine[0]
                minUs = np.floor(us).astype(np.int32)
                maxUs = minUs + 1
                validIndicesMask = np.logical_and(minUs >= 0, maxUs < width)
                vs = vs[validIndicesMask]
                minUs = minUs[validIndicesMask]
                maxUs = maxUs[validIndicesMask]                
                edgeIndices = (height - 1) * width + (vs * (width - 1) + minUs)
                for index, edgeIndex in enumerate(edgeIndices):
                    pixel_1 = vs[index] * width + minUs[index]
                    pixel_2 = vs[index] * width + maxUs[index]
                    proposals_1 = proposals[pixel_1]
                    proposals_2 = proposals[pixel_2]                    
                    for proposalIndex_1, planeIndex_1 in enumerate(proposals_1):
                        if not planeHorizontalVPMask[planeIndex_1][VPIndex]:
                            continue
                        planeDepth_1 = planeDepths[pixel_1][proposalIndex_1]
                        for proposalIndex_2, planeIndex_2 in enumerate(proposals_2):
                            if planeDepths[pixel_2][proposalIndex_2] > planeDepth_1:
                                edges_features[edgeIndex, proposalIndex_1, proposalIndex_2] *= edgeWeights[1]
                                pass
                            continue
                        continue
                    continue
                lineSets[vs * width + minUs, 1] = 1
                lineSets[vs * width + maxUs, 1] = 1
            else:
                us = np.arange(width)
                vs = (occlusionLine[2] - us * occlusionLine[0]) / occlusionLine[1]
                
                minVs = np.floor(vs).astype(np.int32)
                maxVs = minVs + 1
                validIndicesMask = np.logical_and(minVs >= 0, maxVs < height)
                us = us[validIndicesMask]
                minVs = minVs[validIndicesMask]
                maxVs = maxVs[validIndicesMask]                
                edgeIndices = (minVs * width + us)
                for index, edgeIndex in enumerate(edgeIndices):
                    pixel_1 = minVs[index] * width + us[index]
                    pixel_2 = maxVs[index] * width + us[index]
                    proposals_1 = proposals[pixel_1]
                    proposals_2 = proposals[pixel_2]                    
                    for proposalIndex_1, planeIndex_1 in enumerate(proposals_1):
                        if not planeHorizontalVPMask[planeIndex_1][VPIndex]:
                            continue
                        planeDepth_1 = planeDepths[pixel_1][proposalIndex_1]
                        for proposalIndex_2, planeIndex_2 in enumerate(proposals_2):
                            if planeDepths[pixel_2][proposalIndex_2] > planeDepth_1:
                                edges_features[edgeIndex, proposalIndex_1, proposalIndex_2] *= edgeWeights[1]
                                pass
                            continue
                        continue
                    continue
                lineSets[minVs * width + us, 1] = 1
                lineSets[maxVs * width + us, 1] = 1                
                pass
            continue
        continue

    for line in remainingLines:
        if abs(line[3] - line[1]) > abs(line[2] - line[0]):
            if line[3] < line[1]:
                line = np.array([line[2], line[3], line[0], line[1]])
                pass
            vs = np.arange(line[1], line[3] + 1, dtype=np.int32)
            us = line[0] + (vs - line[1]) / (line[3] - line[1]) * (line[2] - line[0])
            minUs = np.floor(us).astype(np.int32)
            maxUs = minUs + 1
            validIndicesMask = np.logical_and(minUs >= 0, maxUs < width)
            vs = vs[validIndicesMask]
            minUs = minUs[validIndicesMask]
            maxUs = maxUs[validIndicesMask]                
            edgeIndices = (height - 1) * width + (vs * (width - 1) + minUs)
            for edgeIndex in edgeIndices:
                edges_features[edgeIndex] *= edgeWeights[2]
                continue
            lineSets[(vs * width + minUs), 2] = 1
            lineSets[(vs * width + maxUs), 2] = 1            
        else:
            if line[2] < line[0]:
                line = np.array([line[2], line[3], line[0], line[1]])
                pass
            us = np.arange(line[0], line[2] + 1, dtype=np.int32)
            vs = line[1] + (us - line[0]) / (line[2] - line[0]) * (line[3] - line[1])
            
            minVs = np.floor(vs).astype(np.int32)
            maxVs = minVs + 1
            validIndicesMask = np.logical_and(minVs >= 0, maxVs < height)
            us = us[validIndicesMask]
            minVs = minVs[validIndicesMask]
            maxVs = maxVs[validIndicesMask]
            edgeIndices = (minVs * width + us)
            for edgeIndex in edgeIndices:
                edges_features[edgeIndex] *= edgeWeights[2]
                continue
            lineSets[minVs * width + us, 2] = 1
            lineSets[maxVs * width + us, 2] = 1
            continue
        continue
    cv2.imwrite('test/line_sets.png', drawMaskImage(lineSets.reshape((height, width, 3))))
    

    if 'smoothnessWeight' in parameters:
        smoothnessWeight = parameters['smoothnessWeight']
    else:
        smoothnessWeight = 4
        pass

    print('start')
    refined_segmentation = inference_ogm(unaries, -edges_features * smoothnessWeight, edges, return_energy=False, alg='trw')
    print('done')
    refined_segmentation = refined_segmentation.reshape([height, width, 1])    
    refined_segmentation = readProposalInfo(proposals, refined_segmentation)
    planeSegmentation = refined_segmentation.reshape([height, width])

    planeSegmentation[np.logical_not(validMask.reshape((height, width)))] = planes.shape[0]    
    cv2.imwrite('test/segmentation_refined.png', drawSegmentationImage(planeSegmentation))
    
    return planes, planeSegmentation

if __name__ == '__main__':
    from pystruct.inference import get_installed, inference_ogm, inference_dispatch
    
    input_dict = np.load('test/input_dict.npy')[()]
    input_dict['normal'] = calcNormal(input_dict['depth'], input_dict['info'])

    modelType = sys.argv[1]
    if 'nyu' in modelType:
        if 'gt' in modelType:
            parameters = {'distanceCostThreshold': 0.1, 'smoothnessWeight': 0.05, 'semantics': True}
        else:
            parameters = {'distanceCostThreshold': 0.1, 'smoothnessWeight': 0.03, 'semantics': True, 'distanceThreshold': 0.2}
            pass
        pred_p, pred_s = fitPlanesNYU(input_dict['image'], input_dict['depth'], input_dict['normal'], input_dict['semantics'], input_dict['info'], numOutputPlanes=20, parameters=parameters)
    elif 'manhattan' in modelType:
        if 'gt' in modelType:
            parameters = {'numProposals': 5, 'distanceCostThreshold': 0.1, 'smoothnessWeight': 30, 'dominantLineThreshold': 3, 'offsetGap': 0.1}
        else:
            parameters = {'numProposals': 5, 'distanceCostThreshold': 0.1, 'smoothnessWeight': 100, 'dominantLineThreshold': 3, 'offsetGap': 0.6}
            pass
        pred_p, pred_s = fitPlanesManhattan(input_dict['image'], input_dict['depth'], input_dict['normal'], input_dict['info'], numOutputPlanes=20, parameters=parameters)                    
    elif 'piecewise' in modelType:
        if 'gt' in modelType:
            parameters = {'distanceCostThreshold': 0.1, 'smoothnessWeight': 300, 'numProposals': 5, 'normalWeight': 1, 'meanshift': 0.2}
        else:
            parameters = {'numProposals': 5, 'distanceCostThreshold': 0.1, 'smoothnessWeight': 300, 'normalWeight': 1, 'meanshift': 0.2}
            pass
        pred_p, pred_s = fitPlanesPiecewise(input_dict['image'], input_dict['depth'], input_dict['normal'], input_dict['info'], numOutputPlanes=20, parameters=parameters)
        pass
    np.save('test/output_dict', {'segmentation': pred_s, 'plane': pred_p})
    pass
