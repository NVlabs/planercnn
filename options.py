"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import argparse

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='CornerNet')
    parser.add_argument('--gpu', dest='gpu',
                        help='gpu',
                        default=1, type=int)    
    parser.add_argument('--task', dest='task',
                        help='task type: [train, test, predict]',
                        default='train', type=str)
    parser.add_argument('--restore', dest='restore',
                        help='how to restore the model',
                        default=1, type=int)
    parser.add_argument('--batchSize', dest='batchSize',
                        help='batch size',
                        default=16, type=int)
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset name for training',
                        default='', type=str)
    parser.add_argument('--testingDataset', dest='testingDataset',
                        help='dataset name for test/predict',
                        default='', type=str)
    parser.add_argument('--dataFolder', dest='dataFolder',
                        help='data folder',
                        default='../../Data/ScanNet/', type=str)
    parser.add_argument('--anchorFolder', dest='anchorFolder',
                        help='anchor folder',
                        default='anchors/', type=str)    
    parser.add_argument('--customDataFolder', dest='customDataFolder',
                        help='data folder',
                        default='test/custom', type=str)
    parser.add_argument('--MaskRCNNPath', dest='MaskRCNNPath',
                        help='path to Mask R-CNN weights',
                        default='../mask_rcnn_coco.pth', type=str)    
    parser.add_argument('--numTrainingImages', dest='numTrainingImages',
                        help='the number of images to train',
                        default=1000, type=int)
    parser.add_argument('--numTestingImages', dest='numTestingImages',
                        help='the number of images to test/predict',
                        default=100, type=int)
    parser.add_argument('--testingIndex', dest='testingIndex',
                        help='the index of images to test',
                        default=-1, type=int)
    parser.add_argument('--LR', dest='LR',
                        help='learning rate',
                        default=1e-5, type=float)
    parser.add_argument('--numEpochs', dest='numEpochs',
                        help='the number of epochs',
                        default=1000, type=int)
    parser.add_argument('--startEpoch', dest='startEpoch',
                        help='starting epoch index',
                        default=-1, type=int)
    parser.add_argument('--modelType', dest='modelType',
                        help='model type',
                        default='', type=str)
    parser.add_argument('--heatmapThreshold', dest='heatmapThreshold',
                        help='heatmap threshold for positive predictions',
                        default=0.5, type=float)
    parser.add_argument('--distanceThreshold3D', dest='distanceThreshold3D',
                        help='distance threshold 3D',
                        default=0.2, type=float)
    parser.add_argument('--distanceThreshold2D', dest='distanceThreshold2D',
                        help='distance threshold 2D',
                        default=20, type=float)
    parser.add_argument('--numNodes', dest='numNodes',
                        help='the number of nodes',
                        default=10, type=int)
    parser.add_argument('--width', dest='width',
                        help='input width',
                        default=640, type=int)
    parser.add_argument('--height', dest='height',
                        help='input height',
                        default=512, type=int)
    parser.add_argument('--outputDim', dest='outputDim',
                        help='output dimension',
                        default=256, type=int)
    parser.add_argument('--numInputChannels', dest='numInputChannels',
                        help='the number of classes',
                        default=4, type=int)
    ## Flags
    parser.add_argument('--visualizeMode', dest='visualizeMode',
                        help='visualization mode',
                        default='', type=str)    
    parser.add_argument('--trainingMode', dest='trainingMode',
                        help='training mode',
                        default='all', type=str)
    parser.add_argument('--debug', dest='debug',
                        help='debug',
                        action='store_true')
    parser.add_argument('--suffix', dest='suffix',
                        help='suffix',
                        default='', type=str)
    parser.add_argument('--losses', dest='losses',
                        help='losses',
                        default='', type=str)
    parser.add_argument('--blocks', dest='blocks',
                        help='blocks',
                        default='', type=str)        
    ## Synthetic dataset
    parser.add_argument('--locationNoise', dest='locationNoise',
                        help='the location noise',
                        default=0.0, type=float)
    parser.add_argument('--cornerLocationNoise', dest='cornerLocationNoise',
                        help='the corner location noise',
                        default=0.0, type=float)
    parser.add_argument('--occlusionNoise', dest='occlusionNoise',
                        help='occlusion noise',
                        default=0, type=int)
    ## Corner net options
    parser.add_argument('--considerPartial', dest='considerPartial',
                        help='consider partial input',
                        action='store_true')
    parser.add_argument('--predictAdjacency', dest='predictAdjacency',
                        help='predict adjacency',
                        action='store_true')
    parser.add_argument('--correctionType', dest='correctionType',
                        help='connection type',
                        default='one', type=str)
    parser.add_argument('--savePoints', dest='savePoints',
                        help='save points',
                        action='store_true')
    parser.add_argument('--numViews', dest='numViews',
                        help='the number of views',
                        default=0, type=int)
    parser.add_argument('--minNumPointRatio', dest='minNumPointRatio',
                        help='the minimum number of points (ratio)',
                        default=0.05, type=float)
    ## Mask RCNN options
    parser.add_argument('--maskWidth', dest='maskWidth',
                        help='mask width',
                        default=56, type=int)    
    parser.add_argument('--maskHeight', dest='maskHeight',
                        help='mask height',
                        default=56, type=int)
    parser.add_argument('--anchorType', dest='anchorType',
                        help='anchor type',
                        default='normal', type=str)
    parser.add_argument('--numPositiveExamples', dest='numPositiveExamples',
                        help='the nummber of positive examples',
                        default=200, type=int)
    ## Plane options
    parser.add_argument('--numAnchorPlanes', dest='numAnchorPlanes',
                        help='the number of anchor planes',
                        default=0, type=int)
    parser.add_argument('--frameGap', dest='frameGap',
                        help='frame gap',
                        default=20, type=int)
    parser.add_argument('--planeAreaThreshold', dest='planeAreaThreshold',
                        help='plane area threshold',
                        default=500, type=int)
    parser.add_argument('--planeWidthThreshold', dest='planeWidthThreshold',
                        help='plane width threshold',
                        default=10, type=int)
    parser.add_argument('--scaleMode', dest='scaleMode',
                        help='scale mode',
                        default='variant', type=str)
    ## Refinement options
    parser.add_argument('--cornerPositiveWeight', dest='cornerPositiveWeight',
                        help='larger weight for corners to fight the positive-negative balance issue',
                        default=0, type=int)    
    parser.add_argument('--positiveWeight', dest='positiveWeight',
                        help='positive weight',
                        default=0.33, type=float)
    parser.add_argument('--maskWeight', dest='maskWeight',
                        help='mask weight',
                        default=1, type=int)
    parser.add_argument('--warpingWeight', dest='warpingWeight',
                        help='warping weight',
                        default=0.1, type=float)    
    parser.add_argument('--convType', dest='convType',
                        help='convolution type',
                        default='2', type=str)
    ## Evaluation options
    parser.add_argument('--methods', dest='methods',
                        help='evaluation methods',
                        default='b', type=str)
    
    args = parser.parse_args()
    return args
