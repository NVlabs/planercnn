"""
Copyright (c) 2017 Matterport, Inc.
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import math
import numpy as np
import os
import torch

# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    NAME = None  # Override in sub-classes

    # Path to pretrained imagenet model
    IMAGENET_MODEL_PATH = os.path.join(os.getcwd(), "resnet50_imagenet.pth")

    # NUMBER OF GPUs to use. For CPU use 0
    GPU_COUNT = 1

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 1

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Number of classification classes (including background)
    NUM_CLASSES = 1  # Override in sub-classes

    # The number of input channels
    NUM_INPUT_CHANNELS = 3
    
    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

    # Non-max suppression threshold to filter RPN proposals.
    # You can reduce this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Input image resing
    # Images are resized such that the smallest side is >= IMAGE_MIN_DIM and
    # the longest side is <= IMAGE_MAX_DIM. In case both conditions can't
    # be satisfied together the IMAGE_MAX_DIM is enforced.
    # If True, pad images with zeros such that they're (max_dim by max_dim)
    IMAGE_PADDING = True  # currently, the False option is not supported

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    MEAN_PIXEL_TENSOR = torch.from_numpy(MEAN_PIXEL.astype(np.float32)).cuda()

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.

    # Percent of positive ROIs used to train classifier/mask heads
    if True:
        ## We used a 16G GPU so that we can set TRAIN_ROIS_PER_IMAGE to be 512
        #TRAIN_ROIS_PER_IMAGE = 512
        TRAIN_ROIS_PER_IMAGE = 200
        ROI_POSITIVE_RATIO = 0.33
    else:
        TRAIN_ROIS_PER_IMAGE = 512    
        ROI_POSITIVE_RATIO = 0.5
        pass

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]
    FINAL_MASK_SHAPE = [224, 224]  # (height, width) of the mini-mask
    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Use RPN ROIs or externally generated ROIs for training
    # Keep this True for most situations. Set to False if you want to train
    # the head branches on ROI generated by code rather than the ROIs from
    # the RPN. For example, to debug the classifier head without having to
    # train the RPN.
    USE_RPN_ROIS = True

    NUM_PARAMETERS = 3

    METADATA = np.array([571.87, 571.87, 320, 240, 640, 480, 0, 0, 0, 0])

    IMAGE_MAX_DIM = 640
    IMAGE_MIN_DIM = 480

    GLOBAL_MASK = False
    PREDICT_DEPTH = False

    NUM_PARAMETER_CHANNELS = 0
    
    def __init__(self, options):
        """Set values of computed attributes."""
        # Effective batch size
        if self.GPU_COUNT > 0:
            self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT
        else:
            self.BATCH_SIZE = self.IMAGES_PER_GPU

        # Adjust step size based on batch size
        self.STEPS_PER_EPOCH = self.BATCH_SIZE * self.STEPS_PER_EPOCH

        # Input image size
        self.IMAGE_SHAPE = np.array(
            [self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 3])


        with torch.no_grad():
            self.URANGE_UNIT = ((torch.arange(self.IMAGE_MAX_DIM, requires_grad=False).cuda().float() + 0.5) / self.IMAGE_MAX_DIM).view((1, -1)).repeat(self.IMAGE_MIN_DIM, 1)
            self.VRANGE_UNIT = ((torch.arange(self.IMAGE_MIN_DIM, requires_grad=False).cuda().float() + 0.5) / self.IMAGE_MIN_DIM).view((-1, 1)).repeat(1, self.IMAGE_MAX_DIM)
            self.ONES = torch.ones(self.URANGE_UNIT.shape, requires_grad=False).cuda()
            pass
        
        
        # Compute backbone size from input image size
        self.BACKBONE_SHAPES = np.array(
            [[int(math.ceil(self.IMAGE_SHAPE[0] / stride)),
              int(math.ceil(self.IMAGE_SHAPE[1] / stride))]
             for stride in self.BACKBONE_STRIDES])

        self.dataFolder = options.anchorFolder
        
        self.OCCLUSION = 'occlusion' in options.dataset
        
        self.loadAnchorPlanes(options.anchorType)
        self.PREDICT_DEPTH = True
        self.PREDICT_BOUNDARY = False
        self.PREDICT_NORMAL_NP = 'normal_np' in options.suffix
        
        self.BILINEAR_UPSAMPLING = 'bilinear' in options.suffix
        self.FITTING_TYPE = 0
        return
    
    def loadAnchorPlanes(self, anchor_type = ''):
        ## Load cluster centers to serve as the anchors
        self.ANCHOR_TYPE = anchor_type
        with torch.no_grad():
            if self.ANCHOR_TYPE == 'none':
                self.NUM_CLASSES = 2
                self.NUM_PARAMETERS = 3
            elif self.ANCHOR_TYPE == 'joint':
                self.ANCHOR_PLANES = np.load(self.dataFolder + '/anchor_planes.npy')
                self.NUM_CLASSES = len(self.ANCHOR_PLANES) + 1
                self.NUM_PARAMETERS = 4
                self.ANCHOR_PLANES_TENSOR = torch.from_numpy(self.ANCHOR_PLANES.astype(np.float32)).cuda()
            elif self.ANCHOR_TYPE == 'joint_Nd':
                self.ANCHOR_PLANES = np.load(self.dataFolder + '/anchor_planes_Nd.npy')            
                self.NUM_CLASSES = len(self.ANCHOR_PLANES) + 1
                self.NUM_PARAMETERS = 4
                self.ANCHOR_PLANES_TENSOR = torch.from_numpy(self.ANCHOR_PLANES.astype(np.float32)).cuda()            
            elif self.ANCHOR_TYPE == 'Nd':
                self.ANCHOR_NORMALS = np.load(self.dataFolder + '/anchor_planes_N.npy')
                self.ANCHOR_OFFSETS = np.squeeze(np.load(self.dataFolder + '/anchor_planes_d.npy'), -1)
                self.NUM_CLASSES = len(self.ANCHOR_NORMALS) * len(self.ANCHOR_OFFSETS) + 1
                self.NUM_PARAMETERS = 4
                self.ANCHOR_NORMALS_TENSOR = torch.from_numpy(self.ANCHOR_NORMALS.astype(np.float32)).cuda()
                self.ANCHOR_OFFSETS_TENSOR = torch.from_numpy(self.ANCHOR_OFFSETS.astype(np.float32)).cuda()
            elif 'normal' in self.ANCHOR_TYPE:
                if self.ANCHOR_TYPE == 'normal':
                    self.ANCHOR_NORMALS = np.load(self.dataFolder + '/anchor_planes_N.npy')
                else:
                    k = int(self.ANCHOR_TYPE[6:])
                    if k == 0:
                        self.ANCHOR_NORMALS = np.zeros((1, 3))
                    else:
                        self.ANCHOR_NORMALS = np.load(self.dataFolder + '/anchor_planes_N_' + str(k) + '.npy')
                        pass
                    pass
                self.NUM_CLASSES = len(self.ANCHOR_NORMALS) + 1
                self.NUM_PARAMETERS = 3
                self.ANCHOR_NORMALS_TENSOR = torch.from_numpy(self.ANCHOR_NORMALS.astype(np.float32)).cuda()                
                if self.OCCLUSION:
                    self.NUM_PARAMETER_CHANNELS = 1
                    pass
            elif self.ANCHOR_TYPE in ['patch', 'patch_Nd']:
                self.ANCHOR_NORMALS = np.load(self.dataFolder + '/anchor_planes_N.npy')
                self.NUM_CLASSES = len(self.ANCHOR_NORMALS) + 1
                self.ANCHOR_NORMALS_TENSOR = torch.from_numpy(self.ANCHOR_NORMALS.astype(np.float32)).cuda()
                if self.ANCHOR_TYPE == 'patch':
                    self.NUM_PARAMETER_CHANNELS = 1
                elif self.ANCHOR_TYPE == 'patch_Nd':
                    self.NUM_PARAMETER_CHANNELS = 4
                    pass
                self.ANCHOR_TYPE = 'normal'
            elif self.ANCHOR_TYPE == 'layout':
                self.ANCHOR_PLANES = np.load(self.dataFolder + '/anchor_planes_layout.npy')
                self.NUM_CLASSES = len(self.ANCHOR_PLANES) + 1
                self.NUM_PARAMETERS = 9
                self.ANCHOR_INFO = []
                for layout_type in range(5):
                    for c in range(3):
                        if layout_type == 0:
                            self.ANCHOR_INFO.append((1, 'none', layout_type))
                        elif layout_type == 1:
                            self.ANCHOR_INFO.append((2, 'convex', layout_type))
                        elif layout_type == 2:
                            self.ANCHOR_INFO.append((2, 'concave', layout_type))
                        elif layout_type == 3:
                            self.ANCHOR_INFO.append((3, 'convex', layout_type))
                        elif layout_type == 4:
                            self.ANCHOR_INFO.append((3, 'concave', layout_type))
                            pass
                        continue
                    continue
                self.ANCHOR_PLANES_TENSOR = torch.from_numpy(self.ANCHOR_PLANES.astype(np.float32)).cuda()            
            elif self.ANCHOR_TYPE == 'structure':
                self.ANCHOR_PLANES = np.load(self.dataFolder + '/anchor_planes_structure.npy')
                self.NUM_CLASSES = len(self.ANCHOR_PLANES) + 1
                num_anchor_planes = [10, 5, 5, 3, 3]
                self.ANCHOR_INFO = []
                self.TYPE_ANCHOR_OFFSETS = [0, ]
                for layout_type, num in enumerate(num_anchor_planes):
                    for c in range(num):
                        if layout_type == 0:
                            self.ANCHOR_INFO.append((1, 'none', layout_type))
                        elif layout_type == 1:
                            self.ANCHOR_INFO.append((2, 'convex', layout_type))
                        elif layout_type == 2:
                            self.ANCHOR_INFO.append((2, 'concave', layout_type))
                        elif layout_type == 3:
                            self.ANCHOR_INFO.append((3, 'convex', layout_type))
                        elif layout_type == 4:
                            self.ANCHOR_INFO.append((3, 'concave', layout_type))
                            pass
                        continue

                    self.TYPE_ANCHOR_OFFSETS.append(self.TYPE_ANCHOR_OFFSETS[-1] + num)
                    continue
                self.NUM_PARAMETERS = 9
                self.ANCHOR_PLANES_TENSOR = torch.from_numpy(self.ANCHOR_PLANES.astype(np.float32)).cuda()
            else:
                assert(False)
                pass
            pass
        return

    def applyAnchorsTensor(self, class_ids, parameters):
        if 'joint' in self.ANCHOR_TYPE:
            anchors = self.ANCHOR_PLANES_TENSOR[class_ids.data - 1]
            parameters = parameters[:, :3] + anchors
        elif self.ANCHOR_TYPE == 'Nd':
            normals = self.ANCHOR_NORMALS_TENSOR[(class_ids.data - 1) % self.ANCHOR_OFFSETS_TENSOR]
            offsets = self.ANCHOR_OFFSETS_TENSOR[(class_ids.data - 1) // self.ANCHOR_OFFSETS_TENSOR]
            parameters = (parameters[:, :3] + normals) * (parameters[:, 3] + offsets)
        elif self.ANCHOR_TYPE == 'layout':
            anchors = self.ANCHOR_PLANES_TENSOR[class_ids.data - 1]
            parameters = parameters[:, :3] + anchors
        elif 'normal' in self.ANCHOR_TYPE:
            normals = self.ANCHOR_NORMALS_TENSOR[class_ids.data - 1]
            parameters = parameters[:, :3] + normals
            pass
        return parameters
    
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
        return

    def getRanges(self, metadata):
        urange = (self.URANGE_UNIT * self.METADATA[4] - self.METADATA[2]) / self.METADATA[0]
        vrange = (self.VRANGE_UNIT * self.METADATA[5] - self.METADATA[3]) / self.METADATA[1]
        ranges = torch.stack([urange, self.ONES, -vrange], dim=-1)
        return ranges
        

class PlaneConfig(Config):
    """Configuration for training on ScanNet.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "plane"

    # We use one GPU with 8GB memory, which can fit one image.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 16

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 2  # COCO has 80 classes
    GLOBAL_MASK = False


class InferenceConfig(PlaneConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0
