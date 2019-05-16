"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn

import numpy as np
import os

from models.modules import *
from utils import *

class RefinementBlockParameter(torch.nn.Module):
   def __init__(self):
       super(RefinementBlockParameter, self).__init__()
       self.linear_1 = nn.Sequential(nn.Linear(3, 64), nn.LeakyReLU(0.1))
       self.linear_2 = nn.Sequential(nn.Linear(64, 128), nn.LeakyReLU(0.1))
       self.linear_3 = nn.Sequential(nn.Linear(256, 64), nn.LeakyReLU(0.1))
       self.linear_4 = nn.Sequential(nn.Linear(64, 64), nn.LeakyReLU(0.1))
       self.linear_5 = nn.Sequential(nn.Linear(64, 64), nn.LeakyReLU(0.1))
       self.pred = nn.Sequential(nn.Linear(64, 32), nn.LeakyReLU(0.1), nn.Linear(32, 3))
       return

   def sim(self, x):
       return (x.unsqueeze(1) * x).mean(-1, keepdim=True)
   
   def forward(self, parameters, mask_features):
       x = self.linear_2(self.linear_1(parameters))
       x = torch.cat([x, mask_features / 100], dim=-1)
       x = self.linear_3(x)
       x = (self.linear_4(x) * self.sim(x)).mean(1)
       x = (self.linear_5(x) * self.sim(x)).mean(1)
       return self.pred(x)
   
class RefinementBlockMask(torch.nn.Module):
   def __init__(self, options):
       super(RefinementBlockMask, self).__init__()
       self.options = options
       use_bn = False
       self.conv_0 = ConvBlock(3 + 5 + 2, 32, kernel_size=3, stride=1, padding=1, use_bn=use_bn)
       self.conv_1 = ConvBlock(64, 64, kernel_size=3, stride=2, padding=1, use_bn=use_bn)       
       self.conv_1_1 = ConvBlock(128, 64, kernel_size=3, stride=1, padding=1, use_bn=use_bn)
       self.conv_2 = ConvBlock(128, 128, kernel_size=3, stride=2, padding=1, use_bn=use_bn)
       self.conv_2_1 = ConvBlock(256, 128, kernel_size=3, stride=1, padding=1, use_bn=use_bn)

       self.up_2 = ConvBlock(128, 64, kernel_size=4, stride=2, padding=1, mode='deconv', use_bn=use_bn)
       self.up_1 = ConvBlock(128, 32, kernel_size=4, stride=2, padding=1, mode='deconv', use_bn=use_bn)              
       self.pred = nn.Sequential(ConvBlock(64, 16, kernel_size=3, stride=1, padding=1, mode='conv', use_bn=use_bn),
                                 torch.nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1))

       self.global_up_2 = ConvBlock(128, 64, kernel_size=4, stride=2, padding=1, mode='deconv', use_bn=use_bn)
       self.global_up_1 = ConvBlock(128, 32, kernel_size=4, stride=2, padding=1, mode='deconv', use_bn=use_bn)              
       self.global_pred = nn.Sequential(ConvBlock(64, 16, kernel_size=3, stride=1, padding=1, mode='conv', use_bn=use_bn),
                                       torch.nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1))
       self.depth_pred = nn.Sequential(ConvBlock(64, 16, kernel_size=3, stride=1, padding=1, mode='conv', use_bn=use_bn),
                                       torch.nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1))       
       
       if 'nonlocal' in options.suffix:
           self.mask_non_local = BatchNonLocalBlock(64, 64)           
           self.up_non_local_1 = BatchNonLocalBlock(64, 64)
           self.down_non_local_4 = BatchNonLocalBlock(256, 512)
           pass
       if 'crfrnn' in options.suffix:
           self.crfrnn = CRFRNNModule(image_dims=(192, 256), num_iterations=5)
           pass
       if 'parameter' in options.suffix:
           self.parameter_refinement_block = RefinementBlockParameter()
           pass
       return

   def accumulate(self, x):
       return torch.cat([x, (x.sum(0, keepdim=True) - x) / max(len(x) - 1, 1)], dim=1)
       
   def forward(self, image, masks, prev_parameters=None):
       x_mask = masks
       
       x_0 = torch.cat([image, x_mask], dim=1)

       x_0 = self.conv_0(x_0)
       x_1 = self.conv_1(self.accumulate(x_0))
       x_1 = self.conv_1_1(self.accumulate(x_1))
       x_2 = self.conv_2(self.accumulate(x_1))
       x_2 = self.conv_2_1(self.accumulate(x_2))
       
       if 'nonlocal' in self.options.suffix:
           x_4 = self.down_non_local_4(x_4)
           pass
       y_2 = self.up_2(x_2)
       y_1 = self.up_1(torch.cat([y_2, x_1], dim=1))
       y_0 = self.pred(torch.cat([y_1, x_0], dim=1))
       
       global_y_2 = self.global_up_2(x_2.mean(dim=0, keepdim=True))
       global_y_1 = self.global_up_1(torch.cat([global_y_2, x_1.mean(dim=0, keepdim=True)], dim=1))
       global_mask = self.global_pred(torch.cat([global_y_1, x_0.mean(dim=0, keepdim=True)], dim=1))
       depth = self.depth_pred(torch.cat([global_y_1, x_0.mean(dim=0, keepdim=True)], dim=1)) + x_mask[:1, :1]

       if 'parameter' in self.options.suffix:
           with torch.no_grad():
               mask_features = x_2.max(-1)[0].max(-1)[0]
               pass
           parameters = self.parameter_refinement_block(prev_parameters, mask_features) + prev_parameters
       else:
           parameters = prev_parameters
           pass
       
       y_0 = torch.cat([global_mask[:, 0], y_0.squeeze(1)], dim=0)
       return y_0, depth, parameters


class RefinementBlockConcat(torch.nn.Module):
   def __init__(self, options):
       super(RefinementBlockConcat, self).__init__()
       self.options = options

       use_bn = False
       max_num_planes = 30
       self.conv_0 = ConvBlock(3 + 2 + max_num_planes, 32, kernel_size=3, stride=1, padding=1, use_bn=use_bn)
       self.conv_1 = ConvBlock(32, 64, kernel_size=3, stride=2, padding=1, use_bn=use_bn)       
       self.conv_1_1 = ConvBlock(64, 64, kernel_size=3, stride=1, padding=1, use_bn=use_bn)
       self.conv_2 = ConvBlock(64, 128, kernel_size=3, stride=2, padding=1, use_bn=use_bn)
       self.conv_2_1 = ConvBlock(128, 128, kernel_size=3, stride=1, padding=1, use_bn=use_bn)

       self.up_2 = ConvBlock(128, 64, kernel_size=4, stride=2, padding=1, mode='deconv', use_bn=use_bn)
       self.up_1 = ConvBlock(128, 32, kernel_size=4, stride=2, padding=1, mode='deconv', use_bn=use_bn)              
       self.pred = nn.Sequential(ConvBlock(64, 32, kernel_size=3, stride=1, padding=1, mode='conv', use_bn=use_bn),
                                 torch.nn.Conv2d(32, max_num_planes, kernel_size=3, stride=1, padding=1))

       if 'nonlocal' in options.suffix:
           self.mask_non_local = BatchNonLocalBlock(64, 64)           
           self.up_non_local_1 = BatchNonLocalBlock(64, 64)
           self.down_non_local_4 = BatchNonLocalBlock(256, 512)
           pass
       if 'crfrnn' in options.suffix:
           self.crfrnn = CRFRNNModule(image_dims=(192, 256), num_iterations=5)
           pass
       if 'parameter' in options.suffix:
           self.parameter_refinement_block = RefinementBlockParameter()
           pass
       return

   def forward(self, masks):
       
       x_0 = masks

       x_0 = self.conv_0(x_0)
       x_1 = self.conv_1(x_0)
       x_1 = self.conv_1_1(x_1)
       x_2 = self.conv_2(x_1)
       x_2 = self.conv_2_1(x_2)
       y_2 = self.up_2(x_2)
       y_1 = self.up_1(torch.cat([y_2, x_1], dim=1))
       y_0 = self.pred(torch.cat([y_1, x_0], dim=1))
       
       return y_0

def convrelu2_block( num_inputs, num_outputs , kernel_size, stride, leaky_coef ):

    """
    :param num_inputs: number of input channels
    :param num_outputs:  number of output channels
    :param kernel_size:  kernel size
    :param stride:  stride
    :param leaky_coef:  leaky ReLU coefficients
    :return: 2x(Conv + ReLU) block
    """

    """ this block does two 1D convolutions, first on row, then on column """

    input = num_inputs; output = num_outputs
    k  = kernel_size; lc = leaky_coef

    if( not isinstance(stride, tuple)):
        s = (stride, stride)
    else:
        s = stride

    conv1_1 = nn.Conv2d( input,  output[0], (k[0], 1), padding=(k[0] // 2, 0), stride=(s[0], 1) )
    leaky_relu1_1 = nn.LeakyReLU( lc )

    conv1_2 = nn.Conv2d( output[0],  output[1], (1, k[1]), padding=(0, k[1] // 2), stride=(1, s[1]) )
    leaky_relu1_2 = nn.LeakyReLU( lc )

    return nn.Sequential(
        conv1_1,
        leaky_relu1_1,
        conv1_2,
        leaky_relu1_2
    )

def convrelu_block( num_inputs, num_outputs, kernel_size, stride, leaky_coef ):

    """
    :param num_inputs: number of input channels
    :param num_outputs:  number of output channels
    :param kernel_size:  kernel size
    :param stride:  stride
    :param leaky_coef:  leaky ReLU coefficients
    :return: (Conv + ReLU) block
    """

    """ this block does one 2D convolutions """

    input = num_inputs; output = num_outputs
    k = kernel_size; lc = leaky_coef

    if( not isinstance(stride, tuple)):
        s = (stride, stride)
    else:
        s = stride

    conv1_1 = nn.Conv2d(input, output, k, padding=(k[0] // 2, k[1] // 2), stride=s )
    leaky_relu1_1 = nn.LeakyReLU(lc)

    return nn.Sequential(
        conv1_1,
        leaky_relu1_1
    )

def linear_block( num_input_channels, num_output_channels, leaky_coef ):

    """
    :param num_input_channels: number of input channels
    :param num_output_channels:  number of output channels
    :param kernel_size:  kernel size
    :param stride:  stride
    :param leaky_coef:  leaky ReLU coefficients
    :return: (Linear + ReLU) block
    """

    """ this block is a fully connected layer """

    linear = nn.Linear(num_input_channels, num_output_channels)
    leaky_relu1_1 = nn.LeakyReLU(leaky_coef)

    return nn.Sequential(
        linear,
        leaky_relu1_1
    )

def predict_flow_block( num_inputs, num_outputs=4, intermediate_num_outputs=24):
    """
    :param num_inputs: number of input channels
    :param predict_confidence:  predict confidence or not
    :return: block for predicting flow
    """

    """"
    this block is --> (Conv+ReLU) --> Conv --> ,

    in the first prediction,  input is 512 x 8 x 6,
    in the second prediction, input is 128 x 64 x 48

    """

    conv1 = convrelu_block( num_inputs, intermediate_num_outputs,  (3, 3), 1, 0.1)
    conv2 = nn.Conv2d( intermediate_num_outputs,  num_outputs, (3, 3), padding=(1, 1), stride=1)

    return nn.Sequential(
        conv1,
        conv2
    )

def predict_motion_block( num_inputs , leaky_coef = 0.1, num_prev_parameters=0):

    """
    :param num_inputs: number of input channels
    :return: rotation, translation and scale
    """

    """
    this block is --> (Conv+ReLU) --> (FC+ReLU) --> (FC+ReLU) --> (FC+ReLU) -->,
    the output is rotation, translation and scale
    """

    conv1 = convrelu_block( num_inputs, 128,  (3, 3), 1, 0.1)

    if num_prev_parameters > 0:
        fc0 = nn.Sequential(linear_block(num_prev_parameters, 64, leaky_coef=leaky_coef),
                                 linear_block(64, 128, leaky_coef=leaky_coef),
                                 linear_block(128, 256, leaky_coef=leaky_coef),                            
        )
        fc1 = nn.Linear(128*8*6 + 256, 1024)
    else:
        fc1 = nn.Linear(128*8*6, 1024)
        fc0 = None
        pass
    fc2 = nn.Linear(1024, 128)
    fc3 = nn.Linear(128, 7)
    
    fc4 = nn.Linear(1024 + 256, 128)
    fc5 = nn.Linear(128, 3)
    
    leaky_relu1 = nn.LeakyReLU(leaky_coef)
    leaky_relu2 = nn.LeakyReLU(leaky_coef)
    leaky_relu3 = nn.LeakyReLU(leaky_coef)
    return conv1, nn.Sequential(fc1, leaky_relu1), nn.Sequential(fc2, leaky_relu2, fc3), fc0, nn.Sequential(fc4, leaky_relu3, fc5), 

class FlowBlock(nn.Module):

    def __init__(self, options, num_prev_channels=0):

        super(FlowBlock, self).__init__()

        self.num_prev_channels = num_prev_channels
        
        self.conv1 = convrelu2_block(6, (32, 32), (9, 9), 2, 0.1)

        if(self.num_prev_channels == 0):
            self.conv2 = convrelu2_block(32, (64, 64), (7, 7), 2, 0.1)
        else:
            """ in this case we also use the information from previous depth prediction """
            self.conv2 = convrelu2_block(32, (32, 32), (7, 7), 2, 0.1)
            self.conv2_extra_inputs = convrelu2_block(self.num_prev_channels, (32,32), (3, 3), 1, 0.1)
        self.conv2_1 = convrelu2_block(64, (64, 64), (3, 3), 1, 0.1)


        self.conv3 = convrelu2_block(64, (128,128), (5,5), 2, 0.1)
        self.conv3_1 = convrelu2_block(128, (128, 128), (3,3), 1, 0.1)

        self.conv4 = convrelu2_block(128, (256, 256), (5,5), 2, 0.1)
        self.conv4_1 = convrelu2_block(256, (256, 256), (3,3), 1, 0.1)


        """for conv5 layer, there is a mistake in the figure of demon paper, kernel size should be 5, not 3"""
        self.conv5 = convrelu2_block(256,(512, 512), (5,5), 2, 0.1)
        self.conv5_1 = convrelu2_block(512,(512, 512), (3,3), 1, 0.1)


        """five groups of convolution layers are finished"""

        self.flow1 = predict_flow_block(512, num_outputs=4)
        self.flow1_upconv = nn.ConvTranspose2d( 4, 2, (4,4), stride=(2,2), padding=1 )

        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d( 512, 256, (4,4), stride=(2,2), padding=1),
            nn.LeakyReLU(0.1))

        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d( 514, 128, (4,4), stride=(2,2), padding=1),
            nn.LeakyReLU(0.1))

        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d( 256, 64,  (4,4), stride=(2,2), padding=1),
            nn.LeakyReLU(0.1))

        self.flow2 = predict_flow_block(128, num_outputs=4)


    def forward(self, image_pair, prev_predictions = None):
        """
        image_pair: Tensor
            Image pair concatenated along the channel axis.

        image2_2: Tensor
            Second image at resolution level 2 (downsampled two times)

        intrinsics: Tensor
            The normalized intrinsic parameters

        prev_predictions: dict of Tensor
            Predictions from the previous depth block
        """

        conv1 = self.conv1(image_pair)
        conv2 = self.conv2(conv1)

        if self.num_prev_channels > 0:
            """use torch.cat to concatenate tensors"""
            extra = self.conv2_extra_inputs( prev_predictions )
            conv2 = torch.cat((conv2, extra), 1)
            pass
        
        conv2_1 = self.conv2_1( conv2 )
            
        conv3 = self.conv3( conv2_1 )
        conv3_1 = self.conv3_1( conv3 )
        conv4 = self.conv4( conv3_1 )
        conv4_1 = self.conv4_1( conv4 )
        conv5 = self.conv5( conv4_1 )
        conv5_1 = self.conv5_1( conv5 )

        upconv1 = self.upconv1(conv5_1)
        flow1 = self.flow1(conv5_1)
        flow1_upconv = self.flow1_upconv(flow1)

        """ concatenation along the channel axis """
        upconv2 = self.upconv2( torch.cat( (upconv1, conv4_1, flow1_upconv), 1 ) )
        upconv3 = self.upconv3( torch.cat( (upconv2, conv3_1), 1 ) )
        flow2 = self.flow2( torch.cat( (upconv3, conv2_1), 1) )

        """flow2 combines flow and flow confidence"""
        return flow2


"""
Refinement Block
"""
class RefinementBlock(nn.Module):

    def __init__(self):

        super(RefinementBlock, self).__init__()

        self.conv0 = convrelu_block(4, 32, (3,3), (1,1), 0.1)
        self.conv1 = convrelu_block(32, 64, (3,3), (2,2), 0.1)
        self.conv1_1 = convrelu_block(64, 64, (3,3), (1,1), 0.1)

        self.conv2 = convrelu_block(64, 128, (3,3), (2,2), 0.1)
        self.conv2_1 = convrelu_block(128, 128, (3,3), (1,1), 0.1)

        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (4,4), stride=(2,2), padding=1),
            nn.LeakyReLU(0.1)
        )

        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, (4,4), stride=(2,2), padding=1),
            nn.LeakyReLU(0.1)
        )

        self.depth_refine = predict_flow_block(64, num_outputs=1, intermediate_num_outputs=16)

    def forward(self, image, prev_result):

        """
        :param image1:
        :param depth:
        :return:
        """

        """
        fix me, update upsampling
        """
        conv0 = self.conv0(input)
        conv1 = self.conv1(conv0)
        conv1_1 = self.conv1_1(conv1)

        conv2 = self.conv2(conv1_1)
        conv2_1 = self.conv2_1(conv2)

        upconv1 = self.upconv1(conv2_1)
        upconv2 = self.upconv2( torch.cat((upconv1, conv1_1), 1) )

        depth_refine = self.depth_refine( torch.cat((upconv2, conv0), 1) )

        return depth_refine



"""
RefinementNet, refine depth output
"""
class RefinementNet(nn.Module):

    def __init__(self, options):
        super(RefinementNet, self).__init__()
        self.options = options
        self.refinement_block = RefinementBlockMask(options)

        if 'large' in options.suffix:
            self.upsample = torch.nn.Upsample(size=(480, 640), mode='bilinear')
            self.plane_to_depth = PlaneToDepth(normalized_K=True, W=640, H=480)
        else:
            self.upsample = torch.nn.Upsample(size=(192, 256), mode='bilinear')            
            self.plane_to_depth = PlaneToDepth(normalized_K=True, W=256, H=192)
            pass
        return
    
    def forward(self, image_1, camera, prev_result):
        masks = prev_result['mask']

        if 'refine_only' in self.options.suffix:
            with torch.no_grad():
                prev_predictions = torch.cat([torch.cat([prev_result['plane_depth'], prev_result['depth']], dim=1).repeat((len(masks), 1, 1, 1)), masks, (masks.sum(0, keepdim=True) - masks)[:, :1]], dim=1)
        else:
            prev_predictions = torch.cat([torch.cat([prev_result['plane_depth'], prev_result['depth']], dim=1).repeat((len(masks), 1, 1, 1)), masks, (masks.sum(0, keepdim=True) - masks)[:, :1]], dim=1)
            pass
        
        masks, depth, plane = self.refinement_block(image_1.repeat((len(masks), 1, 1, 1)), prev_predictions, prev_result['plane'])
        result = {}

        result = {'plane': plane, 'depth': depth}

        plane_depths, plane_XYZ = self.plane_to_depth(camera[0], result['plane'], return_XYZ=True)
        all_depths = torch.cat([result['depth'].squeeze(1), plane_depths], dim=0)
        
        all_masks = torch.softmax(masks, dim=0)
        plane_depth = (all_depths * all_masks).sum(0, keepdim=True)

        result['mask'] = masks.unsqueeze(1)
        result['plane_depth'] = plane_depth.unsqueeze(1)
        result['all_depths'] = all_depths
        result['all_masks'] = all_masks
        
        all_masks_one_hot = (all_masks.max(0, keepdim=True)[1] == torch.arange(len(all_masks)).cuda().long().view(-1, 1, 1)).float()
        plane_depth_one_hot = (all_depths * all_masks_one_hot).sum(0, keepdim=True)
        result['plane_depth_one_hot'] = plane_depth_one_hot.unsqueeze(1)
        return result

class RefinementNetConcat(nn.Module):

    def __init__(self, options):
        super(RefinementNetConcat, self).__init__()
        self.options = options
        self.refinement_block = RefinementBlockConcat(options)
        
        self.upsample = torch.nn.Upsample(size=(192, 256), mode='bilinear')
        self.plane_to_depth = PlaneToDepth(normalized_K=True, W=256, H=192)        
        return
    
    def forward(self, image_1, camera, prev_result):
        masks = prev_result['mask']
        masks = masks[:, :1].view((1, -1, int(masks.shape[2]), int(masks.shape[3])))
        masks = torch.cat([torch.clamp(1 - masks.sum(1, keepdim=True), min=0), masks], dim=1)
        max_num_planes = 30
        num_planes = int(masks.shape[1])
        if num_planes < max_num_planes:
            masks = torch.cat([masks, torch.zeros((1, max_num_planes - num_planes, int(masks.shape[2]), int(masks.shape[3]))).cuda()], dim=1)
            pass
        prev_predictions = torch.cat([image_1, prev_result['plane_depth'], prev_result['depth'], masks], dim=1)
        masks = self.refinement_block(prev_predictions)
        masks = masks[0, :num_planes]
        result = {}
        
        result = {'plane': prev_result['plane'], 'depth': prev_result['depth']}
        plane_depths, plane_XYZ = self.plane_to_depth(camera[0], result['plane'], return_XYZ=True)
        all_depths = torch.cat([result['depth'].squeeze(1), plane_depths], dim=0)

        all_masks = torch.softmax(masks, dim=0)
        plane_depth = (all_depths * all_masks).sum(0, keepdim=True)

            
        result['mask'] = masks.unsqueeze(1)
        result['plane_depth'] = plane_depth.unsqueeze(1)
        result['all_depths'] = all_depths
        result['all_masks'] = all_masks
        
        all_masks_one_hot = (all_masks.max(0, keepdim=True)[1] == torch.arange(len(all_masks)).cuda().long().view(-1, 1, 1)).float()
        plane_depth_one_hot = (all_depths * all_masks_one_hot).sum(0, keepdim=True)
        result['plane_depth_one_hot'] = plane_depth_one_hot.unsqueeze(1)
        return result    


def loadStateDict(flow_net, depth_net, state_dict):
    flow_net_state = flow_net.state_dict()
    depth_net_state = depth_net.state_dict()

    def reshapeTensor(source, target):
        for dim in range(len(source.shape)):
            if source.shape[dim] < target.shape[dim]:
                new_target = torch.cat([source, source.index_select(dim, torch.zeros(target.shape[dim] - source.shape[dim]).long())], dim=dim)
                break                                
            elif source.shape[dim] > target.shape[dim]:
                new_target = source.index_select(dim, torch.arange(target.shape[dim]).long())
                break
        return new_target
    
    new_flow_net_state = {}
    new_depth_net_state = {}        
    for k, v in state_dict.items():
        if 'flow_block' in k:
            name = k.replace('flow_block.', '')
            if name in flow_net_state:
                if v.shape == flow_net_state[name].shape:
                    new_flow_net_state[name] = v
                else:
                    new_flow_net_state[name] = reshapeTensor(v, flow_net_state[name])
                    pass
            else:
                print('flow', name in flow_net_state, name)
                assert(False)
                pass
            pass
        elif 'depth_motion_block' in k:
            name = k.replace('depth_motion_block.', '')
            if name in depth_net_state:
                if v.shape == depth_net_state[name].shape:
                    new_depth_net_state[name] = v
                else:
                    new_depth_net_state[name] = reshapeTensor(v, depth_net_state[name])
                    pass
            else:
                print('depth', name in depth_net_state, name, v.shape)
                pass
            pass
        else:
            print('not exist', k)
            assert(False)                    
            pass
        continue
    flow_net_state.update(new_flow_net_state)
    flow_net.load_state_dict(flow_net_state)
    depth_net_state.update(new_depth_net_state)
    depth_net.load_state_dict(depth_net_state)
    return

class RefineModel(nn.Module):
    def __init__(self, options):
        super(RefineModel, self).__init__()

        self.options = options
        
        K = [[0.89115971,  0,  0.5],
             [0,  1.18821287,  0.5],
             [0,           0,    1]]
        with torch.no_grad():
            self.intrinsics = torch.Tensor(K).cuda()
            pass
        """ the whole network """

        if 'concat' in self.options.suffix:
            self.refinement_net = RefinementNetConcat(options)
        else:
            self.refinement_net = RefinementNet(options)
            pass

        W, H = 64, 48
        self.upsample = torch.nn.Upsample(size=(H, W), mode='bilinear')

        if 'crfrnn_only' in self.options.suffix:
            self.plane_to_depth = PlaneToDepth(normalized_K = True, W=256, H=192)
            self.crfrnn = CRFRNNModule(image_dims=(192, 256), num_iterations=5)
            pass
        return

    
    def forward(self, image_1, image_2, camera, masks, planes, plane_depth, depth_np, gt_dict={}):
        results = []
        result = {'plane': planes, 'mask': masks[:, 0], 'depth': depth_np.unsqueeze(1), 'plane_depth': depth_np.unsqueeze(1)}
        results.append(result)

        if 'crfrnn_only' in self.options.suffix:
            detection_masks = masks[:, 0]
            background_mask = torch.clamp(1 - detection_masks.sum(0, keepdim=True), min=0)
            all_masks = torch.cat([background_mask, detection_masks], dim=0)
            all_masks = torch.clamp(all_masks, min=1e-4, max=1 - 1e-4)
            logits = torch.log(all_masks / (1 - all_masks))
            all_masks = self.crfrnn([logits, ((image_1[0] + 0.5) * 255).cpu()])
            masks = all_masks[1:].unsqueeze(1)
            
            plane_depths, plane_XYZ = self.plane_to_depth(camera[0], result['plane'], return_XYZ=True)

            all_depths = torch.cat([result['depth'].squeeze(1), plane_depths], dim=0)
            plane_depth = (all_depths * all_masks).sum(0, keepdim=True)

            all_masks_one_hot = (all_masks.max(0, keepdim=True)[1] == torch.arange(len(all_masks)).cuda().long().view(-1, 1, 1)).float()
            plane_depth_one_hot = (all_depths * all_masks_one_hot).sum(0, keepdim=True)
            result = {'mask': masks, 'plane': planes, 'depth': depth_np.unsqueeze(1), 'plane_depth': plane_depth.unsqueeze(1), 'plane_depth_one_hot': plane_depth_one_hot.unsqueeze(1)}
            results.append(result)        
            return results            
        else:
            result = {'mask': masks, 'plane': planes, 'depth': depth_np.unsqueeze(1), 'plane_depth': depth_np.unsqueeze(1)}
            pass

        result = self.refinement_net(image_1, camera, result)
        results.append(result)        
        return results
