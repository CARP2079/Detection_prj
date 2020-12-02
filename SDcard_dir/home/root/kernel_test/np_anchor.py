# -*- coding: utf-8 -*-
# Copyright 2016 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Definition of 512-MobileNet V2-SSD network.

Nevertheless, this is not the case anymore for other kernel sizes, hence
motivating the use of special padding layer for controlling these side-effects.

@@ssd_vgg_300
"""
import math
import numpy as np
from collections import namedtuple

SSDParams = namedtuple('SSDParameters', ['img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'
                                         ])

default_params = SSDParams(
        img_shape=(448, 768),
        num_classes=10,
        no_annotation_label=2,
        feat_layers=['block8', 'block7', 'block6','block5'],
        feat_shapes=[(56, 96), (28, 48), (14, 24), (7, 12)],
        anchor_size_bounds=[0.10, 0.90],
        anchor_sizes=[(10.,20.),  # 9   13
                      (30.,38.),   # 22   33  (22., 50),  
                      (53.,70.),  # 45  58
                      (100.,160.)], # 140   174
        anchor_ratios=[[0.53, 1./0.53],
                       [0.57, 1./0.57],
                       [0.60, 1./0.60],
                       [0.75, 1./0.75]],
        anchor_steps=[8, 16, 32, 64],
        anchor_offset=0.5,
        normalizations=[-1, -1, -1, -1],
        prior_scaling=[0.1, 0.1, 0.2, 0.2]
        )

def anchors(img_shape, dtype=np.float32):
    """Compute the default anchor boxes, given an image shape.
    """
    return ssd_anchors_all_layers(img_shape,
                                  default_params.feat_shapes,
                                  default_params.anchor_sizes,
                                  default_params.anchor_ratios,
                                  default_params.anchor_steps,
                                  default_params.anchor_offset,
                                  dtype)

def ssd_anchor_one_layer(img_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset=0.5,
                         dtype=np.float32):
    """Computer SSD default anchor boxes for one feature layer.
    Return:
      y, x, h, w: Relative x and y grids, and height and width.
    """
    # Compute the position grid: simple way.
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(sizes) * len(ratios) + 2
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)
    # Add first anchor boxes with ratio=1.
    h[0] = sizes[0] / img_shape[0]
    w[0] = sizes[0] / img_shape[1]
    di = 1
    if len(sizes) > 1:
        h[1] = sizes[1] / img_shape[0]#math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        w[1] = sizes[1] / img_shape[1]#math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1
    for i, r in enumerate(ratios):
        h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
        di += 1
        h[i+di] = sizes[1] / img_shape[0] / math.sqrt(r)
        w[i+di] = sizes[1] / img_shape[1] * math.sqrt(r)
        
    return y, x, h, w


def ssd_anchors_all_layers(img_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset=0.5,
                           dtype=np.float32):
    """Compute anchor boxes for all feature layers.
    """
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = ssd_anchor_one_layer(img_shape, s,
                                             anchor_sizes[i],
                                             anchor_ratios[i],
                                             anchor_steps[i],
                                             offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors

def out_reshape(data,data_height ,data_width):
    return np.reshape(data,(1, data_height, data_width, 6, -1))
'''
# network
# =========================================================================== #
# Functional definition of VGG-based SSD 300.
# =========================================================================== #
def tensor_shape(x, rank=3):
    """Returns the dimensions of a tensor.
    Args:
      image: A N-D Tensor of shape.
    Returns:
      A list of dimensions. Dimensions that are statically known are python
        integers,otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


def ssd_multibox_layer(inputs,
                       num_classes,
                       sizes,
                       ratios=[1],
                       normalization=-1,
                       bn_normalization=False,
                       is_training=True):
    """Construct a multibox layer, return a class and localization predictions.
    """
    net = inputs
#    net1 = ops.conv_1x1(net, 128, name='share_conv0', bias=False)
#    net1 = ops.conv_1x1(net1, 256, name='share_conv1', bias=False)
#    net2 = ops.conv_1x1(net, 256, name='share_conv2', bias=False)
#    net = net1 + net2
    
    if normalization > 0:
        net = custom_layers.l2_normalization(net, scaling=True)
    # Number of anchors.
    num_anchors = len(sizes) * len(ratios) + 2

    # Location.
    num_loc_pred = num_anchors * 4
    loc_pred = ops.conv2d_block(net, 128, 3, 1, name='conv_loc_1', bn = False, bias=True)
    loc_pred = ops.conv_1x1(loc_pred, num_loc_pred, name='conv_loc_2', bias=True)
    
    loc_pred = custom_layers.channel_to_last(loc_pred)
    loc_pred = tf.reshape(loc_pred,
                          tensor_shape(loc_pred, 4)[:-1]+[num_anchors, 4])
    """ loc_pred shape[N,H,W,num_anchors,4] """
    
    # Class prediction.
    num_cls_pred = num_anchors * num_classes
    cls_pred = ops.conv2d_block(net, 128, 3, 1, name='conv_cls_1', bn = False, bias=True)
    cls_pred = ops.conv_1x1(cls_pred, num_cls_pred, name='conv_cls_2', bias=True)
    
    cls_pred = custom_layers.channel_to_last(cls_pred)
    cls_pred = tf.reshape(cls_pred,
                          tensor_shape(cls_pred, 4)[:-1]+[num_anchors, num_classes]) # (N H W num_cls_pred)
    return cls_pred, loc_pred

def ssd_net(inputs,
            num_classes=SSDNet.default_params.num_classes,
            feat_layers=SSDNet.default_params.feat_layers,
            anchor_sizes=SSDNet.default_params.anchor_sizes,
            anchor_ratios=SSDNet.default_params.anchor_ratios,
            normalizations=SSDNet.default_params.normalizations,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ssd_mobilenet_v2'):
    """SSD net definition.
    """
    # if data_format == 'NCHW':
    #     inputs = tf.transpose(inputs, perm=(0, 3, 1, 2))

    # End_points collect relevant activations for external use.
    end_points = {}
    exp = 2
    with tf.variable_scope(scope, 'ssd_mobilenet_v2', [inputs], reuse=reuse):
        # Original VGG-16 blocks.
 ######################################################################       
        inputs = tf.identity(inputs,name='image')   #448*768

        net = ops.conv2d_block(inputs, 16, 3, 2, is_train=is_training,name ='conv2d')  #224*384
        
        #################### Block 1.    
        # GhostBottleneck(input, out_channel,name, stride=1, exp=3, is_train=True 
        net = ops.GhostBottleneck(net,16, name='ghost1_1', stride=1, exp=1, is_train=is_training)   
        net = ops.GhostBottleneck(net,24, name='ghost1_2', stride=2, exp=exp, is_train=is_training) 
        net = ops.GhostBottleneck(net,24, name='ghost1_3', stride=1, exp=exp, is_train=is_training)
        net = ops.GhostBottleneck(net,24, name='ghost1_4', stride=1, exp=exp, is_train=is_training)  #112*192
        end_points['block1'] = net               
        
        #################### Block 2.     
        net = ops.GhostBottleneck(net,40, name='ghost2_1', stride=2, exp=exp, is_train=is_training) 
        net = ops.GhostBottleneck(net,40, name='ghost2_2', stride=1, exp=exp, is_train=is_training)
        net = ops.GhostBottleneck(net,40, name='ghost2_3', stride=1, exp=exp, is_train=is_training)  # 56*96
        end_points['block2'] = net

        #################### Block 3.
        net = ops.GhostBottleneck(net,64, name='ghost3_1', stride=2, exp=exp, is_train=is_training) 
        net = ops.GhostBottleneck(net,64, name='ghost3_2', stride=1, exp=exp, is_train=is_training)
        net = ops.GhostBottleneck(net,64, name='ghost3_3', stride=1, exp=exp, is_train=is_training)  # 28*48
        end_points['block3'] = net

        #################### Block 4.
        net = ops.GhostBottleneck(net,80, name='ghost4_1', stride=2, exp=exp, is_train=is_training) 
        net = ops.GhostBottleneck(net,80, name='ghost4_2', stride=1, exp=exp, is_train=is_training)
        net = ops.GhostBottleneck(net,80, name='ghost4_3', stride=1, exp=exp, is_train=is_training)  # 14*24        
        end_points['block4'] = net
        
        #################### Block 5.
        net = ops.GhostBottleneck(net,120, name='ghost5_1', stride=2, exp=exp, is_train=is_training) 
        net = ops.GhostBottleneck(net,120, name='ghost5_2', stride=1, exp=exp, is_train=is_training)
        net = ops.GhostBottleneck(net,120, name='ghost5_3', stride=1, exp=exp, is_train=is_training) # 7*12      
        end_points['block5'] = net
                            
        # FPN                
        net = ops.FPN_plus_block(net,end_points['block4'], name='FPN_1',is_train=is_training) 
        end_points['block6'] = ops.conv2d_block(net, 80, 3, 1, name='FPN_1',is_train= is_training)
        
        net = ops.FPN_plus_block(net,end_points['block3'], name='FPN_2',is_train=is_training) 
        end_points['block7'] = ops.conv2d_block(net, 64, 3, 1, name='FPN_2',is_train= is_training)
        
        net = ops.FPN_plus_block(net,end_points['block2'], name='FPN_3',is_train=is_training)
        end_points['block8'] = ops.conv2d_block(net, 40, 3, 1, name='FPN_3',is_train= is_training)
        
        net = ops.FPN_plus_block(net,end_points['block1'], name='FPN_4',is_train=is_training)
        end_points['block9'] = ops.conv2d_block(net, 24, 3, 1, name='FPN_4',is_train= is_training)
        
        # bottom_up path augmentation
        
        # Prediction and localisations layers.
        predictions = []
        logits = []
        localisations = []
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope(layer + '_box'):
                p, l = ssd_multibox_layer(end_points[layer],
                                          num_classes,
                                          anchor_sizes[i],
                                          anchor_ratios[i],
                                          normalizations[i],
                                          is_training=is_training)
            name_p = 'logits' + str(i)
            name_l = 'localisations' + str(i)
            
            predictions.append(prediction_fn(p))
            logits.append(tf.identity(p,name=name_p))  #### out node for prediction without softmax
            localisations.append(tf.identity(l,name=name_l))#### out node for localisations
        
        return predictions, localisations, logits, end_points
'''
#ssd_net.default_image_size = 512
#
#if __name__ == '__main__':
#    net_shape = (448, 768)
#    ssd= SSDNet()
#    ssd_anchors = ssd.anchors(net_shape)
#    for i, s in enumerate(ssd_anchors):
#         y, x, h, w = s
#         save_mat_bin.save_array_bin(y,'y',i)
#         save_mat_bin.save_array_bin(x,'x',i)
#         save_mat_bin.save_array_bin(h,'h',i)
#         save_mat_bin.save_array_bin(w,'w',i)
