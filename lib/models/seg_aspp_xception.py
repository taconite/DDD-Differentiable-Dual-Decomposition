"""
Xception architecture and weights are borrowed from [this repo](https://github.com/Cadene/pretrained-models.pytorch)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from lib.layers.fixed_point_iteration import FixedPointIterationGapN
from .aspp import ASPP

from .sync_bn.inplace_abn.bn import InPlaceABNSync

BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

bn_mom = 0.0003
__all__ = ['xception']

logger = logging.getLogger(__name__)

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False,activate_first=True):
        super(SeparableConv2d,self).__init__()
        self.relu0 = nn.ReLU(inplace=False)
        self.depthwise = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.bn1 = BatchNorm2d(in_channels, momentum=bn_mom)
        self.relu1 = nn.ReLU(inplace=False)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
        self.bn2 = BatchNorm2d(out_channels, momentum=bn_mom)
        self.relu2 = nn.ReLU(inplace=False)
        self.activate_first = activate_first

    def forward(self,x):
        if self.activate_first:
            x = self.relu0(x)

        x = self.depthwise(x)
        x = self.bn1(x)
        if not self.activate_first:
            x = self.relu1(x)

        x = self.pointwise(x)
        x = self.bn2(x)
        if not self.activate_first:
            x = self.relu2(x)

        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,strides=1,atrous=None,grow_first=True,activate_first=True):
        super(Block, self).__init__()
        if atrous == None:
            atrous = [1]*3
        elif isinstance(atrous, int):
            atrous_list = [atrous]*3
            atrous = atrous_list
        idx = 0
        # self.head_relu = True
        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = BatchNorm2d(out_filters, momentum=bn_mom)
            # self.head_relu = False
        else:
            self.skip=None

        if grow_first:
            filters = out_filters
        else:
            filters = in_filters

        self.sepconv1 = SeparableConv2d(in_filters,filters,3,stride=1,padding=1*atrous[0],dilation=atrous[0],bias=False,activate_first=activate_first)
        self.sepconv2 = SeparableConv2d(filters,out_filters,3,stride=1,padding=1*atrous[1],dilation=atrous[1],bias=False,activate_first=activate_first)
        self.sepconv3 = SeparableConv2d(out_filters,out_filters,3,stride=strides,padding=1*atrous[2],dilation=atrous[2],bias=False,activate_first=activate_first)

    def forward(self,inp):

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = self.sepconv1(inp)
        x = self.sepconv2(x)
        x = self.sepconv3(x)

        x = x + skip
        return x


class SegXception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, cfg, **kwargs):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        extra = cfg.MODEL.EXTRA

        super(SegXception, self).__init__()

        stride_list = None
        if cfg.MODEL.OUTPUT_STRIDE == 8:
            stride_list = [2,1,1]
        elif cfg.MODEL.OUTPUT_STRIDE == 16:
            stride_list = [2,2,1]
        else:
            raise ValueError( 'xception.py: output stride={} is not supported.'.format(cfg.MODEL.OUTPUT_STRIDE) )
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = BatchNorm2d(32, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(32,64,3,1,1,bias=False)
        self.bn2 = BatchNorm2d(64, momentum=bn_mom)
        #do relu here

        self.block1=Block(64,128,2)
        self.block2=Block(128,256,stride_list[0])
        self.block3=Block(256,728,stride_list[1])

        rate = 16 // cfg.MODEL.OUTPUT_STRIDE
        self.block4=Block(728,728,1,atrous=rate)
        self.block5=Block(728,728,1,atrous=rate)
        self.block6=Block(728,728,1,atrous=rate)
        self.block7=Block(728,728,1,atrous=rate)

        self.block8=Block(728,728,1,atrous=rate)
        self.block9=Block(728,728,1,atrous=rate)
        self.block10=Block(728,728,1,atrous=rate)
        self.block11=Block(728,728,1,atrous=rate)

        self.block12=Block(728,728,1,atrous=rate)
        self.block13=Block(728,728,1,atrous=rate)
        self.block14=Block(728,728,1,atrous=rate)
        self.block15=Block(728,728,1,atrous=rate)

        self.block16=Block(728,728,1,atrous=[1*rate,1*rate,1*rate])
        self.block17=Block(728,728,1,atrous=[1*rate,1*rate,1*rate])
        self.block18=Block(728,728,1,atrous=[1*rate,1*rate,1*rate])
        self.block19=Block(728,728,1,atrous=[1*rate,1*rate,1*rate])

        self.block20=Block(728,1024,stride_list[2],atrous=rate,grow_first=False)
        #self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1*rate,dilation=rate,activate_first=False)
        # self.bn3 = BatchNorm2d(1536, momentum=bn_mom)

        self.conv4 = SeparableConv2d(1536,1536,3,1,1*rate,dilation=rate,activate_first=False)
        # self.bn4 = BatchNorm2d(1536, momentum=bn_mom)

        #do relu here
        self.conv5 = SeparableConv2d(1536,2048,3,1,1*rate,dilation=rate,activate_first=False)
        # self.bn5 = BatchNorm2d(2048, momentum=bn_mom)
        self.layers = []

        # ASPP layer
        self.aspp = ASPP(2048, 256, norm=BatchNorm2d, momentum=bn_mom)

        self.unary_layer = nn.Conv2d(
            in_channels=256,
            out_channels=extra.NUM_CLASSES,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.num_classes = extra.NUM_CLASSES

        self.pairwise_layer = nn.Conv2d(
            in_channels=256 * 2,
            out_channels=extra.NUM_CLASSES ** 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

        self.fixed_point_iteration = FixedPointIterationGapN(extra.NUM_CLASSES, cfg.MODEL.LEARN_GAMMA)

        self.max_iter = cfg.MODEL.FPI_ITERS

        self.use_pairwise = cfg.MODEL.LEARN_PAIRWISE_TERMS
        self.use_unary_loss = cfg.LOSS.USE_UNARY_LOSS

        self.num_pw_terms = cfg.MODEL.NUM_PAIRWISE_TERMS
        self.pw_step_size = cfg.MODEL.PAIRWISE_STEP_SIZE

        if cfg.MODEL.LEARN_OUTPUT_TEMPERATURE:
            self.gamma = torch.nn.Parameter(cfg.MODEL.INIT_OUTPUT_TEMPERATURE * torch.ones(1))
        else:
            self.gamma = 1.0

        self.use_all_fpi_loss = cfg.LOSS.USE_ALL_FPI_LOSS

        # self.alphas = nn.ParameterList([torch.nn.Parameter(max(2 ** (cfg.MODEL.FPI_ITERS - i - 2), 1) * torch.ones(1)) for i in range(cfg.MODEL.FPI_ITERS)])

        # #------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, InPlaceABNSync):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------

    def forward(self, input, gamma=None):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)
        x = self.block20(x)

        x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu(x)

        x = self.conv4(x)
        # x = self.bn4(x)
        # x = self.relu(x)

        x = self.conv5(x)
        # x = self.bn5(x)
        # x = self.relu(x)

        x = self.aspp(x)
        unary = self.unary_layer(x)

        if self.use_pairwise:
            horizontal_features = []
            vertical_features = []
            horizontal_unaries = []
            vertical_unaries = []

            denom = self.num_pw_terms * 2

            # stride 32 and beyond pairwise connections
            stride = 1
            for _ in range(self.num_pw_terms):
                for i in range(stride):
                    for j in range(stride):
                        horizontal_features.append(
                            torch.cat(
                                [x[:, :, i::stride, j::stride][:, :, :, :-1], x[:, :, i::stride, j::stride][:, :, :, 1:]],
                                dim=1
                            )
                        )
                        vertical_features.append(
                            torch.cat(
                                [x[:, :, i::stride, j::stride][:, :, :-1, :], x[:, :, i::stride, j::stride][:, :, 1:, :]],
                                dim=1
                            )
                        )

                        horizontal_unaries.append(unary[:, :, i::stride, j::stride] / denom)
                        vertical_unaries.append(unary[:, :, i::stride, j::stride] / denom)

                stride += self.pw_step_size

            horizontal_pairwises = []
            vertical_pairwises = []

            for horizontal_feature, vertical_feature in zip(horizontal_features, vertical_features):
                horizontal_pairwises.append(self.pairwise_layer(horizontal_feature) \
                    .view(horizontal_feature.size(0),
                          self.num_classes, self.num_classes,
                          horizontal_feature.size(2),
                          horizontal_feature.size(3))
                )
                vertical_pairwises.append(self.pairwise_layer(vertical_feature) \
                    .view(vertical_feature.size(0),
                          self.num_classes, self.num_classes,
                          vertical_feature.size(2),
                          vertical_feature.size(3))
                )

            outputs = []

            if self.use_unary_loss:
                outputs.append(unary)

            prev_dual_obj = 0

            for iter in range(self.max_iter):
                # self.alphas[iter].data.clamp_(min=1.0)

                horizontal_unaries, vertical_unaries, \
                horizontal_marginals, vertical_marginals, \
                marginal_hs, marginal_vs= \
                    self.fixed_point_iteration(
                        horizontal_unaries,
                        vertical_unaries,
                        horizontal_pairwises,
                        vertical_pairwises,
                        gamma,
                        self.pw_step_size,
                        # self.alphas[iter],
                    )

                if iter == 0:
                    prev_horizontal_marginals = horizontal_marginals
                    prev_vertical_marginals = vertical_marginals

                with torch.no_grad():
                    dual_obj = 0
                    if gamma > 0:
                        for marginal_h, marginal_v in zip(marginal_hs, marginal_vs):
                            dual_obj += ((torch.logsumexp(marginal_h[:, :, :, 0] / gamma, dim=1) * gamma).sum() +
                                         (torch.logsumexp(marginal_v[:, :, 0, :] / gamma, dim=1) * gamma).sum())
                    else:
                        for marginal_h, marginal_v in zip(marginal_hs, marginal_vs):
                            dual_obj += (torch.max(marginal_h[:, :, :, 0], dim=1)[0].sum() +
                                        torch.max(marginal_v[:, :, 0, :], dim=1)[0].sum())

                    # print ("device {} iter {} dual obj: {}".format(horizontal_unaries[0].device.index, iter, dual_obj.item()))

                    if iter == 0:
                        last_change = dual_obj
                        prev_dual_obj = dual_obj
                        continue

                    if prev_dual_obj - dual_obj <= -1e-4:
                        break
                    else:
                        last_change = prev_dual_obj - dual_obj
                        prev_dual_obj = dual_obj

                if iter > 0:
                    prev_horizontal_marginals = horizontal_marginals
                    prev_vertical_marginals = vertical_marginals

                # if self.use_all_fpi_loss:
                #     outputs.append((horizontal_marginals + vertical_marginals) * self.gamma)

            # if not self.use_all_fpi_loss:
            outputs.append((prev_horizontal_marginals + prev_vertical_marginals) * self.gamma)

            horizontal_sol = torch.argmax(horizontal_marginals, dim=1)
            vertical_sol = torch.argmax(vertical_marginals, dim=1)
            disagreement = horizontal_sol != vertical_sol

            # # Computing primal objective
            # xv, yv, zv = torch.meshgrid([torch.arange(0, unary.size(0)), torch.arange(0, unary.size(-2)), torch.arange(0, unary.size(-1))])
            # primal_sol = torch.argmax(prev_horizontal_marginals + prev_vertical_marginals, dim=1)
            # primal_obj = unary[xv, primal_sol, yv, zv].sum()

            # stride = 1
            # cnt = 0
            # for _ in range(self.num_pw_terms):
            #     for i in range(stride):
            #         for j in range(stride):
            #             batch_size = horizontal_unaries[cnt].size(0)
            #             width = horizontal_unaries[cnt].size(-1)
            #             height = horizontal_unaries[cnt].size(-2)
            #             xv, yv, zv = torch.meshgrid([torch.arange(0, batch_size), torch.arange(0, height), torch.arange(0, width)])
            #             primal_sol_ = primal_sol[:, i::stride, j::stride]
            #             primal_obj += horizontal_pairwises[cnt][xv[:, :, :width-1], primal_sol_[:, :, :width-1], primal_sol_[:, :, 1:], yv[:, :, :width-1], zv[:, :, :width-1]].sum() + \
            #                          vertical_pairwises[cnt][xv[:, :height-1, :], primal_sol_[:, :height-1, :], primal_sol_[:, 1:, :], yv[:, :height-1, :], zv[:, :height-1, :]].sum()
            #
            #             cnt += 1

            #     stride += self.pw_step_size

            # print ("device {} dual obj {}, primal obj {}".format(horizontal_unaries[0].device.index, dual_obj.item(), primal_obj.item()))

            # return outputs, horizontal_pairwises, vertical_pairwises, disagreement
            return outputs, disagreement, iter * torch.ones(1, device=disagreement.device)
            # return outputs, disagreement, dual_obj - primal_obj
        else:
            return unary * self.gamma

    def init_weights(self, init_deconvs, pretrained=''):
        if os.path.isfile(pretrained):
            for name, m in self.aspp.named_modules():
                if isinstance(m, InPlaceABNSync):
                    logger.info('=> init aspp.{}.weight as 1'.format(name))
                    logger.info('=> init aspp.{}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            # Initialize final classification layers with zero-mean gaussian weights and 0 bias,
            # contrary to default pytorch initialization which assumes ensuing ReLU activation
            # for name, m in self.unary_layer.named_modules():
            #     if isinstance(m, InPlaceABNSync):
            #         logger.info('=> init unary_layer.{}.weight as 1'.format(name))
            #         logger.info('=> init unary_layer.{}.bias as 0'.format(name))
            #         nn.init.constant_(m.weight, 1)
            #         nn.init.constant_(m.bias, 0)

            # assert (isinstance(self.unary_layer[-1], nn.Conv2d))
            # logger.info('=> init unary_layer.{}.weight as normal(0, 0.0625)'.format(len(self.unary_layer) - 1))
            # logger.info('=> init unary_layer.{}.bias as 0'.format(len(self.unary_layer) - 1))
            # nn.init.normal_(self.unary_layer[-1].weight, std=0.0625)
            # nn.init.constant_(self.unary_layer[-1].bias, 0)

            assert (isinstance(self.unary_layer, nn.Conv2d))
            logger.info('=> init unary_layer.weight as normal(0, 0.001)')
            logger.info('=> init unary_layer.bias as 0')
            nn.init.normal_(self.unary_layer.weight, std=0.001)
            nn.init.constant_(self.unary_layer.bias, 0)

            # for name, m in self.pairwise_layer.named_modules():
            #     if isinstance(m, InPlaceABNSync):
            #         logger.info('=> init pairwise_layer.{}.weight as 1'.format(name))
            #         logger.info('=> init pairwise_layer.{}.bias as 0'.format(name))
            #         nn.init.constant_(m.weight, 1)
            #         nn.init.constant_(m.bias, 0)

            # assert (isinstance(self.pairwise_layer[-1], nn.Conv2d))
            # logger.info('=> init pairwise_layer.{}.weight as normal(0, 0.045)'.format(len(self.pairwise_layer) - 1))
            # logger.info('=> init pairwise_layer.{}.bias as 0'.format(len(self.pairwise_layer) - 1))
            # nn.init.normal_(self.pairwise_layer[-1].weight, std=0.045)
            # nn.init.constant_(self.pairwise_layer[-1].bias, 0)

            assert (isinstance(self.pairwise_layer, nn.Conv2d))
            logger.info('=> init pairwise_layer.weight as normal(0, 0.001)')
            logger.info('=> init pairwise_layer.bias as 0')
            nn.init.normal_(self.pairwise_layer.weight, std=0.001)
            nn.init.constant_(self.pairwise_layer.bias, 0)

            # pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            # self.load_state_dict(pretrained_state_dict, strict=False)
            checkpoint = torch.load(pretrained)
            if isinstance(checkpoint, OrderedDict):
                state_dict_old = checkpoint
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict_old = checkpoint['state_dict']
            else:
                raise RuntimeError(
                    'No state_dict found in checkpoint file {}'.format(pretrained))

            state_dict = OrderedDict()
            # delete 'module.' because it is saved from DataParallel module
            state_dict_old = {k: v for k,v in state_dict_old.items() if ('itr' not in k and 'tmp' not in k and 'track' not in k)}
            for key in state_dict_old.keys():
                if key.startswith('module.'):
                    # state_dict[key[7:]] = state_dict[key]
                    # state_dict.pop(key)
                    state_dict[key[7:]] = state_dict_old[key]
                else:
                    state_dict[key] = state_dict_old[key]

            self.load_state_dict(state_dict, strict=False)
        else:
            logger.error('=> imagenet pretrained model dose not exist')
            logger.error('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')

def get_seg_net(cfg, is_train, **kwargs):
    model = SegXception(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.INIT_DECONVS,
                           cfg.MODEL.PRETRAINED)

    return model
