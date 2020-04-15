from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import torch
import torch.nn as nn
from collections import OrderedDict

from lib.layers.fixed_point_iteration import FixedPointIterationGapN
from .aspp import ASPP

from .sync_bn.inplace_abn.bn import InPlaceABNSync

BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')


BACKBONE_BN_MOMENTUM = 0.0003
HEAD_BN_MOMENTUM = 0.05
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     dilation=dilation, padding=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = BatchNorm2d(planes, momentum=BACKBONE_BN_MOMENTUM)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BACKBONE_BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BACKBONE_BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BACKBONE_BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BACKBONE_BN_MOMENTUM)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck_CAFFE(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_CAFFE, self).__init__()
        # add stride to conv1x1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BACKBONE_BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BACKBONE_BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BACKBONE_BN_MOMENTUM)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class SegResNet(nn.Module):

    def __init__(self, block, layers, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA

        super(SegResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BACKBONE_BN_MOMENTUM)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # output stride is 8x
        if cfg.MODEL.OUTPUT_STRIDE == 16:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # output stride is 16x
            # We change stride of res5 to 1 and use dilation rate of 2
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2) # output stride is 16x
        elif cfg.MODEL.OUTPUT_STRIDE == 8:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2) # output stride is 8x
            # We change stride of res5 to 1 and use dilation rate of 2
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4) # output stride is 8x

        # ASPP layer
        self.aspp = ASPP(self.inplanes, 256, norm=BatchNorm2d, momentum=HEAD_BN_MOMENTUM)

        if cfg.MODEL.UNARY_BN:
            self.unary_layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                BatchNorm2d(256, momentum=HEAD_BN_MOMENTUM),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=256,
                    out_channels=extra.NUM_CLASSES,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                ),
            )
        else:
            self.unary_layer = nn.Conv2d(
                in_channels=256,
                out_channels=extra.NUM_CLASSES,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )

        self.train_unary_bn = cfg.MODEL.UNARY_BN

        self.num_classes = extra.NUM_CLASSES

        if cfg.MODEL.PAIRWISE_BN:
            self.pairwise_layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=256 * 2,
                    out_channels=256 * 2,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                BatchNorm2d(256 * 2, momentum=HEAD_BN_MOMENTUM),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=256 * 2,
                    out_channels=extra.NUM_CLASSES ** 2,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True
                ),
            )
        else:
            self.pairwise_layer = nn.Conv2d(
                in_channels=256 * 2,
                out_channels=extra.NUM_CLASSES ** 2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            )

        self.train_pairwise_bn = cfg.MODEL.PAIRWISE_BN

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

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BACKBONE_BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, gamma=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

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

            # with torch.no_grad():
            #     # Computing primal objective
            #     xv, yv, zv = torch.meshgrid([torch.arange(0, unary.size(0)), torch.arange(0, unary.size(-2)), torch.arange(0, unary.size(-1))])
            #     primal_sol = torch.argmax(prev_horizontal_marginals + prev_vertical_marginals, dim=1)
            #     primal_obj = unary[xv, primal_sol, yv, zv].sum()

            #     stride = 1
            #     cnt = 0
            #     for _ in range(self.num_pw_terms):
            #         for i in range(stride):
            #             for j in range(stride):
            #                 batch_size = horizontal_unaries[cnt].size(0)
            #                 width = horizontal_unaries[cnt].size(-1)
            #                 height = horizontal_unaries[cnt].size(-2)
            #                 xv, yv, zv = torch.meshgrid([torch.arange(0, batch_size), torch.arange(0, height), torch.arange(0, width)])
            #                 primal_sol_ = primal_sol[:, i::stride, j::stride]
            #                 primal_obj += horizontal_pairwises[cnt][xv[:, :, :width-1], primal_sol_[:, :, :width-1], primal_sol_[:, :, 1:], yv[:, :, :width-1], zv[:, :, :width-1]].sum() + \
            #                              vertical_pairwises[cnt][xv[:, :height-1, :], primal_sol_[:, :height-1, :], primal_sol_[:, 1:, :], yv[:, :height-1, :], zv[:, :height-1, :]].sum()

            #                 cnt += 1

            #         stride += self.pw_step_size

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
                    # if "aspp" in name:
                    logger.info('=> init aspp.{}.weight as 1'.format(name))
                    logger.info('=> init aspp.{}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                    # else:
                    #     logger.info('=> init aspp.{}.weight as 0.1'.format(name))
                    #     logger.info('=> init aspp.{}.bias as 0'.format(name))
                    #     nn.init.constant_(m.weight, 0.1)
                    #     nn.init.constant_(m.bias, 0)

            # Initialize final classification layers with zero-mean gaussian weights and 0 bias,
            # contrary to default pytorch initialization which assumes ensuing ReLU activation
            if self.train_unary_bn:
                for name, m in self.unary_layer.named_modules():
                    if isinstance(m, InPlaceABNSync):
                        logger.info('=> init unary_layer.{}.weight as 1'.format(name))
                        logger.info('=> init unary_layer.{}.bias as 0'.format(name))
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)

                assert (isinstance(self.unary_layer[-1], nn.Conv2d))
                logger.info('=> init unary_layer.{}.weight as normal(0, 0.001)'.format(len(self.unary_layer) - 1))
                logger.info('=> init unary_layer.{}.bias as 0'.format(len(self.unary_layer) - 1))
                nn.init.normal_(self.unary_layer[-1].weight, std=0.001)
                nn.init.constant_(self.unary_layer[-1].bias, 0)
            else:
                assert (isinstance(self.unary_layer, nn.Conv2d))
                logger.info('=> init unary_layer.weight as normal(0, 0.001)')
                logger.info('=> init unary_layer.bias as 0')
                nn.init.normal_(self.unary_layer.weight, std=0.001)
                nn.init.constant_(self.unary_layer.bias, 0)

            if self.train_pairwise_bn:
                for name, m in self.pairwise_layer.named_modules():
                    if isinstance(m, InPlaceABNSync):
                        logger.info('=> init pairwise_layer.{}.weight as 1'.format(name))
                        logger.info('=> init pairwise_layer.{}.bias as 0'.format(name))
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)

                assert (isinstance(self.pairwise_layer[-1], nn.Conv2d))
                logger.info('=> init pairwise_layer.{}.weight as normal(0, 0.001)'.format(len(self.pairwise_layer) - 1))
                logger.info('=> init pairwise_layer.{}.bias as 0'.format(len(self.pairwise_layer) - 1))
                nn.init.normal_(self.pairwise_layer[-1].weight, std=0.001)
                nn.init.constant_(self.pairwise_layer[-1].bias, 0)
            else:
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


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_seg_net(cfg, is_train, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
    style = cfg.MODEL.STYLE

    block_class, layers = resnet_spec[num_layers]

    if style == 'caffe':
        block_class = Bottleneck_CAFFE

    model = SegResNet(block_class, layers, cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.INIT_DECONVS,
                           cfg.MODEL.PRETRAINED)

    return model
