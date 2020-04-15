from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import os

import numpy as np
import torch
import torch.autograd as autograd

from lib.core.config import get_model_name
from lib.core.evaluate import calc_IoU
from lib.core.inference import get_final_preds
from lib.utils.vis import vis_segments

logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, lr_scheduler, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    if config.TRAIN.TRAIN_BN:
        model.train()
    else:
        model.eval()

    end = time.time()
    for i, (input, target, h_target, v_target, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if config.TRAIN.LR_SCHEDULER == 'poly':
            lr_scheduler.step()

        # compute output
        if config.MODEL.LEARN_PAIRWISE_TERMS:
            assert (config.TRAIN.LR_SCHEDULER == 'poly')
            if not config.MODEL.LEARN_GAMMA:
                if float(lr_scheduler.last_epoch) / (lr_scheduler.max_iter * config.TRAIN.NE_ITER_RATIO) <= 1:
                    gamma = (config.TRAIN.NE_GAMMA_U - config.TRAIN.NE_GAMMA_L) * \
                            (1 - float(lr_scheduler.last_epoch) / (lr_scheduler.max_iter * config.TRAIN.NE_ITER_RATIO) ) ** \
                            config.TRAIN.NE_GAMMA_EXP + config.TRAIN.NE_GAMMA_L
                else:
                    gamma = config.TRAIN.NE_GAMMA_L

                # outputs, h_output, v_output, _ = model(input, gamma)
                outputs, _, duality_gap = model(input, gamma)
            else:
                # outputs, h_output, v_output, _ = model(input)
                outputs, _ = model(input)
        else:
            output = model(input)

        target = target.cuda(non_blocking=True)

        if config.MODEL.LEARN_PAIRWISE_TERMS:
            if len(outputs) == 1:
                u_loss = criterion(outputs[0], target)
            else:
                # Supervise auxiliary losses in output resolution (e.g. 1/16 of the input)
                u_loss = [criterion(output, target, resize_scores=False) for output in outputs[:-1]]
                # Supervise final loss in original input resolution
                u_loss.append(criterion(outputs[-1], target, resize_scores=True))
                u_loss = torch.stack(u_loss, dim=0)


            # optimizer.zero_grad()
            # u_loss[0].backward(retain_graph=True)

            # w_grad0 = model.module.final_layer.weight.grad.detach().clone()
            # b_grad0 = model.module.final_layer.bias.grad.detach().clone()

            # optimizer.zero_grad()
            # u_loss[-1].backward(retain_graph=True)

            # w_grad1 = model.module.final_layer.weight.grad.detach().clone()
            # b_grad1 = model.module.final_layer.bias.grad.detach().clone()

            # assert torch.all(torch.abs(w_grad0 - w_grad1) < 1e-4)
            # assert torch.all(torch.abs(b_grad0 - b_grad1) < 1e-4)
            # optimizer.zero_grad()

            # u_loss = u_loss[1:]

            if config.LOSS.USE_PAIRWISE_LOSS:
                # we supervise pairwise terms at network output resolution (e.g. 1/16
                # of input image), as it would be too big to fit into GPU memory if
                # using input resolution
                # if isinstance(h_output, list):
                #     h_pw_loss = torch.stack(
                #         [criterion(o, t, resize_scores=False) for o, t in zip(h_output, h_target)],
                #         dim=0
                #     )
                #     v_pw_loss = torch.stack(
                #         [criterion(o, t, resize_scores=False) for o, t in zip(v_output, v_target)],
                #         dim=0
                #     )
                #     pw_loss = h_pw_loss.mean() + v_pw_loss.mean()
                # else:
                #     pw_loss = criterion(h_output, h_target, resize_scores=False) + \
                #             criterion(v_output, v_target, resize_scores=False)

                # loss = u_loss.mean() + config.LOSS.PAIRWISE_LOSS_WEIGHT * pw_loss
                raise NotImplementedError("Explicit pairwise loss is not supported for now")
            else:
                loss = u_loss.mean()
        else:
            loss = criterion(output, target)

        # compute gradient and do update step
        optimizer.zero_grad()
        if config.DEBUG.DEBUG:
            loss.backward(retain_graph=True)
            unary_w_grad = model.module.final_layer.weight.grad.clone().detach()
            unary_b_grad = model.module.final_layer.bias.grad.clone().detach()
            if config.MODEL.LEARN_PAIRWISE_TERMS:
                pw1_w_grad = model.module.pairwise_layer[0].weight.grad.clone().detach()
                pw1_b_grad = model.module.pairwise_layer[0].bias.grad.clone().detach()
                pw2_w_grad = model.module.pairwise_layer[-1].weight.grad.clone().detach()
                pw2_b_grad = model.module.pairwise_layer[-1].bias.grad.clone().detach()

        else:
            loss.backward()

        if config.MODEL.NAME == 'seg_fcn':
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

        optimizer.step()

        if config.DEBUG.DEBUG:
            unary_w_exp_avg = optimizer.state[model.module.final_layer.weight]['exp_avg']
            unary_w_exp_avg_sq = optimizer.state[model.module.final_layer.weight]['exp_avg_sq']

            unary_b_exp_avg = optimizer.state[model.module.final_layer.bias]['exp_avg']
            unary_b_exp_avg_sq = optimizer.state[model.module.final_layer.bias]['exp_avg_sq']

            if config.MODEL.LEARN_PAIRWISE_TERMS:
                pw1_w_exp_avg = optimizer.state[model.module.pairwise_layer[0].weight]['exp_avg']
                pw1_w_exp_avg_sq = optimizer.state[model.module.pairwise_layer[0].weight]['exp_avg_sq']

                pw1_b_exp_avg = optimizer.state[model.module.pairwise_layer[0].bias]['exp_avg']
                pw1_b_exp_avg_sq = optimizer.state[model.module.pairwise_layer[0].bias]['exp_avg_sq']

                pw2_w_exp_avg = optimizer.state[model.module.pairwise_layer[-1].weight]['exp_avg']
                pw2_w_exp_avg_sq = optimizer.state[model.module.pairwise_layer[-1].weight]['exp_avg_sq']

                pw2_b_exp_avg = optimizer.state[model.module.pairwise_layer[-1].bias]['exp_avg']
                pw2_b_exp_avg_sq = optimizer.state[model.module.pairwise_layer[-1].bias]['exp_avg_sq']

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
        #                                  target.detach().cpu().numpy())
        # acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']

            writer.add_scalar('train_loss', losses.val, global_steps)

            if config.MODEL.LEARN_PAIRWISE_TERMS:
                if config.MODEL.LEARN_GAMMA:
                    temperature = model.module.fixed_point_iteration.gamma.detach().clone().item()
                    temperature_grad = model.module.fixed_point_iteration.gamma.grad.detach().clone().item()
                    writer.add_scalar('smoothed_max_temperature_grad', temperature_grad, global_steps)
                else:
                    temperature = gamma

                writer.add_scalar('smoothed_max_temperature', temperature, global_steps)
                if isinstance(duality_gap, torch.Tensor):
                    writer.add_scalar('duality_gap', duality_gap.min().item(), global_steps)
                elif isinstance(duality_gap, float):
                    writer.add_scalar('duality_gap', duality_gap, global_steps)
                else:
                    raise ValueError("duality_gap is of wrong type")

                # for i, alpha in enumerate(model.module.alphas):
                #     writer.add_scalar('step_size{}'.format(i), alpha.item(), global_steps)
                # writer.add_scalar('step_size', model.module.alpha.item(), global_steps)

            if config.MODEL.LEARN_OUTPUT_TEMPERATURE:
                temperature = torch.reciprocal(model.module.gamma.detach().clone()).item()
                writer.add_scalar('output_temperature', temperature, global_steps)

            if config.DEBUG.DEBUG:
                unary_w_2norm = torch.norm(unary_w_grad.view(unary_w_grad.size(0), -1))
                unary_b_2norm = torch.norm(unary_b_grad)
                if config.MODEL.LEARN_PAIRWISE_TERMS:
                    pw1_w_2norm = torch.norm(pw1_w_grad.view(pw1_w_grad.size(0), -1))
                    pw1_b_2norm = torch.norm(pw1_b_grad)
                    pw2_w_2norm = torch.norm(pw2_w_grad.view(pw2_w_grad.size(0), -1))
                    pw2_b_2norm = torch.norm(pw2_b_grad)

                unary_w_1norm = torch.norm(unary_w_grad.view(unary_w_grad.size(0), -1), p=1)
                unary_b_1norm = torch.norm(unary_b_grad, p=1)
                if config.MODEL.LEARN_PAIRWISE_TERMS:
                    pw1_w_1norm = torch.norm(pw1_w_grad.view(pw1_w_grad.size(0), -1), p=1)
                    pw1_b_1norm = torch.norm(pw1_b_grad, p=1)
                    pw2_w_1norm = torch.norm(pw2_w_grad.view(pw2_w_grad.size(0), -1), p=1)
                    pw2_b_1norm = torch.norm(pw2_b_grad, p=1)

                writer.add_scalar('unary_weight_grad_norm2', unary_w_2norm.item(), global_steps)
                writer.add_scalar('unary_bias_grad_norm2', unary_b_2norm.item(), global_steps)
                if config.MODEL.LEARN_PAIRWISE_TERMS:
                    writer.add_scalar('pw1_weight_grad_norm2', pw1_w_2norm.item(), global_steps)
                    writer.add_scalar('pw1_bias_grad_norm2', pw1_b_2norm.item(), global_steps)
                    writer.add_scalar('pw2_weight_grad_norm2', pw2_w_2norm.item(), global_steps)
                    writer.add_scalar('pw2_bias_grad_norm2', pw2_b_2norm.item(), global_steps)

                writer.add_scalar('unary_weight_grad_norm1', unary_w_1norm.item(), global_steps)
                writer.add_scalar('unary_bias_grad_norm1', unary_b_1norm.item(), global_steps)
                if config.MODEL.LEARN_PAIRWISE_TERMS:
                    writer.add_scalar('pw1_weight_grad_norm1', pw1_w_1norm.item(), global_steps)
                    writer.add_scalar('pw1_bias_grad_norm1', pw1_b_1norm.item(), global_steps)
                    writer.add_scalar('pw2_weight_grad_norm1', pw2_w_1norm.item(), global_steps)
                    writer.add_scalar('pw2_bias_grad_norm1', pw2_b_1norm.item(), global_steps)

                unary_w_exp_avg_2norm = torch.norm(unary_w_exp_avg.view(unary_w_exp_avg.size(0), -1))
                unary_w_exp_avg_sq_2norm = torch.norm(unary_w_exp_avg_sq.view(unary_w_exp_avg_sq.size(0), -1))

                unary_b_exp_avg_2norm = torch.norm(unary_b_exp_avg)
                unary_b_exp_avg_sq_2norm = torch.norm(unary_b_exp_avg_sq)

                if config.MODEL.LEARN_PAIRWISE_TERMS:
                    pw1_w_exp_avg_2norm = torch.norm(pw1_w_exp_avg.view(pw1_w_exp_avg.size(0), -1))
                    pw1_w_exp_avg_sq_2norm = torch.norm(pw1_w_exp_avg_sq.view(pw1_w_exp_avg_sq.size(0), -1))

                    pw1_b_exp_avg_2norm = torch.norm(pw1_b_exp_avg)
                    pw1_b_exp_avg_sq_2norm = torch.norm(pw1_b_exp_avg_sq)

                    pw2_w_exp_avg_2norm = torch.norm(pw2_w_exp_avg.view(pw2_w_exp_avg.size(0), -1))
                    pw2_w_exp_avg_sq_2norm = torch.norm(pw2_w_exp_avg_sq.view(pw2_w_exp_avg_sq.size(0), -1))

                    pw2_b_exp_avg_2norm = torch.norm(pw2_b_exp_avg)
                    pw2_b_exp_avg_sq_2norm = torch.norm(pw2_b_exp_avg_sq)

                writer.add_scalar('unary_w_exp_avg_norm2', unary_w_exp_avg_2norm.item(), global_steps)
                writer.add_scalar('unary_w_exp_avg_sq_norm2', unary_w_exp_avg_sq_2norm.item(), global_steps)

                writer.add_scalar('unary_b_exp_avg_norm2', unary_b_exp_avg_2norm.item(), global_steps)
                writer.add_scalar('unary_b_exp_avg_sq_norm2', unary_b_exp_avg_sq_2norm.item(), global_steps)

                if config.MODEL.LEARN_PAIRWISE_TERMS:
                    writer.add_scalar('pw1_w_exp_avg_norm2', pw1_w_exp_avg_2norm.item(), global_steps)
                    writer.add_scalar('pw1_w_exp_avg_sq_norm2', pw1_w_exp_avg_sq_2norm.item(), global_steps)

                    writer.add_scalar('pw1_b_exp_avg_norm2', pw1_b_exp_avg_2norm.item(), global_steps)
                    writer.add_scalar('pw1_b_exp_avg_sq_norm2', pw1_b_exp_avg_sq_2norm.item(), global_steps)

                    writer.add_scalar('pw2_w_exp_avg_norm2', pw2_w_exp_avg_2norm.item(), global_steps)
                    writer.add_scalar('pw2_w_exp_avg_sq_norm2', pw2_w_exp_avg_sq_2norm.item(), global_steps)

                    writer.add_scalar('pw2_b_exp_avg_norm2', pw2_b_exp_avg_2norm.item(), global_steps)
                    writer.add_scalar('pw2_b_exp_avg_sq_norm2', pw2_b_exp_avg_sq_2norm.item(), global_steps)

            # input_image = input.detach().cpu().numpy()[0]
            # min_val = input_image.min()
            # max_val = input_image.max()
            # input_image = (input_image - min_val) / (max_val - min_val)
            # heatmap_target = target.detach().cpu().numpy()[0]
            # heatmap_pred = output.detach().cpu().numpy()[0]
            # heatmap_pred[heatmap_pred < 0.0] = 0
            # heatmap_pred[heatmap_pred > 1.0] = 1.0

            # writer.add_image('input_recording', input_image, global_steps,
            #     dataformats='CHW')
            # writer.add_image('heatmap_target', heatmap_target, global_steps,
            #     dataformats='CHW')
            # writer.add_image('heatmap_pred', heatmap_pred, global_steps,
            #     dataformats='CHW')

            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            # save_debug_images(config, input, meta, target, pred*4, output,
            #                   prefix)


def validate(config, val_loader, val_dataset, model, criterion,
             output_dir, tb_log_dir, writer_dict=None, gamma=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    # acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    all_preds = []
    all_gts = []

    with torch.no_grad():
        end = time.time()
        for i, (input, target, h_target, v_target, _) in enumerate(val_loader):
            if len(input.shape) > 4:
                input = input.view(input.shape[0] * input.shape[1], input.shape[2], input.shape[3], input.shape[4])
                target = target.view(target.shape[0] * target.shape[1], target.shape[2], target.shape[3])

            # compute output
            if config.MODEL.LEARN_PAIRWISE_TERMS:
                # outputs, h_output, v_output, disagreement = model(input, config.TRAIN.NE_GAMMA_L if gamma is None else gamma)
                # h_output = h_output[0]
                # v_output = v_output[0]
                outputs, disagreement, _ = model(input, config.TRAIN.NE_GAMMA_L if gamma is None else gamma)
                output = outputs[-1]
            else:
                output = model(input)

            target = target.cuda(non_blocking=True)
            # if config.MODEL.LEARN_PAIRWISE_TERMS:
            #     loss = torch.stack(
            #         [criterion(output, target) for output in outputs],
            #         dim = 0
            #     )
            #     loss = loss.mean()
            # else:
            loss = criterion(output, target)

            output = torch.nn.functional.interpolate(
                output,
                size=(target.size(1), target.size(2)),
                mode="bilinear",
                align_corners=False)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            # if config.MODEL.LEARN_PAIRWISE_TERMS:
            #     preds = get_final_preds(outputs[-2].detach().cpu().numpy(), outputs[-1].detach().cpu().numpy())
            # else:
            preds = get_final_preds(output.detach().cpu().numpy())

            all_preds.extend(preds)
            all_gts.extend(target.detach().cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses)
                logger.info(msg)

                if writer_dict:
                    writer = writer_dict['writer']
                    global_steps = writer_dict['vis_global_steps']

                    idx = np.random.randint(0, num_images)
                    # idx = 0

                    input_image = input.detach().cpu().numpy()[idx]
                    input_image = input_image * val_dataset.std.squeeze(0) + val_dataset.mean.squeeze(0)
                    input_image[input_image > 1.0] = 1.0
                    input_image[input_image < 0.0] = 0.0

                    target_image = target.detach().cpu().numpy()[idx].astype(np.int64)
                    target_image = val_dataset.decode_segmap(target_image)

                    # if config.MODEL.LEARN_PAIRWISE_TERMS:
                    #     output = (torch.nn.functional.softmax(outputs[-2], dim=1) +
                    #         torch.nn.functional.softmax(outputs[-1], dim=1)) / 2.0
                    #     labels = torch.argmax(output, dim=1, keepdim=False)
                    # else:
                    output = torch.nn.functional.softmax(output, dim=1)
                    labels = torch.argmax(output, dim=1, keepdim=False)

                    labels = labels.detach().cpu().numpy()[idx]
                    output_vis = vis_segments(labels, config.MODEL.EXTRA.NUM_CLASSES)


                    writer.add_image('input_image', input_image, global_steps,
                        dataformats='CHW')
                    writer.add_image('result_vis', output_vis, global_steps,
                        dataformats='HWC')
                    writer.add_image('gt_mask', target_image, global_steps,
                        dataformats='HWC')

                    if config.MODEL.LEARN_PAIRWISE_TERMS:
                        disagreement = disagreement.detach().cpu().numpy()[idx].astype(np.uint8)
                        disagreement = np.expand_dims(disagreement, axis=-1)
                        disagreement = np.repeat(disagreement, 3, axis=-1)
                        disagreement = np.transpose(disagreement, (2, 0, 1)) * 255

                        writer.add_image('disagreement', disagreement, global_steps,
                            dataformats='CHW')

                    # if config.MODEL.LEARN_PAIRWISE_TERMS:
                    #     h_output = h_output.view(h_output.size(0), -1, h_output.size(3), h_output.size(4))
                    #     h_output = torch.nn.functional.softmax(h_output, dim=1)
                    #     h_labels = torch.argmax(h_output, dim=1, keepdim=False)
                    #     h_labels = h_labels.detach().cpu().numpy()[idx].astype(np.int64)
                    #     h_labels = h_labels[:-1, :]

                    #     v_output = v_output.view(v_output.size(0), -1, v_output.size(3), v_output.size(4))
                    #     v_output = torch.nn.functional.softmax(v_output, dim=1)
                    #     v_labels = torch.argmax(v_output, dim=1, keepdim=False)
                    #     v_labels = v_labels.detach().cpu().numpy()[idx].astype(np.int64)
                    #     v_labels = v_labels[:, :-1]
                    #     # h_labels = h_target.detach().cpu().numpy()[idx].astype(np.int64)
                    #     # h_labels = h_labels[:-1, :]

                    #     # v_labels = v_target.detach().cpu().numpy()[idx].astype(np.int64)
                    #     # v_labels = v_labels[:, :-1]

                    #     n_classes = config.MODEL.EXTRA.NUM_CLASSES
                    #     off_diag = np.logical_or(
                    #         (h_labels % n_classes != h_labels // n_classes),
                    #         (v_labels % n_classes != v_labels // n_classes)
                    #     )
                    #     off_diag = off_diag.astype(np.uint8) * 255
                    #     writer.add_image('edges', off_diag, global_steps,
                    #         dataformats='HW')

                    writer_dict['vis_global_steps'] = global_steps + 1

                # prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), i)

        # Calculate IoU score for entire validation set
        if config.DATASET.DATASET.find('mnist') >= 0:
            all_preds = np.concatenate(all_preds, axis=0)
            all_gts = np.concatenate(all_gts, axis=0)

        avg_iou_score = calc_IoU(all_preds, all_gts, config.MODEL.EXTRA.NUM_CLASSES)

        perf_indicator = avg_iou_score

        logger.info('Mean IoU score: {:.3f}'.format(avg_iou_score))

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']

            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_iou_score', avg_iou_score, global_steps)

            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


def test(config, test_loader, model, output_dir, tb_log_dir,
         writer_dict=None):
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    all_preds = []

    with torch.no_grad():
        end = time.time()
        for i, input in enumerate(test_loader):
            # compute output
            output = model(input)

            num_images = input.size(0)

            preds = get_final_preds(output.detach().cpu().numpy())
            all_preds.extend(preds)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                          i, len(test_loader), batch_time=batch_time)
                logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['vis_global_steps']

                # idx = np.random.randint(0, num_images)
                idx = 0

                input_image = input.detach().cpu().numpy()[idx]
                min_val = input_image.min()
                max_val = input_image.max()
                input_image = (input_image - min_val) / (max_val - min_val)
                heatmap_pred = output.detach().cpu().numpy()[idx]
                heatmap_pred[heatmap_pred < 0.0] = 0
                heatmap_pred[heatmap_pred > 1.0] = 1.0

                input_image = (input_image * 255).astype(np.uint8)
                input_image = np.transpose(input_image, (1, 2, 0))
                pred = preds[idx]
                tp = np.ones(pred.shape[0], dtype=bool)

                writer.add_image('final_preds', final_preds, global_steps,
                    dataformats='HWC')
                writer.add_image('input_recording', input_image, global_steps,
                    dataformats='HWC')
                writer.add_image('heatmap_pred', heatmap_pred, global_steps,
                    dataformats='CHW')

                writer_dict['vis_global_steps'] = global_steps + 1

                # prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), i)


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
