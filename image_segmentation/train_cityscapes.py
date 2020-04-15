from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import numpy as np
import random
# import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

# import _init_paths
from lib.core.config import config
from lib.core.config import update_config
from lib.core.config import update_dir
from lib.core.config import get_model_name
from lib.core.function import train
from lib.core.function import validate
from lib.core.loss import CrossEntropy2D
from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger
from lib.utils.schedulers import PolynomialLR

import lib.dataset as dataset
import lib.models as models
import lib.utils.augmentations as aug

seed = 37
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers


def main():
    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_seg_net')(
        config, is_train=True
    )

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', config.MODEL.NAME + '.py'),
        final_output_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
        'vis_global_steps': 0,
    }

    # dump_input = torch.rand((config.TRAIN.BATCH_SIZE,
    #                          3,
    #                          config.MODEL.IMAGE_SIZE[1],
    #                          config.MODEL.IMAGE_SIZE[0]))
    # writer_dict['writer'].add_graph(model, (dump_input, ), verbose=False)

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    optimizer = get_optimizer(config, model)

    # Data loading code
    if 'xception' in config.MODEL.NAME:
        # Xception uses different mean std for input image
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    train_augs = aug.Compose([aug.RandomScale(0.5, 2.0),
                              aug.RandomHorizontallyFlip(0.5),
                              aug.RandomSizedCrop(config.MODEL.IMAGE_SIZE)])

    test_augs = None

    train_dataset = eval('dataset.'+config.DATASET.DATASET)(
        config,
        config.DATASET.ROOT,
        config.DATASET.TRAIN_SET,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
        augmentations=train_augs
    )
    valid_dataset = eval('dataset.'+config.DATASET.DATASET)(
        config,
        config.DATASET.ROOT,
        config.DATASET.TEST_SET,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
        augmentations=test_augs
    )

    # define loss function (criterion) and optimizer
    criterion = CrossEntropy2D(ignore_index=255, weight=train_dataset.class_weights).cuda()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE*len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True if len(gpus) > 2 else False  # PyTorch's DataParallel model cannot handle 0 image on either of the GPUs
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    if config.TRAIN.LR_SCHEDULER == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR
        )
    elif config.TRAIN.LR_SCHEDULER == 'poly':
        max_iter = config.TRAIN.END_EPOCH * len(train_loader)
        lr_scheduler = PolynomialLR(optimizer, max_iter=max_iter, decay_iter=1)
    elif config.TRAIN.LR_SCHEDULER == 'none':
        lr_scheduler = None
    else:
        raise ValueError('Scheduler {} not supported'.format(config.TRAIN.LR_SCHEDULER))

    best_perf = 0.0
    best_model = False
    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        if config.TRAIN.LR_SCHEDULER == 'multistep':
            lr_scheduler.step()

        # train for one epoch
        train(config, train_loader, model, criterion, optimizer, lr_scheduler, epoch,
              final_output_dir, tb_log_dir, writer_dict)


        if (epoch + 1) % config.TRAIN.EVAL_INTERVAL == 0:
            if not config.MODEL.LEARN_GAMMA:
                if float(lr_scheduler.last_epoch) / (lr_scheduler.max_iter * config.TRAIN.NE_ITER_RATIO) <= 1:
                    gamma = (config.TRAIN.NE_GAMMA_U - config.TRAIN.NE_GAMMA_L) * \
                            (1 - float(lr_scheduler.last_epoch) / (lr_scheduler.max_iter * config.TRAIN.NE_ITER_RATIO) ) ** \
                            config.TRAIN.NE_GAMMA_EXP + config.TRAIN.NE_GAMMA_L
                else:
                    gamma = config.TRAIN.NE_GAMMA_L
            else:
                gamma = None

            # evaluate on validation set
            perf_indicator = validate(config, valid_loader, valid_dataset, model,
                                      criterion, final_output_dir, tb_log_dir,
                                      writer_dict, gamma=gamma)

            if perf_indicator > best_perf:
                best_perf = perf_indicator
                best_model = True
            else:
                best_model = False

            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'epoch': epoch + 1,
                'model': get_model_name(config),
                'state_dict': model.state_dict(),
                'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir)
        else:
            perf_indicator = 0.0

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
