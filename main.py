# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import warnings
import logging

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.plateau_lr import PlateauLRScheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

#from datasets import build_dataset
from sifar_pytorch.engine import train_one_epoch, evaluate
from sifar_pytorch.samplers import RASampler
from sifar_pytorch import models
from sifar_pytorch import my_models
import torch.nn as nn
#import simclr
from sifar_pytorch import utils
from sifar_pytorch.losses import DeepMutualLoss, ONELoss, MulMixturelLoss, SelfDistillationLoss

from sifar_pytorch.video_dataset import VideoDataSet, VideoDataSetLMDB, VideoDataSetOnline
from sifar_pytorch.video_dataset_aug import get_augmentor, build_dataflow
from sifar_pytorch.video_dataset_config import get_dataset_config, DATASET_CONFIG

from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
warnings.filterwarnings("ignore", category=UserWarning)
#torch.multiprocessing.set_start_method('spawn', force=True)
_logger = logging.getLogger(__name__)
from ssl_sifar_utils import *

import signal
import traceback
    
def lineno(signalnum, frame):
    print(" ===========================================================================================================")
    traceback.print_stack(frame)
    print(" ===========================================================================================================")

    
signal.signal(signal.SIGTSTP, lineno) 

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=150, type=int)

    # Dataset parameters
    parser.add_argument('--data_dir', type=str, metavar='DIR', help='path to dataset')
    parser.add_argument('--dataset', default='st2stv2',
                        choices=list(DATASET_CONFIG.keys()), help='path to dataset file list')
    parser.add_argument('--duration', default=8, type=int, help='number of frames')
    parser.add_argument('--frames_per_group', default=1, type=int,
                        help='[uniform sampling] number of frames per group; '
                             '[dense sampling]: sampling frequency')
    parser.add_argument('--threed_data', action='store_true',
                        help='load data in the layout for 3D conv')
    parser.add_argument('--input_size', default=224, type=int, metavar='N', help='input image size')
    parser.add_argument('--disable_scaleup', action='store_true',
                        help='do not scale up and then crop a small region, directly crop the input_size')
    parser.add_argument('--random_sampling', action='store_true',
                        help='perform determinstic sampling for data loader')
    parser.add_argument('--dense_sampling', action='store_true',
                        help='perform dense sampling for data loader')
    parser.add_argument('--augmentor_ver', default='v1', type=str, choices=['v1', 'v2'],
                        help='[v1] TSN data argmentation, [v2] resize the shorter side to `scale_range`')
    parser.add_argument('--scale_range', default=[256, 320], type=int, nargs="+",
                        metavar='scale_range', help='scale range for augmentor v2')
    parser.add_argument('--modality', default='rgb', type=str, help='rgb or flow',
                        choices=['rgb', 'flow'])
    parser.add_argument('--use_lmdb', action='store_true', help='use lmdb instead of jpeg.')
    parser.add_argument('--use_pyav', action='store_true', help='use video directly.')

    # temporal module
    parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
    parser.add_argument('--temporal_module_name', default=None, type=str, metavar='TEM', choices=['ResNet3d', 'TAM', 'TTAM', 'TSM', 'TTSM', 'MSA'],
                        help='temporal module applied. [TAM]')
    parser.add_argument('--temporal_attention_only', action='store_true', default=False,
                        help='use attention only in temporal module]')
    parser.add_argument('--no_token_mask', action='store_true', default=False, help='do not apply token mask')
    parser.add_argument('--temporal_heads_scale', default=1.0, type=float, help='scale of the number of spatial heads')
    parser.add_argument('--temporal_mlp_scale', default=1.0, type=float, help='scale of spatial mlp')
    parser.add_argument('--rel_pos', action='store_true', default=False,
                        help='use relative positioning in temporal module]')
    parser.add_argument('--temporal_pooling', type=str, default=None, choices=['avg', 'max', 'conv', 'depthconv'],
                        help='perform temporal pooling]')
    parser.add_argument('--bottleneck', default=None, choices=['regular', 'dw'],
                        help='use depth-wise bottleneck in temporal attention')

    parser.add_argument('--window_size', default=7, type=int, help='number of frames')
    parser.add_argument('--super_img_rows', default=1, type=int, help='number of frames per row')

    parser.add_argument('--hpe_to_token', default=False, action='store_true',
                        help='add hub position embedding to image tokens')
    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
#    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=False)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.0, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Dataset parameters
#    parser.add_argument('--data-path', default=os.path.join(os.path.expanduser("~"), 'datasets/image_cls/imagenet1k/'), type=str,
#                        help='dataset path')
#    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR10', 'CIFAR100', 'IMNET', 'INAT', 'INAT19', 'IMNET21K', 'Flowers102', 'StanfordCars', 'iNaturalist2019', 'Caltech101'],
#                        type=str, help='Image Net dataset path')
#    parser.add_argument('--inat-category', default='name',
#                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
#                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--no-resume-loss-scaler', action='store_false', dest='resume_loss_scaler')
    parser.add_argument('--no-amp', action='store_false', dest='amp', help='disable amp')
    parser.add_argument('--use_checkpoint', default=False, action='store_true', help='use checkpoint to save memory')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=20, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # for testing and validation
    parser.add_argument('--num_crops', default=1, type=int, choices=[1, 3, 5, 10])
    parser.add_argument('--num_clips', default=1, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local-rank", type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')


    parser.add_argument('--auto-resume', action='store_true', help='auto resume')
    # exp
    parser.add_argument('--simclr_w', type=float, default=0., help='weights for simclr loss')
    parser.add_argument('--contrastive_nomixup', action='store_true', help='do not involve mixup in contrastive learning')
    parser.add_argument('--temperature', type=float, default=0.07, help='temperature of NCE')
    parser.add_argument('--branch_div_w', type=float, default=0., help='add branch divergence in the loss')
    parser.add_argument('--simsiam_w', type=float, default=0., help='weights for simsiam loss')
    parser.add_argument('--moco_w', type=float, default=0., help='weights for moco loss')
    parser.add_argument('--byol_w', type=float, default=0., help='weights for byol loss')
    parser.add_argument('--finetune', action='store_true', help='finetune model')
    parser.add_argument('--initial_checkpoint', type=str, default='', help='path to the pretrained model')
    parser.add_argument('--dml_w', type=float, default=0., help='enable deep mutual learning')
    parser.add_argument('--one_w', type=float, default=0., help='enable ONE')
    parser.add_argument('--kd_temp', type=float, default=1.0, help='temperature for kd loss')
    parser.add_argument('--mulmix_b', type=float, default=0., help='mulmix beta')
    parser.add_argument('--hard_contrastive', action='store_true', help='use HEXA')
    parser.add_argument('--selfdis_w', type=float, default=0., help='enable self distillation')


   # New parameter added for spliting of training list

    parser.add_argument('--percentage', type=float, default=0.95, help='Percent of Unlabeled training list')
    parser.add_argument('--strategy', type=str, default='classwise', help='spliting strategy, classwise or overall')

    parser.add_argument('--mu', type=int, default=1, help='batch size factor for unlabeled')

    parser.add_argument('--gamma', type=float, default=1.0, help='instace constractive loss factor')
    parser.add_argument('--beta', type=float, default=1.0, help='group constractive loss factor')

    #used in simclr 
    parser.add_argument('--sup_thresh', type=int, default=25, help='Supervise threshold')

    parser.add_argument('--list_root', type=str, default="/home/prithwish/aftab/workspace/ssl-sifar-dgx/dataset_list", help='Path of the train val list')
    parser.add_argument('--lr_factor', type=float, default=0.1, help='factor multiply with the original lr after sup_thres')

    parser.add_argument('--auto_resume', action='store_true', default=False,
                    help='automatically resume from the output dir checkpoint')

    parser.add_argument('--pretrained-path', type=str, default='', help='path to the pretrained ckpt imagenet')
    parser.add_argument('--no_flip', action='store_true', default=False, 
                    help='Disable RandomHorizontalFlip in augmentaion')
    parser.add_argument('--fast-backprop', action='store_true', default=False, 
                    help='use fast back prop')
    parser.add_argument('--remark', type=str, default="--")
    parser.add_argument('--lr-cycle', type=float, default=1.0, help='LR sched cycle')
    parser.add_argument('--lr-min', type=float, default=0.0, help='min LR in sched')
    parser.add_argument('--model-type', type=str, default='swin', help='Swin')
    parser.add_argument('--drop-last', action='store_true', default=False, help='Drop last batch')
    parser.add_argument('--frame-order', type=str, default='normal', help='Frame order in super image')
    parser.add_argument('--no-group-loss', action='store_true', default=False, help='Drop last batch')
    parser.add_argument('--decay-t', type=int, default=50, help='decay epoch in step lr')
    parser.add_argument('--use-pl-loss', action='store_true', default=False, help='Drop last batch')
    parser.add_argument('--threshold', type=float, default=0.8, help='pl loss threshold')
    parser.add_argument('--test-batch-size', type=int, default=15, help='test batch size')
    parser.add_argument('--classwise-eval', action='store_true', default=False, help='Do a classwise evaluation')

    return parser

def main(args):
    args.distributed = False
    # utils.init_distributed_mode(args)
    args_dict = vars(args)
    torch.save(args_dict, os.path.join(args.output_dir, 'args'))
    argstr = '\n'.join([f"{k:<30}:\t{v}" for k, v in args_dict.items()])
    _logger.info(argstr)
    

    if not hasattr(args, 'hard_contrastive'):
        args.hard_contrastive = False
    if not hasattr(args, 'selfdis_w'):
        args.selfdis_w = 0.0

    #is_imnet21k = args.data_set == 'IMNET21K'

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    num_classes, train_list_name, val_list_name, filename_seperator, image_tmpl, filter_video, train_label_list_name, train_unlabel_list_name = get_dataset_config(
        args.dataset, args.use_lmdb)

    args.num_classes = num_classes
    if args.modality == 'rgb':
        args.input_channels = 3
    elif args.modality == 'flow':
        args.input_channels = 2 * 5

#    mean = IMAGENET_DEFAULT_MEAN
#    std = IMAGENET_DEFAULT_STD

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
    
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        img_size=args.input_size,
        pretrained=args.pretrained,
        duration=args.duration,
        hpe_to_token = args.hpe_to_token,
        rel_pos = args.rel_pos,
        window_size=args.window_size,
        super_img_rows = args.super_img_rows,
        token_mask=not args.no_token_mask,
        online_learning = args.one_w >0.0 or args.dml_w >0.0,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        use_checkpoint=args.use_checkpoint,
        ## added by aftab, for loading pretrained imagenet from ckpt
        pretrained_model=args.pretrained_path,
        fast_backprop=args.fast_backprop,
        enable_amp=args.amp,
        model_type=args.model_type
    )

    # TODO: finetuning

    # print("Flops: ", model.flops())
    model= nn.DataParallel(model)
    model.to(device)
    model_ema = None
    print(model)
    # import ipdb; ipdb.set_trace()
    # exit(0)
    
    ################################################################
    ## To freeze some blocks of the model.
    
    # for name, param in model.named_parameters():
    #     if "layers.2.blocks.0." in name or "layers.0." in name or "layers.1." in name or "layers.2.blocks.1." in name or "layers.2.blocks.2." in name or "layers.2.blocks.3." in name or "layers.2.blocks.4." in name or "layers.2.blocks.5." in name or "layers.2.blocks.7." in name or "layers.2.blocks.8." in name or "layers.2.blocks.9." in name:
    #         param.requires_grad = False
    #
    # train_param=0
    # tot_param=0
    #
    # for name, param in model.named_parameters():
    #     if param.requires_grad == True:
    #         if len(param.size()) < 2:
    #             #print(name," ",param.size()[0])
    #             train_param = tpython3 main.py --data_dir '/nobackup/users/rpanda/datasets/Kinetics400_sifar/compress/'  --list_root '/nobackup/users/rpanda/owais/ssl-sifar/dataset_list/k400_1per_SVformer'  --use_pyav --dataset 'kinetics400' --opt adamw --lr 8e-6 --epochs 50 --sched cosine --duration 8 --batch-size 3  --super_img_rows 3 --disable_scaleup --mixup 0.8 --cutmix 1.0 --drop-path 0.1  --model sifar_large_patch4_window12_192_3x3 --output_dir '/nobackup/users/rpanda/owais/SITAR/output/sitar_large/kinetics_1per/lr_8e-6_with_smooth_ssl' --hpe_to_token  --sup_thresh 0 --num_workers 16 --mu 4 --input_size 192 --temperature 0.5  --gamma 0.6 --beta 1 --model-type 'swin' --test-batch-size 30 --smoothing 0.3 --resume  '/nobackup/users/rpanda/owais/SITAR/output/sitar_large/kinetics_1per/lr_8e-6_with_smooth_ssl/checkpoint.pth'rain_param+param.size()[0]
    #             a=1
    #         elif len(param.size()) < 3:
    #             #print(name," ",param.size()[0]*param.size()[1])
    #             train_param = train_param+(param.size()[0]*param.size()[1])
    #             a=1
    #         else:
    #             #print(name," ",param.size()[0]*param.size()[1]*param.size()[2])
    #             train_param = train_param+(param.size()[0]*param.size()[1]*param.size()[2])
    #             a=1
    #
    # print("train_param:",train_param)
    
    
    # exit(0)
    
    
    ##################################################################

    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    # model_without_ddp = model
    # if args.distributed:
    #     print("Using distributed training...")
    #     #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    #     # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model = torch.nn.DataParallel(model, device_ids=[args.gpu]).cuda()
    #     #model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    #linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    #args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScalerWithGradNormCount()
    #print(f"Scaled learning rate (batch size: {args.batch_size * utils.get_world_size()}): {linear_scaled_lr}")
    
    
    # lr_sched_cosine = CosineLRScheduler(optimizer, t_initial=args.epochs, warmup_t= 0.1 * args.epochs, warmup_lr_init=1e-6)
    #lr_sched_cosine = CosineLRScheduler(optimizer, t_initial=args.epochs/3, lr_min=1e-10, cycle_limit=3)
    
    lr_sched_cosine = CosineLRScheduler(optimizer, 
                                        t_initial=args.epochs/args.lr_cycle, 
                                        lr_min=args.lr_min, 
                                        cycle_limit=args.lr_cycle,
                                        warmup_t=args.warmup_epochs,
                                        warmup_lr_init=args.warmup_lr
                                        )

    # lr_sched_cosine = StepLRScheduler(optimizer, decay_t = args.decay_t, decay_rate=args.decay_rate)

    # lr_sched_cosine = PlateauLRScheduler(optimizer, decay_rate=0.1, threshold=0.1)
    # lr_sched_cosine = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15,20], gamma=0.1)  ##added by owais
    
    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss() 

    if args.dml_w > 0.:
        criterion = DeepMutualLoss(criterion, args.dml_w, args.kd_temp)
    elif args.one_w > 0.:
        criterion = ONELoss(criterion, args.one_w, args.kd_temp)
    elif args.mulmix_b > 0.:
        criterion = MulMixturelLoss(criterion, args.mulmix_b)
    elif args.selfdis_w > 0.:
        criterion = SelfDistillationLoss(criterion, args.selfdis_w, args.kd_temp)

    # all are None as of now
    simclr_criterion = simclr.NTXent(temperature=args.temperature) if args.simclr_w > 0. else None
    branch_div_criterion = torch.nn.CosineSimilarity() if args.branch_div_w > 0. else None
    simsiam_criterion = simclr.SimSiamLoss() if args.simsiam_w > 0. else None
    moco_criterion = torch.nn.CrossEntropyLoss() if args.moco_w > 0. else None
    byol_criterion = simclr.BYOLLoss() if args.byol_w > 0. else None

    max_accuracy = 0.0
    output_dir = Path(args.output_dir)

    if args.initial_checkpoint:
        print("Loading pretrained model")
        checkpoint = torch.load(args.initial_checkpoint, map_location='cpu')
        utils.load_checkpoint(model, checkpoint['model'])

    if args.auto_resume:
        if args.resume == '':
            args.resume = str(output_dir / "checkpoint.pth")
            if not os.path.exists(args.resume):
                args.resume = ''

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        utils.load_checkpoint(model, checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_sched_cosine.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint and args.resume_loss_scaler:
                print("Resume with previous loss scaler state")
                loss_scaler.load_state_dict(checkpoint['scaler'])
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            max_accuracy = checkpoint['max_accuracy']
    # edited by aftab, mean, std will be used that init in get_aug method
    mean = None #(0.5, 0.5, 0.5) 
    std = None #(0.5, 0.5, 0.5)

    # if args.distributed:
    #     mean = (0.5, 0.5, 0.5) if 'mean' not in model.module.default_cfg else model.module.default_cfg['mean']
    #     std = (0.5, 0.5, 0.5) if 'std' not in model.module.default_cfg else model.module.default_cfg['std']
    # else:
    #     mean = (0.5, 0.5, 0.5) if 'mean' not in model.default_cfg else model.default_cfg['mean']
    #     std = (0.5, 0.5, 0.5) if 'std' not in model.default_cfg else model.default_cfg['std']


    # dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    # create data loaders w/ augmentation pipeiine
    if args.use_lmdb:
        video_data_cls = VideoDataSetLMDB
    elif args.use_pyav:
        video_data_cls = VideoDataSetOnline
    else:
        video_data_cls = VideoDataSet

    
    ## Datasets and Dataloaders

    train_label_list = os.path.join(args.list_root, train_label_list_name)
    train_unlabel_list = os.path.join(args.list_root, train_unlabel_list_name)
    

    train_augmentor = get_augmentor(True, args.input_size, mean, std, threed_data=args.threed_data,
                                    version=args.augmentor_ver, scale_range=args.scale_range, dataset=args.dataset, no_flip=args.no_flip)
    dataset_labeled_train = video_data_cls(args.data_dir, train_label_list, args.duration, args.frames_per_group,
                                   num_clips=args.num_clips,
                                   modality=args.modality, image_tmpl=image_tmpl,
                                   dense_sampling=args.dense_sampling,
                                   transform=train_augmentor, is_train=True, test_mode=False,
                                   seperator=filename_seperator, filter_video=filter_video,
                                   frame_order=args.frame_order)

    dataset_unlabeled_train = video_data_cls(args.data_dir, train_unlabel_list, args.duration, args.frames_per_group,
                                    num_clips=args.num_clips,
                                    modality=args.modality, image_tmpl=image_tmpl,
                                    dense_sampling=args.dense_sampling,
                                    transform=train_augmentor, is_train=True, test_mode=False,
                                    seperator=filename_seperator, filter_video=filter_video,
                                    frame_order=args.frame_order)

    num_tasks = utils.get_world_size()
    labeled_trainloader = build_dataflow(dataset_labeled_train, is_train=True, batch_size=args.batch_size,
                                       workers=args.num_workers, is_distributed=args.distributed, drop_last=args.drop_last)

    unlabeled_trainloader = build_dataflow(dataset_unlabeled_train, is_train=True, batch_size=(args.batch_size * args.mu),
                                       workers=args.num_workers, is_distributed=args.distributed, drop_last=args.drop_last)

    val_list = os.path.join(args.list_root, val_list_name)
    val_augmentor = get_augmentor(False, args.input_size, mean, std, args.disable_scaleup,
                                  threed_data=args.threed_data, version=args.augmentor_ver,
                                  scale_range=args.scale_range, num_clips=args.num_clips, num_crops=args.num_crops, dataset=args.dataset, no_flip=args.no_flip)
    dataset_val = video_data_cls(args.data_dir, val_list, args.duration, args.frames_per_group,
                                 num_clips=args.num_clips,
                                 modality=args.modality, image_tmpl=image_tmpl,
                                 dense_sampling=args.dense_sampling,
                                 transform=val_augmentor, is_train=False, test_mode=False,
                                 seperator=filename_seperator, filter_video=filter_video, frame_order=args.frame_order,
                                 )

    data_loader_val = build_dataflow(dataset_val, is_train=False, batch_size=args.test_batch_size,
                                     workers=args.num_workers, is_distributed=args.distributed, drop_last=args.drop_last)


    #saving the sample superimage from data loader
    # sample_si_root = "/home/mt0/22CS60R54/ssl-sifar/superimages/"
    
    # save_super_image_from_dataloader(data_loader_labeled_train, sample_si_root, "labeled.jpg", True, args.input_size, args.super_img_rows)
    # save_super_image_from_dataloader(data_loader_unlabeled_train, sample_si_root, "unlabeled.jpg", False, args.input_size, args.super_img_rows)

    if args.classwise_eval:
        test_stats = evaluate(data_loader_val, model, device, num_tasks, distributed=args.distributed, amp=args.amp, num_crops=args.num_crops, num_clips=args.num_clips, args=args, classwise=True)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return


    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, num_tasks, distributed=args.distributed, amp=args.amp, num_crops=args.num_crops, num_clips=args.num_clips, args=args)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return
    # test_stats = evaluate(data_loader_val, model, device, num_tasks, distributed=args.distributed, amp=args.amp, num_crops=args.num_crops, num_clips=args.num_clips, args=args)
    # print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    print(f"Start training, currnet max acc is {max_accuracy:.2f}")
    start_time = time.time()
    eval_count = 0

   
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            labeled_trainloader.sampler.set_epoch(epoch)
            unlabeled_trainloader.sampler.set_epoch(epoch)
        
        start_time = time.time()
        train_stats = train_one_epoch(
            model, criterion, labeled_trainloader, unlabeled_trainloader,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn, num_tasks, True,
            args = args,
            amp=args.amp,
            simclr_criterion=simclr_criterion, simclr_w=args.simclr_w,
            branch_div_criterion=branch_div_criterion, branch_div_w=args.branch_div_w,
            simsiam_criterion=simsiam_criterion, simsiam_w=args.simsiam_w,
            moco_criterion=moco_criterion, moco_w=args.moco_w,
            byol_criterion=byol_criterion, byol_w=args.byol_w,
            contrastive_nomixup=args.contrastive_nomixup,
            hard_contrastive=args.hard_contrastive,
            finetune=args.finetune
        )
        end_time = time.time()
        _logger.info(f"Epoch: {epoch}, Time: {(end_time - start_time) / 60}, Fastbackprop: {args.fast_backprop}")
        lr_sched_cosine.step(epoch)
        
        test_stats = evaluate(data_loader_val, model, device, num_tasks, distributed=args.distributed, amp=args.amp, args=args)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

        # added for LR on Platetue
        # lr_sched_cosine.step(test_stats['loss'], epoch)

        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')
        
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if test_stats["acc1"] == max_accuracy:
                checkpoint_paths.append(output_dir / 'model_best.pth')
            for checkpoint_path in checkpoint_paths:
                state_dict = {
                    'model': model.state_dict(), #model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_sched_cosine.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'scaler': loss_scaler.state_dict(),
                    'max_accuracy': max_accuracy
                }
                if args.model_ema:
                    state_dict['model_ema'] = get_state_dict(model_ema)
                utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'test_{k}': v for k, v in test_stats.items()},
                    'epoch': epoch,
                    'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO, 
                            filename=os.path.join(args.output_dir, 'logs.log'),
                            filemode="w",
                            format="%(name)s %(asctime)s %(levelname)s \n%(message)s"
                            )

    main(args)
