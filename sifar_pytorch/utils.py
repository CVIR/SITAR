# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
from collections import defaultdict, deque
import datetime
import tempfile
import math

import torch
import torch.distributed as dist
from fvcore.common.checkpoint import Checkpointer

import numpy as np
from PIL import Image

from torch import nn
from einops import rearrange

from torchvision import transforms

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, lenn=None, header=None):
        i = 0
        if not header:
            header = ''
        
        if not lenn:
            lenn = len(iterable)
            
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(lenn))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == lenn - 1:
                eta_seconds = iter_time.global_avg * (lenn - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, lenn, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, lenn, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / lenn))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def load_checkpoint(model, state_dict, mode=None):

    # reuse Checkpointer in fvcore to support flexible loading
    ckpt = Checkpointer(model, save_to_disk=False)
    # since Checkpointer requires the weight to be put under `model` field, we need to save it to disk
    tmp_path = tempfile.NamedTemporaryFile('w+b')
    torch.save({'model': state_dict}, tmp_path.name)
    ckpt.load(tmp_path.name)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    print("args GPU: ", args.gpu)
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)



def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]


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
        self.avg = self.sum / self.count


def save_super_image(x, output):
    root = "/scratch/20cs91r11/aftab/ssl-sifar-dgx/superimage"
    temp_img = x[0]
    temp_img = temp_img.permute(1, 2, 0)
    numpy_image = temp_img.cpu().numpy()
    numpy_image = numpy_image * 255  # Scale values to [0, 255]
    numpy_image = numpy_image.astype(np.uint8)  # Convert to 8-bit unsigned integers

    
    image = Image.fromarray(numpy_image)

    # Save the image file
    path = os.path.join(root, output)
    image.save(path)
    print(f"Image saved {path}")

def create_super_image(x, isLabeled=True):
    large_nrow = math.ceil(math.sqrt(x.shape[1] / 3))
    small_nrow = large_nrow - 1
    
    frame_size = x.shape[2]
    # print(frame_size)
    
    large_super_image_size = frame_size * large_nrow
    
    if not isLabeled:
        #pick alternate frames for smaller super image
        nframetopick = small_nrow * small_nrow * 6
        small_frames = [x[:,i:i+3, :, : ] for i in range(0, nframetopick, 6)]
        small_frames = torch.cat(small_frames, dim=1)
        
        # padding for small sm
        if small_nrow * small_nrow  != small_frames.shape[1] // 3: 
            npadding = small_nrow * small_nrow - small_frames.shape[1] // 3
            small_frames_padding = torch.zeros((small_frames.shape[0], npadding * 3) + small_frames.size()[2:]).cuda()
            small_frames = torch.cat((small_frames, small_frames_padding), dim=1)

        #creating super image small
        img_size_small = (large_super_image_size // small_nrow, large_super_image_size // small_nrow)
        small_frames = nn.functional.interpolate(small_frames, size=img_size_small, mode='bilinear')
        super_image_small = rearrange(small_frames, 'b (th tw c) h w -> b c (th h) (tw w)', th=small_nrow, c=3)
        
        # super_image_small = nn.functional.interpolate(super_image_small, size=384, mode='bilinear')

    # padding for large sm
    if large_nrow * large_nrow  != x.shape[1] / 3: 
        npadding = large_nrow * large_nrow - x.shape[1] // 3
        x_padding = torch.zeros((x.shape[0], npadding * 3) + x.size()[2:]).cuda()
        x = torch.cat((x, x_padding), dim=1)
    

    #creating super image large
    img_size_large = (frame_size, frame_size)
    x = nn.functional.interpolate(x, size=img_size_large, mode='bilinear')
    super_image_large = rearrange(x, 'b (th tw c) h w -> b c (th h) (tw w)', th=large_nrow, c=3)
    
    # super_image_large = nn.functional.interpolate(super_image_large, size=384, mode='bilinear')
    # print(super_image_large.shape)
    # exit()

    # if not isLabeled:
    #     img_size = (576 / (nrows - 1), 576 / (nrows - 1))
    #     # print("seg4 shape", seg4_image.shape)
    #     stack = None
    #     for i in range(0, 12, 3):
    #         frames = seg4_image[:, i:i + 3, :, :]
    #         framestack = None
    #         for b in range(frames.shape[0]):    
    #             pilimage = transforms.ToPILImage()(frames[b])

    #             resized_image = pilimage.resize(img_size)

    #             # Convert the resized PIL Image back to a tensor
    #             resized_tensor = transforms.ToTensor()(resized_image).unsqueeze(0)

    #             if framestack is None:
    #                 framestack = resized_tensor
    #             else:
    #                 framestack = torch.cat((framestack, resized_tensor), dim = 0)
                
    #         if stack is None:
    #             stack = framestack
    #         else:
    #             stack = torch.cat((stack, framestack), dim = 1)
            
        
    #     seg4_image = nn.functional.interpolate(seg4_image, size=img_size, mode='bilinear')
    #     super_image_2x2 = rearrange(stack, 'b (th tw c) h w -> b c (th h) (tw w)', th=2, c=3)


    
    if not isLabeled:
        return super_image_large, super_image_small

    return super_image_large
