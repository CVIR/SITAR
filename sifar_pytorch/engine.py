# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Train and eval functions used in main.py
"""

import os
import time
import shutil
import random
import datetime
import numpy as np
import math
import sys
from typing import Iterable, Optional
from einops import rearrange
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma, reduce_tensor

from .utils import *
from .utils import save_super_image, create_super_image
from .losses import DeepMutualLoss, ONELoss, SelfDistillationLoss
from collections import defaultdict 
from itertools import cycle
import torch.nn.functional as F

import logging
_logger = logging.getLogger(__name__)



# @profile
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    labeled_trainloader: Iterable, unlabeled_trainloader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    world_size: int = 1, distributed: bool = True, amp=True,
                    simclr_criterion=None, simclr_w=0.,
                    branch_div_criterion=None, branch_div_w=0.,
                    simsiam_criterion=None, simsiam_w=0.,
                    moco_criterion=None, moco_w=0.,
                    byol_criterion=None, byol_w=0.,
                    contrastive_nomixup=False, hard_contrastive=False,
                    finetune=False,
                    args=None
                    ):

    def process_samples_target(samples, targets):
        batch_size = targets.size(0)
        if simclr_criterion is not None or simsiam_criterion is not None or moco_criterion is not None or byol_criterion is not None:
            samples = [samples[0].to(device, non_blocking=True), samples[1].to(device, non_blocking=True)]
            targets = targets.to(device, non_blocking=True)
            ori_samples = [x.clone() for x in samples]  # copy the original samples

            if mixup_fn is not None:
                samples[0], targets_ = mixup_fn(samples[0], targets)
                if contrastive_nomixup:  # remain one copy for ce loss
                    samples[1] = ori_samples[0]
                    samples.append(ori_samples[1])
                elif hard_contrastive:
                    samples[1] = samples[1]
                else:
                    samples[1], _ = mixup_fn(samples[1], targets)
                targets = targets_

        else:
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if mixup_fn is not None:
                # batch size has to be an even number
                if batch_size == 1:
                    return samples, targets
                if batch_size % 2 != 0:
                     samples, targets = samples[:-1], targets[:-1]
                samples, targets = mixup_fn(samples, targets)
                # targets = F.one_hot(targets, num_classes=args.num_classes)
        return samples, targets

    # TODO fix this for finetuning
    if finetune:
        model.train(not finetune)
    else:
        model.train()

    #criterion.train()

    lenn = max(len(labeled_trainloader), len(unlabeled_trainloader))
    if epoch >= args.sup_thresh:
        data_loader = zip(cycle(labeled_trainloader), unlabeled_trainloader)    #ucf, k400
        # data_loader = zip(labeled_trainloader, cycle(unlabeled_trainloader))    #hmdb
    else:
        data_loader = labeled_trainloader
    ## Average meter changed to SmoothedValue
    ## removed by aftab, because it is already in add_meter
    # batch_time = SmoothedValue()
    # data_time = SmoothedValue()
    # total_losses = SmoothedValue()
    # supervised_losses = SmoothedValue()
    # contrastive_losses = SmoothedValue()
    # group_contrastive_losses = SmoothedValue()
    # pl_losses = SmoothedValue()
    # top1 = SmoothedValue()
    # top5 = SmoothedValue()
    # grad_norms = SmoothedValue()
    



    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt="{value:.6f} ({global_avg:.4f})"))
    metric_logger.add_meter('instance_contrastive_loss',  SmoothedValue(window_size=lenn,fmt="{value:.6f} ({global_avg:.4f})"))
    metric_logger.add_meter('group_contrastive_loss',  SmoothedValue(window_size=lenn, fmt="{value:.6f} ({global_avg:.4f})"))
    metric_logger.add_meter('supervised_loss',  SmoothedValue(window_size=lenn, fmt="{value:.6f} ({global_avg:.4f})"))
    metric_logger.add_meter('supervised_loss',  SmoothedValue(window_size=lenn,fmt="{value:.6f} ({global_avg:.4f})"))
    metric_logger.add_meter('grad_norm',  SmoothedValue(window_size=lenn, fmt="{value:.6f} ({global_avg:.4f})"))
    
    if args.use_pl_loss:
        metric_logger.add_meter('pl_loss',  SmoothedValue(window_size=lenn, fmt="{value:.6f} ({global_avg:.4f})"))
    
    header = 'Epoch: [{}]'.format(epoch)
    
    print_freq = 10
    batch_norm = []
    for data in metric_logger.log_every(data_loader, print_freq, lenn if epoch >= args.sup_thresh else len(labeled_trainloader), header):
         #reseting losses
        contrastive_loss = torch.tensor(0.0).cuda()
        pl_loss = torch.tensor(0.0).cuda()
        loss = torch.tensor(0.0).cuda()
        group_contrastive_loss = torch.tensor(0.0).cuda()
        grad_norm = torch.tensor(0.0).cuda()

        if epoch >= args.sup_thresh:
            labeled_data,unlabeled_data = data
            samples_u, targets = unlabeled_data
            samples_u = samples_u.cuda()
            targets = targets.cuda()
            # samples_u, _ = process_samples_target(samples_u, targets)

            # print("sample target ", samples_u.shape, targets.shape)
            super_image_3x3, super_image_2x2 = create_super_image(samples_u, isLabeled=False)
            # print(super_image_3x3.shape)
            # print(super_image_2x2.shape)

            # save_super_image(super_image_3x3, "sm_large_for_ppt.jpg")
            # save_super_image(super_image_2x2, "sm_small_for_ppt.jpg")
            # exit(0)
            # print(super_image_3x3.shape, super_image_2x2.shape)
        else:
            labeled_data = data


        samples, targets = labeled_data
        samples = samples.cuda()
        targets = targets.cuda()

        samples, targets = process_samples_target(samples, targets)
        # print("sample lab", samples.shape, targets.shape)
        super_image_lab = create_super_image(samples, isLabeled=True)
        # save_super_image(super_image_lab, "super_large_for_ppt.jpg")
        # exit(0)
        # with torch.cuda.amp.autocast(enabled=amp): #, dtype=torch.float16):
            
        if epoch >= args.sup_thresh:
            # assert not torch.isnan(super_image_3x3).any()
            # assert not torch.isnan(super_image_2x2).any()
            output_8f = model(super_image_3x3)
            output_8f_detach = output_8f.detach()
            if args.use_pl_loss:
                pseudo_label = torch.softmax(output_8f_detach, dim=-1)
                max_probs, targets_pl = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(args.threshold).float()
                targets_pl = torch.autograd.Variable(targets_pl)
                pl_loss = F.cross_entropy(output_8f, targets_pl,
                                                    reduction='none')
                # print("pl_loss", pl_loss)
                pl_loss = (pl_loss * mask).mean()
                # print("pl_loss", pl_loss)
                

            else:
                output_4f = model(super_image_2x2)
                
                contrastive_loss = simclr_loss(torch.softmax(output_8f_detach,dim=1),torch.softmax(output_4f,dim=1), args)
                grp_unlabeled_8seg = get_group(output_8f_detach)
                grp_unlabeled_4seg = get_group(output_4f)
                group_contrastive_loss = compute_group_contrastive_loss(grp_unlabeled_8seg,grp_unlabeled_4seg, args)
            
        
        outputs = model(super_image_lab)

        if simclr_criterion is not None:
            # outputs 0: ce logits, bs x class, outputs 1: normalized embeddings of two views, bs x 2 x dim
            loss_ce = criterion(outputs[0], targets)
            loss_simclr = simclr_criterion(outputs[1])
            loss = loss_ce * (1.0 - simclr_w) + loss_simclr * simclr_w
        elif simsiam_criterion is not None:
            # outputs 0: ce logits, bs x class, outputs 1: normalized embeddings of two views, 4[bs x dim], [p1, z1, p2, z2]
            loss_ce = criterion(outputs[0], targets)
            loss_simsiam = simsiam_criterion(*outputs[1])
            loss = loss_ce * (1.0 - simsiam_w) + loss_simsiam * simsiam_w
        elif branch_div_criterion is not None:
            # outputs 0: ce logits, bs x class, outputs 1: embeddings of K branches, K[bs x dim]
            loss_ce = criterion(outputs[0], targets)
            loss_div = 0.0
            for i in range(0, len(outputs[1]), 2):
                loss_div += torch.mean(branch_div_criterion(outputs[1][i], outputs[1][i + 1]))
            loss = loss_ce * (1.0 - branch_div_w) + loss_div * branch_div_w
        elif moco_criterion is not None:
            loss_ce = criterion(outputs[0], targets)
            loss_moco = moco_criterion(outputs[1][0], outputs[1][1])
            loss = loss_ce * (1.0 - moco_w) + loss_moco * moco_w
        elif byol_criterion is not None:
            loss_ce = criterion(outputs[0], targets)
            loss_byol = byol_criterion(*outputs[1])
            loss = loss_ce * (1.0 - byol_w) + loss_byol * byol_w
        else:
            if isinstance(criterion, (DeepMutualLoss, ONELoss, SelfDistillationLoss)):
                loss, loss_ce, loss_kd = criterion(outputs, targets)
            else:
                loss = criterion(outputs, targets)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                # raise ValueError("Loss is {}, stopping training".format(loss_value))
        
        if args.no_group_loss:
            total_loss = args.gamma * contrastive_loss + loss
        elif args.use_pl_loss:
            total_loss = pl_loss + loss
        else:
            total_loss = args.gamma * contrastive_loss + loss
            # total_loss = args.gamma * contrastive_loss + args.beta * group_contrastive_loss + loss
        # total_loss = loss + args.beta * group_contrastive_loss

        
        # measure accuracy and record loss
        # if epoch >= args.sup_thresh: 
        #     total_losses.update(total_loss.item(), samples.size(0)+ args.mu*samples.size(0) )
        # else:
        #     total_losses.update(total_loss.item(), samples.size(0))

        # supervised_losses.update(loss.item(), samples.size(0))
        # contrastive_losses.update(contrastive_loss.item(), samples.size(0)+args.mu*samples.size(0))
        # group_contrastive_losses.update(group_contrastive_loss.item(), samples.size(0)+args.mu*samples.size(0))
 
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        if amp:
            # loss_scaler(total_loss, optimizer, clip_grad=max_norm,
            #             parameters=model.parameters(), create_graph=is_second_order)
            
            grad_norm = loss_scaler(total_loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
            
            # grad_norms.update(grad_norm, samples.size(0))
            # metric_logger.update(grad_norm=grad_norms.item())
        else:
            total_loss.backward(create_graph=is_second_order)
            # for p in model.parameters():
            #     if torch.isinf(p.grad).any():
            #         _logger.warning("INF in grad detected")
            grad_norm  = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))

            if max_norm is not None and max_norm != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
        # total_norm = 0.0
        # for p in model.parameters():
        #     param_norm = p.grad.detach().data.norm(2)
        #     total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** 0.5
        # print('after clip grad norm: ', total_norm)
        optimizer.zero_grad()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        if simclr_criterion is not None:
            metric_logger.update(loss_ce=loss_ce.item())
            metric_logger.update(loss_simclr=loss_simclr.item())
        elif simsiam_criterion is not None:
            metric_logger.update(loss_ce=loss_ce.item())
            metric_logger.update(loss_simsiam=loss_simsiam.item())
        elif branch_div_criterion is not None:
            metric_logger.update(loss_ce=loss_ce.item())
            metric_logger.update(loss_div=loss_div.item())
        elif moco_criterion is not None:
            metric_logger.update(loss_ce=loss_ce.item())
            metric_logger.update(loss_moco=loss_moco.item())
        elif byol_criterion is not None:
            metric_logger.update(loss_ce=loss_ce.item())
            metric_logger.update(loss_byol=loss_byol.item())
        elif isinstance(criterion, (DeepMutualLoss, ONELoss)):
            metric_logger.update(loss_ce=loss_ce.item())
            metric_logger.update(loss_kd=loss_kd.item())
        
        metric_logger.update(loss=total_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(instance_contrastive_loss=contrastive_loss.item())
        metric_logger.update(group_contrastive_loss=group_contrastive_loss.item())
        metric_logger.update(supervised_loss=loss.item())
        metric_logger.update(grad_norm=grad_norm.item())
        
        if args.use_pl_loss:
            metric_logger.update(pl_loss=pl_loss.item())
            
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, world_size, args, distributed=True, amp=False, num_crops=1, num_clips=1, classwise=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    outputs = []
    targets = []
    classwise_clf = defaultdict(lambda: [0, 0]) 
    
    lenn = len(data_loader)
    for images, target in metric_logger.log_every(data_loader, 10, lenn , header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # images = torch.rand((60, 24, 192, 192))
        # target = torch.randint(0, 101, (60,))
        
        # compute output
        batch_size = images.shape[0]
        #images = images.view((batch_size * num_crops * num_clips, -1) + images.size()[2:])
        # with torch.cuda.amp.autocast(enabled=amp):
        super_image_val = create_super_image(images, isLabeled=True)
        output = model(super_image_val)

        # output = torch.rand((60, 101))
        #loss = criterion(output, target)
        
        output = output.reshape(batch_size, num_crops * num_clips, -1).mean(dim=1)
        #acc1, acc5 = accuracy(output, target, topk=(1, 5))
        

        if distributed:
            outputs.append(concat_all_gather(output))
            targets.append(concat_all_gather(target))
        else:
            outputs.append(output)
            targets.append(target)

        batch_size = images.shape[0]
        #metric_logger.update(loss=reduced_loss.item())
        #metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        #metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)


    num_data = len(data_loader.dataset)
    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)
    
    # import os
    # # if os.environ.get('TEST', False):
    # import numpy as np
    # print("dumping results...")
    # tmp = outputs[:num_data].cpu().numpy()
    # tt = targets[:num_data].cpu().numpy()
    # np.savez("/nobackup/users/dasabir/workspace/owais/hmdb51_50per_svf.npz", pred=tmp, gt=tt)
    #

    real_acc1, real_acc5 = accuracy(outputs[:num_data], targets[:num_data], topk=(1, 5))

    if classwise:
        index_to_name = {}
        if args.dataset == 'ucf101':
            index_to_name = torch.load('ucf_index_to_name.dict')
        elif args.dataset == 'hmdb51':
            index_to_name = torch.load('hmdb_index_to_name.dict')
        
        _, pred = torch.max(outputs, dim=1)
        

        total_row = len(pred)
        correct = 0
        for i in range(total_row):
            index = targets[i].item()
            if targets[i] == pred[i]:
                classwise_clf[index][0] += 1
                correct += 1
            else:
                classwise_clf[index][1] += 1

        with open('classwise_report_sitar_ucf_101_sora_supv.txt', 'w') as fp:
            print(f'{"Class Index":15}\t{"Class Nmae":25}\t{"Accuracy(in %)":14}', file=fp)
            for i in range(args.num_classes):
                try:
                    acc = (classwise_clf[i][0] * 100.) / (classwise_clf[i][0] + classwise_clf[i][1])
                    print(f'{i:<15}\t{index_to_name[i]:25}\t{acc:14.2f}', file=fp)
                except:
                    print(f'Error for {i}, {classwise_clf[i]}')
            calculated_acc = correct * 100. / total_row
            print(f'Calculated acc: {calculated_acc}\nReal Acc: {real_acc1}', file=fp)
        print("Classwise accuracies saved in classwise_report.txt")




    real_loss = criterion(outputs, targets)
    metric_logger.update(loss=real_loss.item())
    metric_logger.meters['acc1'].update(real_acc1.item())
    metric_logger.meters['acc5'].update(real_acc5.item())
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor.contiguous(), async_op=False)

    #output = torch.cat(tensors_gather, dim=0)
    if tensor.dim() == 1:
        output = rearrange(tensors_gather, 'n b -> (b n)')
    else:
        output = rearrange(tensors_gather, 'n b c -> (b n) c')

    return output





def get_group(output):
    logits = torch.softmax(output, dim=-1)
    _ , target = torch.max(logits, dim=-1)
    groups ={}
    for x,y in zip(target, logits):
        group = groups.get(x.item(),[])
        group.append(y)
        groups[x.item()]= group
    
    return groups



def compute_group_contrastive_loss(grp_dict_un,grp_dict_lab, args):
    
    l_fast_list =[]
    l_slow_list =[]
    for key in grp_dict_un.keys():
        if key in grp_dict_lab:
            l_fast_list.append(torch.stack(grp_dict_un[key]).mean(dim=0))
            l_slow_list.append(torch.stack(grp_dict_lab[key]).mean(dim=0))
    if len(l_fast_list) > 0:
        l_fast = torch.stack(l_fast_list)
        l_slow = torch.stack(l_slow_list)
        # print("Group loss")
        loss = simclr_loss(l_fast,l_slow, args)
        loss = max(torch.tensor(0.000).cuda(),loss)
    else:
        loss= torch.tensor(0.0).cuda()
    return loss


def simclr_loss(output_fast,output_slow, args,normalize=True):
    out = torch.cat((output_fast, output_slow), dim=0)
    sim_mat = torch.mm(out, torch.transpose(out,0,1))
    if normalize:
        sim_mat_denom = torch.mm(torch.norm(out, dim=1).unsqueeze(1), torch.norm(out, dim=1).unsqueeze(1).t())
        sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)
    sim_mat = torch.exp(sim_mat / args.temperature)
    if normalize:
        sim_mat_denom = torch.norm(output_fast, dim=1) * torch.norm(output_slow, dim=1)
        sim_match = torch.exp(torch.sum(output_fast * output_slow, dim=-1) / sim_mat_denom / args.temperature)
    else:
        sim_match = torch.exp(torch.sum(output_fast * output_slow, dim=-1) / args.temperature)
    sim_match = torch.cat((sim_match, sim_match), dim=0)
    norm_sum = torch.exp(torch.ones(out.size(0)) / args.temperature )
    norm_sum = norm_sum.cuda()
    loss = torch.mean(-torch.log(sim_match / torch.abs(torch.sum(sim_mat, dim=-1) - norm_sum)))
  
    
    return loss




# def train_ssl(labeled_trainloader, unlabeled_trainloader, model, criterion, optimizer, epoch, mixup_fn, args):
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     total_losses = AverageMeter()
#     supervised_losses = AverageMeter()
#     contrastive_losses = AverageMeter()
#     group_contrastive_losses = AverageMeter()
#     pl_losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()

#     model = model.cuda()

   
#     # switch to train mode
#     model.train()
#     lenn = max(len(labeled_trainloader), len(unlabeled_trainloader))
#     if epoch >= args.sup_thresh:
#         data_loader = zip(cycle(labeled_trainloader), unlabeled_trainloader)
#     else:
#         data_loader = labeled_trainloader

#     end = time.time()


#     metric_logger = MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     metric_logger.add_meter('instance_constrastive_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     metric_logger.add_meter('group_constrastive_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     metric_logger.add_meter('supervised_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#     print_freq = 50

#     for data in metric_logger.log_every(data_loader, print_freq, lenn if epoch >= args.sup_thresh else len(labeled_trainloader), header):
#         # measure data loading time
#         data_time.update(time.time() - end)
        
#         #reseting losses

#         loss = torch.tensor(0.0).cuda()
#         contrastive_loss = torch.tensor(0.0).cuda()
#         group_contrastive_loss = torch.tensor(0.0).cuda()

#         if epoch >= args.sup_thresh:
#             (labeled_data,unlabeled_data) = data
#             images_8f, _ = unlabeled_data
#             images_8f = images_8f.cuda()
#             images_8f = torch.autograd.Variable(images_8f)


#             super_image_3x3, super_image_2x2 = create_super_image(images_8f, isLabeled=False)
            
#             save_super_image(super_image_3x3, "super_image_3x3new.jpg")
#             save_super_image(super_image_2x2, "super_image_2x2new.jpg")

#             # print(super_image_3x3.shape, super_image_2x2.shape)
            
#             output_8f = model(super_image_3x3)
#             output_4f = model(super_image_2x2)

#             output_8f_detach = output_8f.detach()
#             contrastive_loss = simclr_loss(torch.softmax(output_8f_detach,dim=1),torch.softmax(output_4f,dim=1), args)
    
#             grp_unlabeled_8seg = get_group(output_8f_detach)
#             grp_unlabeled_4seg = get_group(output_4f)
#             group_contrastive_loss = compute_group_contrastive_loss(grp_unlabeled_8seg,grp_unlabeled_4seg, args)         
#         else:
#             labeled_data = data
        
       
#         input, target_original = labeled_data
#         input, target = mixup_fn(input, target_original)

#         target = target.cuda()
#         input = input.cuda()
        
#         input = torch.autograd.Variable(input)
#         target_var = torch.autograd.Variable(target)
        
#         super_image_lab = create_super_image(input, isLabeled=True)

#         # save_super_image(super_image_lab, "super_image_lab.jpg")
#         output = model(super_image_lab)

#         # print("target shape: ", target_var.shape, output.shape)
#         loss = criterion(output, target_var)

#         total_loss = loss + args.gamma * contrastive_loss  + args.beta * group_contrastive_loss

#         # measure accuracy and record loss
        
#         prec1, prec5 = accuracy(output, target_original.cuda(), topk=(1, 5))
        
#         total_losses.update(total_loss.item(), input.size(0)+ args.mu*input.size(0))
        
#         supervised_losses.update(loss.item(), input.size(0))
#         contrastive_losses.update(contrastive_loss.item(), input.size(0)+args.mu*input.size(0))
#         group_contrastive_losses.update(group_contrastive_loss.item(), input.size(0)+args.mu*input.size(0))
        
#         top1.update(prec1.item(), input.size(0))
#         top5.update(prec5.item(), input.size(0))

#         # compute gradient and do SGD step
#         total_loss.backward()

#         optimizer.step()
#         optimizer.zero_grad()

#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()

#         # if i % 100 == 0:
#         #     output = ('Epoch: [{0}][{1}], lr: {lr:.5f}\t'
#         #               'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#         #               'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
#         #               'TotalLoss {total_loss.val:.4f} ({total_loss.avg:.4f})\t'
#         #               'Supervised Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#         #               'Contrastive_Loss {contrastive_loss.val:.4f} ({contrastive_loss.avg:.4f})\t'
#         #               'Group_contrastive_Loss {group_contrastive_loss.val:.4f} ({group_contrastive_loss.avg:.4f})\t'
#         #               'Pseudo_Loss {pl_loss.val:.4f} ({pl_loss.avg:.4f})\t'
#         #               'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#         #               'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
#         #                   epoch, i, batch_time=batch_time,
#         #                   data_time=data_time, total_loss=total_losses,loss=supervised_losses,
#         #                   contrastive_loss=contrastive_losses,group_contrastive_loss=group_contrastive_losses,pl_loss=pl_losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
#         #     print(output)
            
#         metric_logger.update(loss=total_losses.avg)
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
#         metric_logger.update(instance_constrastive_loss=contrastive_losses.avg)
#         metric_logger.update(group_constrastive_loss=group_contrastive_losses.avg)
#         metric_logger.update(supervised_loss=supervised_losses.avg)


#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
