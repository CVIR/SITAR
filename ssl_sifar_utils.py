
import os
import time
from collections import defaultdict
import random

from sifar_pytorch.my_models.sifar_util import create_super_img, pad_frames

import numpy as np
from PIL import Image
import torch
try:
    from torch._six import inf
except:
    from torch import inf

from torch.nn import functional as F


def validate_split(labeled_file_path, unlabeled_file_path, percentage):
    main_dict_label= defaultdict(int)
    ul_lines = 0
    l_lines = 0
    with open(labeled_file_path, 'r') as fp:
        lines = fp.readlines()
        l_lines = len(lines)
        for line in lines:
            record =  line.strip().split()
            main_dict_label[int(record[3])] += 1
    
    main_dict_unlabel = defaultdict(int)
    with open(unlabeled_file_path, 'r') as fp:
        lines = fp.readlines()
        ul_lines = len(lines)
        for line in lines:
            record = line.strip().split()
            main_dict_unlabel[int(record[3])] += 1

    print("{:10} {:10}".format("Label", "Percent"))
    for label in range(101):
        per = main_dict_label[label] / (main_dict_label[label] + main_dict_unlabel[label])
        
        if(per > (percentage - 2) / 100 and per < (percentage + 2) / 100 ):
            print('\033[92m' + '{:<10} {:.6f}'.format(label, per) + '\033[0m')
        else:
            print('\033[91m' + '{:<10} {:.6f}'.format(label, per) + '\033[0m')
    
   
    print(f"Total labeled video: {l_lines}")
    print(f"Total unlabeled video: {ul_lines}")


def get_training_filenames(root, train_file_path, percent, strategy):
    labeled_file_path = os.path.join(root, "labeled_training.txt")
    unlabeled_file_path = os.path.join(root,"unlabeled_training.txt")
    split_file(train_file_path, unlabeled_file_path,
               labeled_file_path,percent, isShuffle=True, strategy=strategy)
    return labeled_file_path, unlabeled_file_path



def split_file(file, unlabeled, labeled, percentage, isShuffle=True, seed=123, strategy='classwise'):
    """Splits a file in 2 given the `percentage` to go in the large file."""
    if strategy == 'classwise':
        # if os.path.exists(unlabeled) and os.path.exists(labeled):
        #   print("path exists with this seed and strategy")
        #   return 
        random.seed(seed)
        #creating dictionary against each category
        def del_list(list_delete,indices_to_delete):
            for i in sorted(indices_to_delete, reverse=True):
                del(list_delete[i])

        main_dict= defaultdict(list)
        with open(file,'r') as mainfile:
            lines = mainfile.readlines()
            print(f"Total videos: {len(lines)}")
            for line in lines:
                video_info = line.strip().split()
                main_dict[video_info[3]].append((video_info[0],video_info[1], video_info[2]))
        with open(unlabeled,'w') as ul,\
            open(labeled,'w') as l:
            for key,value in main_dict.items():
                length_videos = len(value)
                ul_no_videos = int((length_videos* percentage))
                indices = random.sample(range(length_videos),ul_no_videos)
                for index in indices:
                    line_to_written = value[index][0] + " " + value[index][1] + " " + value[index][2] + " " +key+"\n"
                    ul.write(line_to_written)
                del_list(value,indices)
                for label_index in range(len(value)):
                    line_to_written = value[label_index][0] + " " + value[label_index][1] + " " + value[label_index][2] + " " +key+"\n"
                    l.write(line_to_written)
        

        


    if strategy == 'overall':
        if os.path.exists(unlabeled) and os.path.exists(labeled):
          print("path exists with this seed and strategy")
          return 
        random.seed(seed)
        with open(file, 'r') as fin, \
            open(unlabeled, 'w') as foutBig, \
            open(labeled, 'w') as foutSmall:
        # if didn't count you could only approximate the percentage
            lines = fin.readlines()
            random.shuffle(lines)
            nLines = sum(1 for line in lines)
            nTrain = int(nLines*percentage)
            i = 0
            for line in lines:
                line = line.rstrip('\n') + "\n"
                if i < nTrain:
                     foutBig.write(line)
                     i += 1
                else:
                     foutSmall.write(line)



def save_super_image_from_dataloader(data_loader, root, output, isLabeled, img_size, super_img_rows):
    
    for images_fast, images_slow in data_loader:
        print("Images fast shape", images_fast.shape)
        
        images_fast = pad_frames(images_fast.cuda(), 9, 1)
        x1 = create_super_img(images_fast, img_size, super_img_rows)
        x2 = ""
        print("Fast SI shape", x1.shape)
        if not isLabeled:
            images_slow = pad_frames(images_slow.cuda(), 4, 0)
            x2 = create_super_img(images_slow, img_size, 2)
            print("Slow SI shape", x2.shape)
        
       
        for i, x in enumerate([x1, x2]):
            temp_img = x[0]
            temp_img = temp_img.permute(1, 2, 0)
            numpy_image = temp_img.cpu().numpy()
            numpy_image = numpy_image * 255  # Scale values to [0, 255]
            numpy_image = numpy_image.astype(np.uint8)  # Convert to 8-bit unsigned integers

            
            image = Image.fromarray(numpy_image)

            # Save the image file
            path = os.path.join(root, str(i) + "_" +  output)
            image.save(path)
            print(f"Image saved {path}")
            if isLabeled:
                break
        break


def save_super_image(x, output):
    root = "/home/mt0/22CS60R54/ssl-sifar/superimages/"
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



## taken from https://github.com/microsoft/Swin-Transformer/blob/main/utils.py
def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm



## taken from https://github.com/microsoft/Swin-Transformer/blob/main/utils.py
class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                
                
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
