## SITAR: Semi-Supervised Image Transformer for Action Recognition

This repo contains implementation of our work SITAR for semi-supervised action recognition in videos.

Owais Iqbal, Omprakash Chakraborty, Aftab Hussain, Rameswar Panda, and Abir Das

## Requirements
The code is written for python `3.8.10`, but should work for other version with some modifications.
```

pip install -r requirements.txt
```

## Data Preparation

The code expect you to have your dataset stored in the following format:
```
-- train_split.txt
-- test_split.txt
-- dataset_dir
---- Action_Class_1
------ video_clip1.avi 
------ video_clip2.avi
------ video_clip3.avi
------ ...
---- Action_Class_2
------ video_clip1.avi 
------ video_clip2.avi
------ video_clip3.avi
------ ...
```


### Test train split

The code also expect you to have two file one for the train split and other for the test split (generally the official train and test split of any dataset). Each of the file contain relative path to dataset directory of one video clip and the number of frame in that clip.

example for both the file are given below.

```
$ head ucf101_train_split_1_videos.txt  -n 3

PushUps/v_PushUps_g20_c03.avi 71
FloorGymnastics/v_FloorGymnastics_g09_c05.avi 29
PlayingFlute/v_PlayingFlute_g17_c07.avi 61

$ head ucf101_val_split_1_videos.txt  -n 3

RockClimbingIndoor/v_RockClimbingIndoor_g03_c06.avi 73
FrisbeeCatch/v_FrisbeeCatch_g01_c05.avi 30
Punch/v_Punch_g03_c03.avi 70
```


## Labeled and Unlabeled split

The following script will split the training data into labeled and unlabeled data and generates four file inside `output_dir`. Each line in all four file `train.txt`,  `test.txt`, `labeled_training.txt` and `unlabeled_training.txt` includes 4 elements and separated by space. Four elements (in order) include (1)relative paths to video_x_folder from dataset_dir, (2) starting frame number (3) total number of frames, (4) label id (a numeric number). 
`x%` from each class is taken as Labeled data. 

```
 python3 data_preparation_av.py \
 --dataset_root '/path/to/dataset/videos' \
 --output_dir '/path/to/save/video_list' \
 --trainlist_path '/path/to/train_list/mentioned above' \
 --testlist_path '/path/to/train_list/mentioned above' \
 --percentage x
```


## Python script overview

`main.py` - Entry point for the code. It contain default value for different parameter.

`data_preparation+av.py` - It contain script for generating `x%` of labeled spilt from the training set.

`/sifar_pytorch/engine.py` - It contain the training loop and the evaluate method.

`/sifar_pytorch/my_models/sifar_swin.py` - It contain model classes.

## Key Parameters

`data_dir` - path to the dataset videos.\
`list_root` - path to the output_dir created in Labeled and Unlabeled split section.\
`lr` - starting learning rate.\
`sched` - LR schedular.\
`epochs` - Total number of epochs.\
`sup_thresh` - Number of supervised only epochs.\
`batch_size` - labeled batch size.\
`mu` - factor for unlabeled batch size. `unlabeled batch size = mu * labeled batch size`.\
`output_dir` - path where the log and checkpoints will be saved.\
`gamma` - factor to instance contrastive loss.\
`beta` - factor to group contrastive loss. 



## Sample Code to train



```
python3 main.py \
 --data_dir <PATH_TO_DATASET> \
 --list_root <PATH_TO_ANNOTATION_FOLDER> \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 2 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir <PATH_TO_OUTPUT_FOLDER> --hpe_to_token \
 --sup_thresh 150 --num_workers 4 --mu 7 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1
 ```# SITAR
