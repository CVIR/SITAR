import argparse
from pathlib import Path
import cv2
from ssl_sifar_utils import get_training_filenames, validate_split
import os

def get_video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count



def create_label_dict(path):
    with open(path, 'r') as fp:
        lines = fp.readlines()
        label_to_idx = {}
        for line in lines:
            index, label = line.split()
            label_to_idx[label.strip()] = int(index.strip()) - 1
        
    return label_to_idx

def create_train_val_list(dataset_root, listpath, label_to_idx, output_dir):
    
    with open(listpath, 'r') as fp:
        lines = fp.readlines()
        train_list = []
        count = 0
        print("Please Wait...")
        for line in lines:
            vpath = line.split()[0]
            label = vpath.split('/')[0]
            vpath = dataset_root + vpath
            fcount = get_video_frame_count(vpath)
            train_list += [[vpath, str(1), str(fcount), str(label_to_idx[label])]]
            

       
        with open(os.path.join(output_dir, 'train.txt'), 'w') as fp:
            for record in train_list:
                print(" ".join(record), file=fp)
            print("train.txt saved")
        return os.path.join(output_dir, 'train.txt')
       
            

def create_test_list(dataset_root, listpath, label_to_idx, output_dir):
     with open(listpath, 'r') as fp:
        lines = fp.readlines()
        test_list = []
        count = 0
        print('Please Wait...')
        for line in lines:
            vpath = line.split()[0]
            label = vpath.split('/')[0]
            vpath = dataset_root + vpath
            fcount = get_video_frame_count(vpath)
            test_list += [[vpath, str(1), str(fcount), str(label_to_idx[label])]]
            

        with open(os.path.join(output_dir, 'test.txt'), 'w') as fp:
            for record in test_list:
                print(" ".join(record), file=fp)
            print("test.txt saved")

def get_args_parser():
    parser = argparse.ArgumentParser('Datapreparation script', add_help=False)
    parser.add_argument('--dataset_root',  type=str)
    parser.add_argument('--output_dir', default='/dataset_list/default', type=str)
    parser.add_argument('--trainlist_path', type=str)
    parser.add_argument('--testlist_path', type=str)
    parser.add_argument('--percentage', type=int, default=10)
    return parser


def main(args):
    
    classes = os.listdir(args.dataset_root)
    label_to_idx = {item:i for i, item in enumerate(classes)}
    print(label_to_idx)


    train_list = create_train_val_list(args.dataset_root, args.trainlist_path, label_to_idx, args.output_dir)
    create_test_list(args.dataset_root, args.testlist_path, label_to_idx, args.output_dir)
    train_list = os.path.join(args.output_dir, 'train.txt')
    train_label_list, train_unlabel_list = get_training_filenames(args.output_dir, train_list,(100 - args.percentage) / 100, 'classwise')

    validate_split(train_label_list, train_unlabel_list, args.percentage)



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Dataset perperation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
