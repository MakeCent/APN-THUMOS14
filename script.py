import tensorflow as tf
from pathlib import Path
import numpy as np
path = Path("/mnt/louis-consistent/Datasets/THUMOS14/Videos/train_set_avi/UCF101")
with open('/mnt/louis-consistent/Datasets/THUMOS14/Information/action_list', 'r') as f:
    action_list = [line.rstrip('\n') for line in f]
with open('train_video_list.txt', 'a') as f:
    for v in path.glob('*.avi'):
        if v.stem.split('_')[1] in action_list:
            f.writelines('/data/Videos/train_set_avi/UCF101/' + v.name + '\n')