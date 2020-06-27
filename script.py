import tensorflow as tf
from pathlib import Path
import numpy as np
import pandas as pdT
path = Path('/mnt/louis-consistent/Datasets/THUMOS14/Videos/validation_set_mp4')
ann = '/mnt/louis-consistent/Datasets/THUMOS14/Annotations/validation/annotationF'
videos_of_20_action = []
for v in Path(ann).glob('[!A]*.csv'):
    vl = np.unique(pd.read_csv(v, header=None).iloc[:, 0].values)
    videos_of_20_action.extend(vl)
videos_of_20_action = list(np.unique(videos_of_20_action))

with open('/mnt/louis-consistent/Datasets/THUMOS14/Information/action_list', 'r') as f:
    action_list = [line.rstrip('\n') for line in f]
with open('validation_video_list.txt', 'a') as f:
    for v in path.glob('*.mp4'):
        if v.stem in videos_of_20_action:
            f.writelines('/data/Videos/validation_set_mp4/' + v.name + '\n')