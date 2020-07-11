from load_data import *
from utils import *
from custom_class import MultiAction_BiasLayer
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf

mode = 'flow'
model = tf.keras.models.load_model(
    "/mnt/louis-consistent/Saved/THUMOS14_output/Multi-task/Model/2020-07-07-17-07-17/10-18.25.h5",
    custom_objects={'MultiAction_BiasLayer': MultiAction_BiasLayer,
                    'multi_binarycrossentropy': multi_binarycrossentropy, 'multi_od_metric': multi_od_metric})
if mode == 'rgb':
    root = {'train': "/mnt/louis-consistent/Datasets/THUMOS14/Images/train",
            'val': "/mnt/louis-consistent/Datasets/THUMOS14/Images/validation",
            'test': "/mnt/louis-consistent/Datasets/THUMOS14/Images/test"}
elif mode == 'flow':
    root = {'train': "/mnt/louis-consistent/Datasets/THUMOS14/OpticalFlows/train",
            'val': "/mnt/louis-consistent/Datasets/THUMOS14/OpticalFlows/validation",
            'test': "/mnt/louis-consistent/Datasets/THUMOS14/OpticalFlows/test"}
elif mode == 'w_flow':
    root = {'train': "/mnt/louis-consistent/Datasets/THUMOS14/Warped_OpticalFlows/train",
            'val': "/mnt/louis-consistent/Datasets/THUMOS14/Warped_OpticalFlows/validation",
            'test': "/mnt/louis-consistent/Datasets/THUMOS14/Warped_OpticalFlows/test"}

anndir = {
    'train': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/train/annotationF",
    'val': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/validation/annotationF",
    'test': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/test/annotationF"}

video_names = pd.read_csv("/mnt/louis-consistent/Datasets/THUMOS14/Information/test_videos.txt",
                          header=None).values.squeeze().tolist()
untrimmed_predictions = {}
ground_truth = {}
for v in video_names:
    gt = pd.read_csv(
        "/mnt/louis-consistent/Datasets/THUMOS14/Information/video_wise_annotationF/test/{}_annotationF".format(v),
        header=None).values
    ground_truth[v] = gt

    video_path = Path(root['test'], v)
    if mode == 'rgb':
        data_list = find_imgs(video_path, stack_length=10)
    else:
        data_list = find_flows(video_path, stack_length=10)

    ds = build_dataset_from_slices(data_list, batch_size=1, shuffle=False, i3d=True, mode=mode)
    untrimmed_prediction = model.predict(ds, verbose=1)
    untrimmed_predictions[v] = np.squeeze(untrimmed_prediction)

# %% Detect action
action_idx = {'BaseballPitch': 0, 'BasketballDunk': 1, 'Billiards': 2, 'CleanAndJerk': 3, 'CliffDiving': 4,
              'CricketBowling': 5, 'CricketShot': 6, 'Diving': 7, 'FrisbeeCatch': 8, 'GolfSwing': 9,
              'HammerThrow': 10, 'HighJump': 11, 'JavelinThrow': 12, 'LongJump': 13, 'PoleVault': 14, 'Shotput': 15,
              'SoccerPenalty': 16, 'TennisSwing': 17, 'ThrowDiscus': 18, 'VolleyballSpiking': 19}

iou = 0.5
all_detected_action = {}
mutual_exclude = False
ordinal = True
for v, p in untrimmed_predictions.items():
    if ordinal:
        p = ordinal2completeness(p)
    v_ads = []
    for ac_name, ac_idx in action_idx.items():
        ac_prediction = p[:, ac_idx]
        ac_ads = action_search(ac_prediction, min_T=75, max_T=25, min_L=20)
        ac_ads = np.c_[ac_ads, np.ones(ac_ads.shape[0]) * ac_idx]
        v_ads.append(ac_ads)
    v_ads = np.vstack(v_ads)
    if mutual_exclude:
        iou_matrix = matrix_iou(v_ads[:, :2], v_ads[:, :2])
        if iou_matrix.size > 0:
            for i in range(iou_matrix.shape[0]):
                iou_matrix[i, i] = 0
            while iou_matrix.max() > 0 and iou:
                max_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                iou_matrix = np.delete(np.delete(iou_matrix, max_idx[0], axis=0), max_idx[0], axis=1) if \
                v_ads[max_idx[0]][2] > v_ads[max_idx[1]][
                    2] else np.delete(np.delete(iou_matrix, max_idx[1], axis=0), max_idx[1], axis=1)
                v_ads = np.delete(v_ads, max_idx[0], axis=0) if v_ads[max_idx[0]][2] > v_ads[max_idx[1]][
                    2] else np.delete(
                    v_ads, max_idx[1], axis=0)
            all_detected_action[v] = v_ads
        else:
            all_detected_action[v] = v_ads
    else:
        all_detected_action[v] = v_ads

all_detection = np.vstack(list(all_detected_action.values()))

# %% Calculate Average Precision
tps = {}
ap = {}
for ac_name, ac_idx in action_idx.items():
    ac_tps = {}
    ac_det = {}
    ac_gt = {}
    for v in video_names:
        v_gt = ground_truth[v]
        ac_v_gt = v_gt[v_gt[:, 0] == ac_idx][:, 1:]
        if ac_v_gt.size == 0:
            continue
        v_det = all_detected_action[v]
        ac_v_det = v_det[v_det[:, 3] == ac_idx][:, :-1]
        ac_v_tps = calc_truepositive(ac_v_det, ac_v_gt, iou)

        ac_det[v] = ac_v_det
        ac_tps[v] = ac_v_tps
        ac_gt[v] = ac_v_gt

    ac_num_gt = np.vstack(list(ac_gt.values())).shape[0]
    ac_loss = np.vstack(list(ac_det.values()))[:, 2]
    ac_tp_values = np.hstack(list(ac_tps.values()))
    ac_ap = average_precision(ac_tp_values, ac_num_gt, ac_loss)

    tps[ac_name] = ac_tp_values
    ap[ac_name] = ac_ap

mAP = np.array(list(ap.values())).mean()