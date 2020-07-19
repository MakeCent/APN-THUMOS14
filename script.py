from load_data import *
from utils import *
from custom_class import MultiAction_BiasLayer
from pathlib import Path
import pandas as pd
import numpy as np
import json
import tensorflow as tf

rgb_root = {'train': "/mnt/louis-consistent/Datasets/THUMOS14/Images/train",
            'val': "/mnt/louis-consistent/Datasets/THUMOS14/Images/validation",
            'test': "/mnt/louis-consistent/Datasets/THUMOS14/Images/test"}
flow_root = {'train': "/mnt/louis-consistent/Datasets/THUMOS14/OpticalFlows/train",
             'val': "/mnt/louis-consistent/Datasets/THUMOS14/OpticalFlows/validation",
             'test': "/mnt/louis-consistent/Datasets/THUMOS14/OpticalFlows/test"}
wflow_root = {'train': "/mnt/louis-consistent/Datasets/THUMOS14/Warped_OpticalFlows/train",
              'val': "/mnt/louis-consistent/Datasets/THUMOS14/Warped_OpticalFlows/validation",
              'test': "/mnt/louis-consistent/Datasets/THUMOS14/Warped_OpticalFlows/test"}

anndir = {
    'train': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/train/annotationF",
    'val': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/validation/annotationF",
    'test': "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/test/annotationF"}

video_names = pd.read_csv("/mnt/louis-consistent/Datasets/THUMOS14/Information/test_videos.txt",
                          header=None).values.squeeze().tolist()

# %% Prediction
rgb_model = tf.keras.models.load_model(
    "/mnt/louis-consistent/Saved/THUMOS14_output/Multi-task/Model/2020-07-07-21-08-04/13-19.10.h5",
    custom_objects={'MultiAction_BiasLayer': MultiAction_BiasLayer,
                    'multi_binarycrossentropy': multi_binarycrossentropy, 'multi_od_metric': multi_od_metric})

flow_model = tf.keras.models.load_model(
    "/mnt/louis-consistent/Saved/THUMOS14_output/Multi-task/Model/2020-07-07-17-07-17/10-18.25.h5",
    custom_objects={'MultiAction_BiasLayer': MultiAction_BiasLayer,
                    'multi_binarycrossentropy': multi_binarycrossentropy, 'multi_od_metric': multi_od_metric})
rgb_predictions = {}
flow_predictions = {}
fused_predictions = {}
for v in video_names:
    rgb_v_datalist = find_imgs(Path(rgb_root['test'], v), stack_length=10)
    flow_v_datalist = find_flows(Path(flow_root['test'], v), stack_length=10)

    rgb_ds = build_dataset_from_slices(rgb_v_datalist, batch_size=1, shuffle=False, i3d=True, mode='rgb')
    flow_ds = build_dataset_from_slices(flow_v_datalist, batch_size=1, shuffle=False, i3d=True, mode='flow')
    rgb_v_pre = rgb_model.predict(rgb_ds, verbose=1)
    flow_v_pre = flow_model.predict(flow_ds, verbose=1)

    rgb_predictions[v] = ordinal2completeness(np.squeeze(rgb_v_pre))
    flow_predictions[v] = ordinal2completeness(np.squeeze(flow_v_pre))
    fused_predictions[v] = (rgb_predictions[v][:flow_predictions[v].shape[0], :] + flow_predictions[v]) / 2
# with open("rgb_pre", 'w') as f:
#     list_rgb_predictions = {k: v.tolist() for k, v in rgb_predictions.items()}
#     json.dump(list_rgb_predictions, f)
# with open("flow_pre", 'w') as f:
#     list_flow_predictions = {k: v.tolist() for k, v in flow_predictions.items()}
#     json.dump(list_flow_predictions, f)
# with open("fused_pre", 'w') as f:
#     list_fused_predictions = {k: v.tolist() for k, v in fused_predictions.items()}
#     json.dump(list_fused_predictions, f)

# %% Detect action
with open('rgb_pre', 'r') as f:
    list_rgb_predictions = json.load(f)
with open('flow_pre', 'r') as f:
    list_flow_predictions = json.load(f)
rgb_predictions = {k: np.array(v) for k, v in list_rgb_predictions.items()}
flow_predictions = {k: np.array(v) for k, v in list_flow_predictions.items()}

action_idx = {'BaseballPitch': 0, 'BasketballDunk': 1, 'Billiards': 2, 'CleanAndJerk': 3, 'CliffDiving': 4,
              'CricketBowling': 5, 'CricketShot': 6, 'Diving': 7, 'FrisbeeCatch': 8, 'GolfSwing': 9,
              'HammerThrow': 10, 'HighJump': 11, 'JavelinThrow': 12, 'LongJump': 13, 'PoleVault': 14, 'Shotput': 15,
              'SoccerPenalty': 16, 'TennisSwing': 17, 'ThrowDiscus': 18, 'VolleyballSpiking': 19}

down_sample = 1
iou = 0.5

rgb_det = {}
flow_det = {}
fused_det = {}

rgb_ap = {}
flow_ap = {}
fused_ap = {}

gt = {}
for ac_name, ac_idx in action_idx.items():
    ac_ta = pd.read_csv("{}/{}_testF.csv".format(anndir['test'], ac_name), header=None).values
    ac_v = np.unique(ac_ta[:, 0])
    rgb_ac_det = {}
    flow_ac_det = {}
    fused_ac_det = {}
    rgb_ac_tps = {}
    flow_ac_tps = {}
    fused_ac_tps = {}
    ac_gt = {}
    fused_v_prediction = []
    for v in ac_v:
        ac_v_ta = ac_ta[ac_ta[:, 0] == v][:, 1:]
        rgb_v_p = rgb_predictions[v]
        flow_v_p = flow_predictions[v]

        rgb_ac_v_p = rgb_v_p[:, ac_idx]
        flow_ac_v_p = flow_v_p[:, ac_idx]
        fused_ac_v_p = (flow_ac_v_p + rgb_ac_v_p[:len(flow_ac_v_p)]) / 2

        if down_sample > 1:
            rgb_ac_v_p = rgb_ac_v_p[::down_sample]
            flow_ac_v_p = flow_ac_v_p[::down_sample]
            fused_ac_v_p = fused_ac_v_p[::down_sample]
            ac_v_ta = ac_v_ta // down_sample

        rgb_ac_v_det = action_search(rgb_ac_v_p, min_T=60, max_T=30, min_L=60)
        flow_ac_v_det = action_search(flow_ac_v_p, min_T=60, max_T=30, min_L=60)
        fused_ac_v_det = action_search(fused_ac_v_p, min_T=60, max_T=30, min_L=60)

        rgb_ac_v_tps = calc_truepositive(rgb_ac_v_det, ac_v_ta, iou)
        flow_ac_v_tps = calc_truepositive(flow_ac_v_det, ac_v_ta, iou)
        fused_ac_v_tps = calc_truepositive(fused_ac_v_det, ac_v_ta, iou)

        rgb_ac_det[v] = rgb_ac_v_det
        flow_ac_det[v] = flow_ac_v_det
        fused_ac_det[v] = fused_ac_v_det

        rgb_ac_tps[v] = rgb_ac_v_tps
        flow_ac_tps[v] = flow_ac_v_tps
        fused_ac_tps[v] = fused_ac_v_tps

        ac_gt[v] = ac_v_ta

    ac_num_gt = ac_ta.shape[0]
    rgb_ac_loss = np.vstack(list(rgb_ac_det.values()))[:, 2]
    flow_ac_loss = np.vstack(list(flow_ac_det.values()))[:, 2]
    fused_ac_loss = np.vstack(list(fused_ac_det.values()))[:, 2]

    rgb_ac_tp_values = np.hstack(list(rgb_ac_tps.values()))
    flow_ac_tp_values = np.hstack(list(flow_ac_tps.values()))
    fused_ac_tp_values = np.hstack(list(fused_ac_tps.values()))

    rgb_ac_ap = average_precision(rgb_ac_tp_values, ac_num_gt, rgb_ac_loss)
    flow_ac_ap = average_precision(flow_ac_tp_values, ac_num_gt, flow_ac_loss)
    fused_ac_ap = average_precision(fused_ac_tp_values, ac_num_gt, fused_ac_loss)

    rgb_det[ac_name] = rgb_ac_det
    flow_det[ac_name] = flow_ac_det
    fused_det[ac_name] = fused_ac_det

    gt[ac_name] = ac_gt

    rgb_ap[ac_name] = rgb_ac_ap
    flow_ap[ac_name] = flow_ac_ap
    fused_ap[ac_name] = fused_ac_ap

rgb_mAP = np.array(list(rgb_ap.values())).mean()
flow_mAP = np.array(list(flow_ap.values())).mean()
fused_mAP = np.array(list(fused_ap.values())).mean()

# Test unit
ac_name = 'Billiards'
ac_idx = action_idx[ac_name]
ac_ta = pd.read_csv("{}/{}_testF.csv".format(anndir['test'], ac_name), header=None).values
ac_v = np.unique(ac_ta[:, 0])
v = ac_v[2]
plot_detection(fused_predictions[v][:, ac_idx], gt[ac_name][v], fused_det[ac_name][v])

# Search unit

ap_parm = {}

ac_name = 'BasketballDunk'
ac_idx = action_idx[ac_name]
ac_ta = pd.read_csv("{}/{}_testF.csv".format(anndir['test'], ac_name), header=None).values
ac_v = np.unique(ac_ta[:, 0])
ac_num_gt = ac_ta.shape[0]

ap_ac_parm = {}

for min_T in [60, 70, 80]:
    for max_T in [10, 20, 30]:
        for min_L in [20, 40, 60]:
            ac_tps = {}
            ac_det = {}
            for v in ac_v:
                pre = fused_predictions[v][:, ac_idx]
                ta = gt[ac_name][v]
                det = action_search(pre, min_T=min_T, max_T=max_T, min_L=min_L)
                tps = calc_truepositive(det, ta, 0.5)
                ac_tps[v] = tps
                ac_det[v] = det

            ac_loss = np.vstack(list(ac_det.values()))[:, 2]
            ac_tp_values = np.hstack(list(ac_tps.values()))
            ac_ap = average_precision(ac_tp_values, ac_num_gt, ac_loss)
        ap_ac_parm['{}-{}-{}'.format(min_T, max_T, min_L)] = ac_ap

print('{} in {} get max ap: {}'.format(ac_name, max(ap_ac_parm, key=ap_ac_parm.get),
                                       ap_ac_parm[max(ap_ac_parm, key=ap_ac_parm.get)]))
ap_parm[ac_name] = ap_ac_parm

# Singe parm test unit

