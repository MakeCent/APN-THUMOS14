import json
import numpy as np
import pandas as pd
from tools.utils import *

np.set_printoptions(suppress=True)

prediction_file_location = './fused_pre'
with open(prediction_file_location, 'r') as f:
    list_predictions = json.load(f)

predictions = {k: np.array(v) for k, v in list_predictions.items()}


action_idx = {'BaseballPitch': 0, 'BasketballDunk': 1, 'Billiards': 2, 'CleanAndJerk': 3, 'CliffDiving': 4,
              'CricketBowling': 5, 'CricketShot': 6, 'Diving': 7, 'FrisbeeCatch': 8, 'GolfSwing': 9,
              'HammerThrow': 10, 'HighJump': 11, 'JavelinThrow': 12, 'LongJump': 13, 'PoleVault': 14, 'Shotput': 15,
              'SoccerPenalty': 16, 'TennisSwing': 17, 'ThrowDiscus': 18, 'VolleyballSpiking': 19}

IoU = 0.5
# min_T, max_T, min_L = 60, 30, 60
down_sampling = 1


ap = {}
det = {}
gt = {}
best_ap = {}
best_parm = {}
for ac_name, ac_idx in action_idx.items():
    ac_ta = pd.read_csv(
        "{}/{}_testF.csv".format("/mnt/louis-consistent/Datasets/THUMOS14/Annotations/test/annotationF", ac_name),
        header=None).values
    ap[ac_name] = {}
    det[ac_name] = {}
    gt[ac_name] = {}
    for min_T in [60, 70, 80]:
        for max_T in [20, 30, 40]:
            for min_L in [30, 60, 80]:
                ac_ap, ac_det, ac_gt = action_ap(predictions, ac_ta, IoU=IoU, min_T=min_T, max_T=max_T, min_L=min_L,
                                                 down_sampling=down_sampling, action_idx=ac_idx, return_detections=True)
                ap[ac_name]["{}-{}-{}".format(min_T, max_T, min_L)] = ac_ap
                det[ac_name]["{}-{}-{}".format(min_T, max_T, min_L)] = ac_det
                gt[ac_name]["{}-{}-{}".format(min_T, max_T, min_L)] = ac_gt
    best_ap[ac_name] = ap[ac_name][max(ap[ac_name], key=ap[ac_name].get)]
    best_parm[ac_name] = max(ap[ac_name], key=ap[ac_name].get)
    print("{}: get best ap {:.2f} under {}".format(ac_name, best_ap[ac_name], best_parm[ac_name]))

mAP = np.array(list(best_ap.values())).mean()

print("mAP: {:.2f}".format(mAP))


