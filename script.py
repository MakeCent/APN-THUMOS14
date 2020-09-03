## Update wandb runs:
# import wandb
# import json
# api = wandb.Api()
# run = api.run("makecent/thumos14/tq0zqd25")
# with open("saved/HammerThrow_flow_search", 'r') as f:
#     ap = json.load(f)
# best_parm = max(ap, key=ap.get)
# best_ap = ap[best_parm]
# run.summary['average_precision'] = best_ap
# run.summary.update()

## Write new wandb.run from raw data
import socket
import pandas as pd
# file_path = "/mnt/louis-consistent/Saved/THUMOS14_output/CleanAndJerk/History/2020-08-09-10-16-29/history.csv"
# file = pd.read_csv(file_path)
# now ="2020-08-09-04-43-40"
# agent = socket.gethostname()
# default_config = dict(
#     y_s=1,
#     y_e=100,
#     learning_rate=0.0001,
#     batch_size=32,
#     epochs=50,
#     agent=agent,
#     action='CleanAndJerk',
#     mode='flow'
# )
# wandb.init(config=default_config, name=now)
# config = wandb.config
# ordinal = True
# stack_length = 10
# weighted = False
# pretrain = True
# # Configurations. If you don't use wandb, manually set above variables.
# tags = [default_config['action'], default_config['mode'], 'i3d']
# if ordinal:
#     tags.append("od")
# if weighted:
#     tags.append("weighted")
# if stack_length > 1:
#     tags.append("stack{}".format(stack_length))
#
# wandb.run.tags = tags
# wandb.run.notes = 'i3d_{}_{}'.format(default_config['action'], default_config['mode'])
# wandb.run.save()
#
#
# for epoch in range(50):
#     wandb.log({'epoch': epoch, 'loss': file['loss'][epoch]})
import json
action_idx = {'BasketballDunk': 1, 'Billiards': 2, 'CleanAndJerk': 3, 'CliffDiving': 4,
              'CricketBowling': 5, 'CricketShot': 6, 'Diving': 7, 'FrisbeeCatch': 8, 'GolfSwing': 9,
              'HammerThrow': 10, 'HighJump': 11, 'JavelinThrow': 12, 'LongJump': 13, 'Shotput': 15,
              'SoccerPenalty': 16, 'TennisSwing': 17, 'ThrowDiscus': 18, 'VolleyballSpiking': 19}
m = {}
ap = []
for action in action_idx.keys():
    with open('saved/{}_fused_search'.format(action), 'r') as f2:
        t = json.load(f2)
        parm = max(t, key=t.get)
        maximum = t[parm]
        m[action] = (parm, maximum)
        ap.append(maximum)

with open('saved/Seperate_rgb_search', 'w') as f1:
    for k, v in m.items():
        f1.write('{:17} get best ap {:.2} under {}'.format(k, v[1], v[0]))