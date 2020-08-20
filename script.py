import wandb
import json
api = wandb.Api()
run = api.run("makecent/thumos14/9vpxrg4k")
with open("saved/CliffDiving_flow_search", 'r') as f:
    ap = json.load(f)
best_parm = max(ap, key=ap.get)
best_ap = ap[best_parm]
run.summary['average_precision'] = best_ap
run.summary.update()
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
