from load_data import *
from utils import *
from custom_class import MultiAction_BiasLayer
from pathlib import Path
from Flated_Inception import Inception_Inflated3d
from tensorflow.keras.layers import Dense, Dropout, Activation, Reshape
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import wandb
from wandb.keras import WandbCallback
import datetime
import tensorflow as tf
import socket
agent = socket.gethostname()
AUTOTUNE = tf.data.experimental.AUTOTUNE
fix_bug()
now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# %% wandb Initialization
# Configurations. If you don't use wandb, just manually set these values.
default_config = dict(
    y_s=1,
    y_e=100,
    learning_rate=0.0001,
    batch_size=32,
    epochs=20,
    agent=agent
)
ordinal = True
mode = 'flow'
stack_length = 10
weighted = False
pretrain = True
notes = 'i3d_KI_flow10'

# Just for wandb
tags = ['all', mode, 'i3d']
if ordinal:
    tags.append("od")
if weighted:
    tags.append("weighted")
if stack_length > 1:
    tags.append("stack{}".format(stack_length))
wandb.init(config=default_config, name=now, tags=tags, notes=notes)
config = wandb.config
wandbcb = WandbCallback(monitor='val_multi_od_metric', save_model=False)

y_range = (config.y_s, config.y_e)
y_nums = y_range[1] - y_range[0] + 1
learning_rate = config.learning_rate
batch_size = config.batch_size
epochs = config.epochs
action_num = 20

# %% Parameters, Configuration, and Initialization
model_name = now
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

output_path = '/mnt/louis-consistent/Saved/THUMOS14_output/Multi-task'  # Directory to save model and history
history_path = Path(output_path, 'History', model_name)
models_path = Path(output_path, 'Model', model_name)
history_path.mkdir(parents=True, exist_ok=True)
models_path.mkdir(parents=True, exist_ok=True)

# %% Build dataset
datalist = {x: read_from_anndir(root[x], anndir[x], mode=mode, y_range=y_range, ordinal=ordinal, stack_length=stack_length) for x in ['train', 'val', 'test']}
train_val_datalist = [a+b for a, b in zip(datalist['train'], datalist['val'])]

train_dataset = build_dataset_from_slices(*train_val_datalist, batch_size=batch_size, i3d=True, mode=mode)
test_dataset = build_dataset_from_slices(*datalist['test'], batch_size=batch_size, shuffle=False, i3d=True, mode=mode)

model_checkpoint = ModelCheckpoint(str(models_path.joinpath('{epoch:02d}-{val_multi_od_metric:.2f}.h5')), period=1)
with tf.distribute.MirroredStrategy().scope():
    backbone = Inception_Inflated3d(
        include_top=False,
        weights='{}_imagenet_and_kinetics'.format(mode if mode == 'rgb' else 'flow') if pretrain is not None else None,
        input_shape=(stack_length, 224, 224, 3 if mode == 'rgb' else 2))
    x = tf.keras.layers.Reshape((1024,))(backbone.output)
    x = Dense(2048, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.5)(x)
    x = Dense(action_num, kernel_initializer='he_uniform', use_bias=False)(x)
    x = MultiAction_BiasLayer(y_nums)(x)
    output = Activation('sigmoid')(x)
    model = Model(backbone.input, output)
    model.compile(loss=multi_binarycrossentropy, optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=[multi_od_metric])

ftune_his = model.fit(train_dataset, validation_data=test_dataset, callbacks=[model_checkpoint, wandbcb], epochs=epochs, verbose=1)

# %% Save history to csv and images
history = ftune_his.history
save_history(history_path, history)
plot_history(history_path, history)
