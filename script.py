import tensorflow as tf
from PIL import Image
import numpy as np
op_path = "/mnt/louis-consistent/Datasets/THUMOS14/OpticalFlow/validation_optical_flow/video_validation_0000012/flow_y/flow_y_00576.jpg"
rgb_path = "/mnt/louis-consistent/Datasets/THUMOS14/Images/Validation/video_validation_0000012/00442.jpg"
op_tf = tf.io.read_file(op_path)
op_tf = tf.io.decode_jpeg(op_tf)
op_tf = tf.cast(op_tf, tf.float32)
rgb = tf.io.read_file(rgb_path)
rgb = tf.io.decode_jpeg(rgb)