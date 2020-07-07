from Flated_Inception import Inception_Inflated3d
rgb_model = Inception_Inflated3d(
    include_top=False,
    weights='flow_imagenet_and_kinetics',
    input_shape=(10, 224, 224, 2))