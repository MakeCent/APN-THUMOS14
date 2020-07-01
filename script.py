from load_data import *
root = "/mnt/louis-consistent/Datasets/THUMOS14/Images/validation"
anndir = "/mnt/louis-consistent/Datasets/THUMOS14/Annotations/validation/annotationF"
datalist = read_from_anndir(root, anndir, (0, 100), orinal=True)