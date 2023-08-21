from enum import Enum

# ************************************************ #
# Parameters
input_shape = (512, 512, 3)
input_shape_full_size = (1024, 320, 3)

num_classes = 6

colors = [(255, 255, 255), (255, 0, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255), (0, 0, 255)]



system_files = '.DS_Store'

use_unet = True


train_ratio = 0.8  # 80% for train, 20% for val
assert train_ratio > 0 and train_ratio <= 1

# ************************************************ #
data_train_image_dir = 'data2/image/split2'
data_train_gt_dir = 'data2/gt/split2'
data_test_image_dir = 'data2/test_image/split2'
data_test_gt_dir = 'data2/test_gt/split2'

data_location = 'data'

# ************************************************ #

def get_model_path(resnet_type):
    return "models/resnet{}_imagenet_1000_no_top.h5".format(str(resnet_type))


# ************************************************ #
# Model Types

class EncoderType(Enum):
    resnet18 = 18
    resnet34 = 34
    resnet50 = 50
    resnet101 = 101
    resnet152 = 152


classes = 1000
dataset = 'imagenet'
include_top = False
models_location = "models"
