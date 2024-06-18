##
##
##

import functools
import json

from detectron2.data import DatasetCatalog, MetadataCatalog

from detrex.config import get_config
from projects.dino.configs.models.dino_r50 import model


# Dataset registration
def load_vg(split: str) -> list[dict]:
    with open(f"datasets/vg/instances_{split}.json") as f:
        data = json.load(f)

    for sample in data:
        sample["file_name"] = f"datasets/vg/images/{sample['image_id']}.jpg"

    return data


with open("datasets/vg/classes.json") as f:
    classes: list[str] = json.load(f)

splits = ["train", "val"]
for split in splits:
    name = f"vg_{split}"
    DatasetCatalog.register(name, functools.partial(load_vg, split=split))
    MetadataCatalog.get(name).thing_classes = classes.copy()

# Config registration

# get default config
dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_24ep
train = get_config("common/train.py").train

# change the number of classes from COCO to VG
model.num_classes = 150

# modify model config
# use the original implementation of dab-detr position embedding in 24 epochs training.
model.position_embedding.temperature = 20
model.position_embedding.offset = 0.0

# modify training config
train.seed = 3407
train.init_checkpoint = "./dino_r50_4scale_24ep.pth"
train.output_dir = "./output/dino_r50_4scale_24ep"

# max training iterations
train.max_iter = 180000
train.eval_period = 5000
train.log_period = 20
train.checkpointer.period = 5000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.num_classes = 150
model.device = train.device

# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = (
    lambda module_name: 0.1 if "backbone" in module_name else 1
)

dataloader.train.dataset.names = "vg_train"
dataloader.test.dataset.names = "vg_val"
# modify dataloader config
# not filter empty annotations during training
dataloader.train.dataset.filter_empty = False
dataloader.train.num_workers = 16
# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 1
# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
