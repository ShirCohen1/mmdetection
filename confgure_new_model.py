import mmcv
from mmcv import Config
import os.path as osp

from mmdet.apis import inference_detector, show_result_pyplot


from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector


cfg = Config.fromfile('./configs/yolo/yolov3_d53_320_273e_coco.py')

cfg.data.test.ann_file = 'data/coco/annotations/instancesonly_filtered_gtFine_test.json'
cfg.data.test.img_prefix = 'data/coco/cityscapes'
#
# cfg.data.train.type = 'CocoDataset'
# cfg.data.train.data_root = 'kitti_tiny/'
cfg.data.train.ann_file = 'data/coco/annotations/instancesonly_filtered_gtFine_train.json'
cfg.data.train.img_prefix = 'data/coco/cityscapes'
#
# cfg.data.val.type = 'KittiTinyDataset'
# cfg.data.val.data_root = 'kitti_tiny/'
cfg.data.val.ann_file = 'data/coco/annotations/instancesonly_filtered_gtFine_val.json'
cfg.data.val.img_prefix = 'data/coco/cityscapes'


cfg.load_from = 'checkpoints/yolov3_d53_320_273e_coco-421362b6.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './tutorial_exps'

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 20
cfg.runner.max_epochs = 1
# Change the evaluation metric since we use customized dataset.
cfg.evaluation.metric = 'bbox_mAP'   # 'mAP' or 'bbox_mAP'
# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 12
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 12

# Set seed thus the results are more reproducible
cfg.seed = 0
# set_random_seed(0, deterministic=False)
cfg.device = 'cpu'
cfg.gpu_ids = range(1)

# cfg.model.bbox_head.num_classes = 8


# We can also use tensorboard to log the training process
cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')]

print(f'Config:\n{cfg.pretty_text}')

# Build dataset
datasets = [ build_dataset(cfg.data.val)] # build_dataset(cfg.data.train),

print('built dataset')

# Build the detector
model = build_detector(cfg.model)

# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES
print(model.CLASSES)

cfg.workflow = [("val",1)]
print(cfg.workflow)
if __name__ == '__main__':

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=True)

# img = mmcv.imread('data/coco/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png')
# # #
# model.cfg = cfg
# result = inference_detector(model, img)
# show_result_pyplot(model, img, result)
# print(result)
