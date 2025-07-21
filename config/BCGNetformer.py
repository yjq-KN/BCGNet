from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.uavid_dataset import *
from geoseg.models.UNetFormer import UNetFormer
from tools.utils import Lookahead
from tools.utils import process_model_params
from geoseg.models.CMTFNet import CMTFNet
from geoseg.models.MNet import MNet
from geoseg.models.deeplabv3_plus import DeepLab

max_epoch = 40
ignore_index = 255
train_batch_size = 4
val_batch_size = 4
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
num_classes = len(CLASSES_BottleAndGate)
classes = CLASSES_BottleAndGate

weights_name = ""
weights_path = "model_weights/{}".format(weights_name)
test_weights_name = "101_HSO"
log_name = 'Log/{}'.format(weights_name)
monitor = 'val_IoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
resume_ckpt_path = None

net = MNet(num_classes=num_classes)
loss = UnetFormerLoss(ignore_index=ignore_index)
use_aux_loss = True

train_dataset = UAVIDDataset(data_root='data_name/train_val', img_dir='images', mask_dir='masks',
                             mode='train', mosaic_ratio=0.25, transform=train_aug, img_size=(512, 512))

val_dataset = UAVIDDataset(data_root='data_name/val', img_dir='images', mask_dir='masks', mode='val',
                           mosaic_ratio=0.0, transform=val_aug, img_size=(512, 512))


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)


layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)

