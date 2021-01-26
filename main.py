import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
import torch
from torch import nn
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2
import apex
from utils import seed_everything
from datasets import CassavaDataset
from train import Trainer
from model import CassvaImgClassifier
from loss_function import CrossEntropyLossOneHot

CFG = {
    'fold_num': 0,
    'seed': 2021,
    'model_arch': 'tf_efficientnet_b4_ns',
    'img_size': 512,
    'epochs': 10,
    'train_bs': 16,
    'valid_bs': 32,
    'T_0': 10,
    'lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-6,
    'num_workers': 8,
    'accum_iter': 2,  # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 1,
    'device': 'cuda:0',
    'log_file': "/home/samenko/Cassava/logs/tmp.log",
    'test_fold': 0,
    'fp16':True,
    'soft_labels_file': '/home/samenko/Cassava/tmp/soft_labels.csv',
    'save_model': False
}

TRAIN_AUGS = Compose([
    RandomResizedCrop(CFG['img_size'], CFG['img_size']),
    Transpose(p=0.5),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    ShiftScaleRotate(p=0.5),
    HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
    RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
    CoarseDropout(p=0.5),
    Cutout(p=0.5),
    ToTensorV2(p=1.0),
], p=1.)

TEST_AUGS = Compose([
    CenterCrop(CFG['img_size'], CFG['img_size'], p=1.0),
    Resize(CFG['img_size'], CFG['img_size']),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
    ToTensorV2(p=1.0),
], p=1.)

def train_crossval(data):
    device = torch.device('cuda')

    for dev_fold in [1, 2, 3, 4]:
        train_data = data[(data.fold != CFG['test_fold']) & (data.fold != dev_fold)].reset_index(drop=True)
        val_data = data[data.fold == CFG['test_fold']].reset_index(drop=True)

        train_ds = CassavaDataset(train_data, TRAIN_AUGS, TEST_AUGS, mode='train')
        valid_ds = CassavaDataset(val_data, TRAIN_AUGS, TEST_AUGS, mode='val')

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=CFG['train_bs'],
            pin_memory=False,
            drop_last=False,
            shuffle=True,
            num_workers=CFG['num_workers'])

        val_loader = torch.utils.data.DataLoader(
            valid_ds,
            batch_size=CFG['valid_bs'],
            num_workers=CFG['num_workers'],
            shuffle=False,
            pin_memory=False)

        model = CassvaImgClassifier(CFG['model_arch'], data.label.nunique(), pretrained=True).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                         T_0=CFG['epochs'],
                                                                         T_mult=1,
                                                                         eta_min=1e-6,
                                                                         last_epoch=-1)
        trainer = Trainer(CFG, scheduler)
        loss_tr = nn.CrossEntropyLoss().to(device)
        loss_fn = nn.CrossEntropyLoss().to(device)
        if CFG['fp16']:
            model, optimizer = apex.amp.initialize(
                model,
                optimizer,
                opt_level='O1')
        for epoch in range(CFG['epochs']):
            print(f"epoch number {epoch}, lr = {optimizer.param_groups[0]['lr']}")
            trainer.train_one_epoch(model, optimizer, train_loader, loss_tr, epoch)
            val_acc, val_loss = trainer.valid_one_epoch(model, optimizer, val_loader, loss_fn, epoch)
            if CFG['save_model']:
                torch.save(model.state_dict(),
                           '/home/samenko/Cassava/output/{}_dev_fold_{}_test_fold_{}_epoch_{}_val_loss_{:.4f}_val_acc_{:.4f}'
                           .format( CFG['model_arch'], dev_fold, CFG['test_fold'], epoch, val_acc, val_loss))

        torch.cuda.empty_cache()

def train_one_model(data):
    device = torch.device('cuda')
    soft_labels = pd.read_csv('./tmp/soft_labels.csv')
    soft_labels = soft_labels.dropna()
    soft_labels.reset_index(inplace=True, drop=True)
    train_data = soft_labels[(soft_labels.fold != CFG['test_fold'])].reset_index(drop=True)
    val_data = data[data.fold == CFG['test_fold']].reset_index(drop=True)
    train_ds = CassavaDataset(train_data,TRAIN_AUGS, TEST_AUGS, mode='train' ,soft=True)
    valid_ds = CassavaDataset(val_data, TRAIN_AUGS, TEST_AUGS, mode='val', soft=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=CFG['train_bs'],
        pin_memory=False,
        drop_last=False,
        shuffle=True,
        num_workers=CFG['num_workers'])

    val_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=CFG['num_workers'],
        num_workers=8,
        shuffle=False,
        pin_memory=False)
    model = CassvaImgClassifier(CFG['model_arch'], data.label.nunique(), pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG['epochs'], T_mult=1,
                                                                     eta_min=1e-6, last_epoch=-1)

    trainer = Trainer(CFG, scheduler)
    loss_tr = CrossEntropyLossOneHot().to(device)
    loss_val = nn.CrossEntropyLoss().to(device)
    if CFG['fp16']:
        model, optimizer = apex.amp.initialize(
            model,
            optimizer,
            opt_level='O1')
    for epoch in range(CFG['epochs']):
        print(f"epoch number {epoch}, lr = {optimizer.param_groups[0]['lr']}")
        trainer.train_one_epoch(model, optimizer, train_loader, loss_tr, epoch)
        val_acc, val_loss = trainer.valid_one_epoch(model, optimizer, val_loader, loss_val, epoch)
        if CFG['save_model']:
            torch.save(model.state_dict(),'./output/{}_test_fold_{}_epoch_{}_val_loss_{:.4f}_val_acc_{:.4f}'
                       .format(CFG['model_arch'], CFG['test_fold'], epoch, val_acc, val_loss))

    # del model, optimizer, train_loader, val_loader, scaler, scheduler
    torch.cuda.empty_cache()

def main():
    seed_everything(2021)

    data = pd.read_csv(CFG['soft_labels_file'])
    print(data.shape)

    data['fold'] = 0
    strkf = StratifiedKFold(n_splits=5)
    _ = strkf.get_n_splits(data.image_id, data.label)
    f = 0
    for train_index, test_index in strkf.split(data.image_id, data.label):
        data.loc[data.index.isin(test_index), 'fold'] = f
        f = f + 1
    train_crossval(data)


if __name__ == "__main__":
    main()


