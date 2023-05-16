
import os
import torch
import cv2
import numpy as np
import pandas as pd
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform

from .cached_image_folder import CachedImageFolder
from .samplers import SubsetRandomSampler

try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR
except:
    from timm.data.transforms import _pil_interp

def collate_fn(data):
    datas=[torch.unsqueeze(item[0],0) for item in data]
    datas=torch.cat(datas,0)
    names=[item[1] for item in data]
    series=[item[2] for item in data]
    return datas,names,series

class SwinDataSet(Dataset):
    def normalize_simple(self, data: np.ndarray):
        data -= np.min(data)
        return data / np.max(data)

    def __init__(self,d,transform,config):
        imgs=os.listdir(d)
        self.data=[]
        self.names={}
        self.pics=[]
        self.series=[]
        for img in imgs:
            item=Image.open(d+'/'+img)
            if img[-4:]!='.png':
                continue
            if config.MODEL.SWIN.IN_CHANS==1:
                item=item.convert('L')
            else:
                item=item.convert('RGB')
            item=transform(item)

            item=item.reshape(-1,config.DATA.IMG_H,config.DATA.IMG_W)
            self.data.append(item)
            self.pics.append(img)
            self.names[img]=len(self.data)-1
            df=pd.read_csv(config.DATA.DATA_PATH+'/data/' + img.replace('png','csv'))
            df=self.normalize_simple(df[df.columns.values[1]])
            self.series.append(np.array(df))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        return self.data[idx],self.pics[idx],self.series[idx]


def build_loader(config):

    transform=build_transform(False, config)
    dataset_train = SwinDataSet(
        config.DATA.DATA_PATH+'/train/', transform, config)
    print(f"successfully build train dataset")
    dataset_val = SwinDataSet(
        config.DATA.DATA_PATH+'/val/', transform, config)
    print(f"successfully build val dataset")
    dataset_test = SwinDataSet(
        config.DATA.DATA_PATH+'/test/', transform, config)
    print(f"successfully build test dataset")
    
    data_loader_train = DataLoader(dataset_train,
                             batch_size=config.DATA.BATCH_SIZE,
                             num_workers=config.DATA.NUM_WORKERS,
                             pin_memory=config.DATA.PIN_MEMORY,
                             shuffle=True,
                             collate_fn=collate_fn)

    data_loader_val = DataLoader(dataset_val, batch_size=config.DATA.VAL_BATCH_SIZE,collate_fn=collate_fn,shuffle=True)
    data_loader_test = DataLoader(dataset_test, batch_size=config.DATA.VAL_BATCH_SIZE,collate_fn=collate_fn,shuffle=True)


    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.FEATURE_SIZE)

    return dataset_train, dataset_val,dataset_test, data_loader_train, data_loader_val,data_loader_test, mixup_fn


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif config.DATA.DATASET == 'imagenet22K':
        raise NotImplementedError("Imagenet-22K will come soon.")
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            #size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize((config.DATA.IMG_H, config.DATA.IMG_W), interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop((config.DATA.IMG_H, config.DATA.IMG_W)))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_H, config.DATA.IMG_W),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    return transforms.Compose(t)
