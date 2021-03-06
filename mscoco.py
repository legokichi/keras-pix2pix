
from typing import Tuple, List, Text, Dict, Any, Iterator, Union, Sized, Callable

import sys

sys.path.append("/usr/local/Cellar/opencv3/3.2.0/lib/python3.5/site-packages/") # mac opencv path
import cv2

import skimage.io as io
import numpy as np

from chainer.iterators import MultiprocessIterator, SerialIterator
from chainer.dataset.dataset_mixin import DatasetMixin

from imgaug import augmenters as iaa

sys.path.append("./coco/PythonAPI/")
from pycocotools.coco import COCO
from pycocotools import mask


class CamVid(DatasetMixin):
    def __init__(self, coco: COCO, path: str, seq: iaa.Sequential, resize_shape: Tuple[int, int]=None, dice_coef: bool=False):
        self.resize_shape = resize_shape
        self.coco = coco
        self.infos = coco.loadImgs(coco.getImgIds(catIds=coco.getCatIds(catNms=['person']))) # type: List[dict]
        self.seq = seq
        self.seq_norm = iaa.Sequential([
            iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)
        ]) # type: iaa.Sequential
        self.dice_coef = dice_coef
        self.path = path
    def __len__(self) -> int:
        return len(self.infos)
    def get_example(self, i) -> Tuple[np.ndarray, np.ndarray]:
        info = self.infos[i]
        img, mask = self.load_img(info)
        if self.seq != None:
            # image data augumantation
            img = np.expand_dims(img, axis=0)
            mask = np.expand_dims(mask, axis=0)
            img = self.seq.augment_images(img)
            img = self.seq_norm.augment_images(img)
            mask = self.seq.augment_images(mask)
            img = np.squeeze(img)
            mask = np.squeeze(mask)
        if self.resize_shape != None:
            img = cv2.resize(img, self.resize_shape)
            mask = cv2.resize(mask, self.resize_shape)
        if self.dice_coef:
            mask = mask > 0
        else:
            mask[:,:,0] = mask[:,:,0] > 0
        return (img, mask)
    def load_img(self, imgInfo: dict) -> Tuple[np.ndarray, np.ndarray]:
        #img = io.imread(imgInfo['coco_url'])
        img = io.imread(self.path + imgInfo['file_name'])
        if img.ndim != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[imgInfo['id']],iscrowd=False)) # type: List[dict]
        if self.dice_coef:
            mask_human = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        else:
            mask_human = np.zeros((img.shape[0], img.shape[1], 2), np.uint8)
        # mask_human: probability image mask
        for ann in anns:
            cat = self.coco.loadCats([ann["category_id"]])[0]
            if cat["name" ] != "person": continue
            rles = mask.frPyObjects(ann["segmentation"], img.shape[0], img.shape[1]) # type: List[dict]
            for i, rle in enumerate(rles):
                mask_img = mask.decode(rle) # type: np.ndarraya
                if self.dice_coef:
                    mask_human += mask_img
                else:
                    mask_human[:,:,0] += mask_img
        return (img, mask_human)



def get_iter(resize_shape: Tuple[int, int]=None, dice_coef: bool =False, workdir="./", data_aug: bool=False) -> DatasetMixin:

    coco_train = COCO(workdir+"annotations/instances_train2014.json") # type: COCO
    coco_val = COCO(workdir+"annotations/instances_val2014.json") # type: COCO

    if data_aug:
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                    translate_px={"x": (-16, 16), "y": (-16, 16)}, # translate by -16 to +16 pixels (per axis)
                    rotate=(-45, 45), # rotate by -45 to +45 degrees
                    shear=(-16, 16), # shear by -16 to +16 degrees
                    #order=iaa.ALL, # use any of scikit-image's interpolation methods
                    #cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    #mode="wrap" # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            ),
        ]).to_deterministic() # type: iaa.Sequential
    else:
        seq = iaa.Sequential([]).to_deterministic()


    return (
        CamVid(coco_train, workdir+"train2014/", seq, resize_shape, dice_coef),
        CamVid(coco_val, workdir+"val2014/", seq, resize_shape, dice_coef)
    )
