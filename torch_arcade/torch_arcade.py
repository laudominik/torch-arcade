import os
import os.path
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import torch

from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets import VisionDataset

from .encoding import ENCODING, COLOR_DICT

def distinguish_side(segments):
        return "right" if any(seg in segments for seg in ["1", "2", "3", "4", "16a", "16b", "16c"]) else "left"


class _ARCADEBase(VisionDataset):
    URL="https://zenodo.org/records/8386059/files/arcade_challenge_datasets.zip"
    ZIPNAME="arcade_challenge_datasets.zip"
    FILENAME="arcade_challenge_datasets"
    DATASET_DICT = {
    "segmentation" : {
        "train": {
            "path": os.path.join("dataset_phase_1", "segmentation_dataset", "seg_train"),
            "coco": "seg_train.json",
        },
        "val": {
            "path": os.path.join("dataset_phase_1", "segmentation_dataset", "seg_val"),
            "coco": "seg_val.json"
        },
        "test": {
            "path": os.path.join("dataset_final_phase", "test_case_segmentation"),
            "coco": "instances_default.json"
        }
    },
    "stenosis" : {
        "train": {
            "path": os.path.join("dataset_phase_1", "stenosis_dataset", "sten_train"),
            "coco": "sten_train.json"
        },
        "val": {
            "path": os.path.join("dataset_phase_1", "stenosis_dataset", "sten_val"),
            "coco": "sten_val.json"
        },
        "test": {
            "path": os.path.join("dataset_final_phase", "test_cases_stenosis"),
            "coco": "instances_default.json"
        }
    },
}

    def __init__(
        self,
        root: Union[str, Path],
        image_set: str = "train",
        task: str = "segmentation",
        side: str = None,
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)

        if download:
            download_and_extract_archive(_ARCADEBase.URL, self.root, filename=_ARCADEBase.ZIPNAME)

        task_dict = _ARCADEBase.DATASET_DICT[task][image_set]
        self.dataset_dir = os.path.join(self.root,  _ARCADEBase.FILENAME, task_dict["path"])
        self.coco = COCO(os.path.join(self.dataset_dir, "annotations", task_dict["coco"]))
        image_dir = os.path.join(self.dataset_dir, "images")

        self.images = []

        for image in self.coco.dataset['images']:
            img_id = image['id']
            annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            segments = {self.coco.cats[ann["category_id"]]["name"] for ann in annotations}
            this_side = distinguish_side(segments)
            if side is None or this_side == side:
                file_path = os.path.join(image_dir, image['file_name'])
                if os.path.exists(file_path) and file_path.endswith('.png'):
                    self.images.append(file_path)

        self.file_to_id = {
            os.path.join(image_dir, image['file_name']) : image['id'] 
            for image in self.coco.dataset['images']
        }

    def __len__(self) -> int:
        return len(self.images)


def cached_mask(coco, cache_dir, 
                img_filename, img_id, 
                reduce, bg=True):
    img = Image.open(img_filename)
    mask_file = os.path.join(
        cache_dir,
        f"{os.path.basename(img_filename).replace('.png', '.npz')}"
    )
    categories = coco.loadCats(coco.getCatIds())

    if os.path.exists(mask_file):
        mask = np.load(mask_file)["mask"]
    else:
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        mask = None
        for ann in annotations:
            category = categories[ann["category_id"]-1]
            mask = reduce(mask, coco.annToMask(ann), category)    
        if bg:
            bg_channel = (np.sum(mask, axis=-1) == 0).astype(np.uint8)
            bg_channel = bg_channel[..., np.newaxis]
            mask = np.concatenate([bg_channel, mask], axis=-1)
        np.savez_compressed(mask_file, mask=mask)
    return mask


class ARCADEBinarySegmentation(_ARCADEBase):
    TASK = "segmentation"
    MASK_CACHE = "masks"

    def __init__(
        self,
        root: Union[str, Path],
        image_set: str = "train",
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, image_set, ARCADEBinarySegmentation.TASK, None, download, transform, target_transform, transforms)
        self.mask_dir = os.path.join(self.dataset_dir, ARCADEBinarySegmentation.MASK_CACHE)        
        os.makedirs(self.mask_dir, exist_ok=True)

    @staticmethod
    def reduction(mask, other, _):
        if mask is None: return other
        return mask | other

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_filename = self.images[index]
        img = Image.open(img_filename)
        id = self.file_to_id[img_filename]
        mask =  cached_mask(self.coco, self.mask_dir, img_filename, id, ARCADEBinarySegmentation.reduction, False)
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        return img, mask


class ARCADESemanticSegmentation(_ARCADEBase):
    TASK = "segmentation"
    MASK_CACHE = "masks_semantic"
    CAT_TO_ONEHOT = {}

    def __init__(
        self,
        root: Union[str, Path],
        image_set: str = "train",
        side: str = None,
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        task: str = TASK,
        cache: str = MASK_CACHE
    ):
        super().__init__(root, image_set, task, side, download, transform, target_transform, transforms)
        self.mask_dir = os.path.join(self.dataset_dir, cache)                
        os.makedirs(self.mask_dir, exist_ok=True)

    @staticmethod
    def reduction(mask, other, other_cat):
        one_hot_vector = np.array(ENCODING[other_cat['name']])
        width, height = other.shape
        other_oh = np.zeros((width, height, len(one_hot_vector)))
        other_oh[(other == 1)] = one_hot_vector

        if mask is None: return other_oh
        return np.maximum(mask, other_oh)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_filename = self.images[index]
        img = Image.open(img_filename)
        id = self.file_to_id[img_filename]
        mask =  cached_mask(self.coco, self.mask_dir, img_filename, id, ARCADESemanticSegmentation.reduction)
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        return img, mask


class ARCADEStenosisSegmentation(ARCADESemanticSegmentation):
    TASK = "stenosis"
    MASK_CACHE = "masks_stenosis"
    STENOSIS_IN_MASK_INDEX = 25
    
    def __init__(
        self,
        root: Union[str, Path],
        image_set: str = "train",
        side: str = None,
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, image_set, side, download, transform, target_transform, transforms, ARCADEStenosisSegmentation.TASK, ARCADEStenosisSegmentation.MASK_CACHE)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, mask = super().__getitem__(index)
        selected_segments = [0, ARCADEStenosisSegmentation.STENOSIS_IN_MASK_INDEX]
        return img, mask[selected_segments, :, :]


class ARCADEStenosisDetection(_ARCADEBase):
    TASK = "stenosis"

    def __init__(
        self,
        root: Union[str, Path],
        image_set: str = "train",
        side: str = None,
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, image_set, ARCADEStenosisDetection.TASK, side, download, transform, target_transform, transforms)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_filename = self.images[index]
        img_id = self.file_to_id[img_filename]
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        img = Image.open(img_filename)
        
        if self.transforms is not None:
            img, annotations = self.transforms(img, annotations)

        return img, annotations


class ARCADESemanticSegmentationBinary(_ARCADEBase):
    TASK = "segmentation"
    MASK_CACHE = "masks_semantic"
    CAT_TO_ONEHOT = {}

    def __init__(
        self,
        root: Union[str, Path],
        image_set: str = "train",
        side: str = None,
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, image_set, ARCADESemanticSegmentationBinary.TASK, side, download, transform, target_transform, transforms)
        self.mask_dir = os.path.join(self.dataset_dir, ARCADESemanticSegmentationBinary.MASK_CACHE)                
        os.makedirs(self.mask_dir, exist_ok=True)

    @staticmethod
    def reduction(mask, other, other_cat):
        one_hot_vector = np.array(ENCODING[other_cat['name']])
        width, height = other.shape
        other_oh = np.zeros((width, height, len(one_hot_vector)))
        other_oh[(other == 1)] = one_hot_vector

        if mask is None: return other_oh
        return np.maximum(mask, other_oh)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_filename = self.images[index]
        img = Image.open(img_filename)
        id = self.file_to_id[img_filename]
        mask =  cached_mask(self.coco, self.mask_dir, img_filename, id, ARCADESemanticSegmentationBinary.reduction)
        binary = (mask[:,:, 1:].sum(axis=-1, keepdims=True) > 0)
        if self.transforms is not None:
            binary, mask = self.transforms(binary, mask)
        return binary, mask


class ARCADEArteryClassification(_ARCADEBase):
    TASK = "segmentation"
    MASK_CACHE = "masks"
    ID2LABEL = {
        0: "right",
        1: "left",
    }

    def __init__(
        self,
        root: Union[str, Path],
        image_set: str = "train",
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, image_set, ARCADEArteryClassification.TASK, None, download, transform, target_transform, transforms)
        self.mask_dir = os.path.join(self.dataset_dir, ARCADEArteryClassification.MASK_CACHE)
        os.makedirs(self.mask_dir, exist_ok=True)

    @staticmethod
    def reduction(mask, other, _):
        if mask is None:
            return other
        return mask | other

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_filename = self.images[index]
        id = self.file_to_id[img_filename]
        mask = Image.fromarray(cached_mask(self.coco, self.mask_dir, img_filename, id, ARCADEArteryClassification.reduction))

        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=id))
        segments = {self.coco.cats[ann["category_id"]]["name"] for ann in annotations}

        label = 0 if distinguish_side(segments) == "right" else "left"

        if self.transforms is not None:
            mask, label = self.transforms(mask, label)
            
        return mask, label


def onehot_to_rgb(onehot, color_dict):
    single_layer = np.argmax(onehot, axis=-1)
    width, height, _ = onehot.shape
    output = np.zeros( (width, height, 3) )
    for k in range(len(color_dict)):
        output[single_layer==k] = np.array(color_dict[k])
    return np.uint8(output)


def visualize_artery_classification():
    import matplotlib.pyplot as plt

    dataset = ARCADEArteryClassification(
        "datasets/",
        image_set="train",
        download="true"
    )

    plt.figure(figsize=(15, 10))
    n_samples = 16
    for i, (mask, label) in enumerate(dataset):
        if i >= n_samples:
            break
        
        plt.subplot(4, 4, i + 1)
        plt.imshow(mask)
        plt.title(f"GT {i+1} Label: {dataset.ID2LABEL[label]}")
        plt.axis('off')
        
    plt.savefig("arcade_artery_side_classification.png")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = ARCADESemanticSegmentation(
        "datasets/",
        image_set="train",
        download="true"
    )

    plt.figure(figsize=(15, 10))
    n_samples = 8
    for i, (img, mask) in enumerate(dataset):
        if i >= n_samples:
            break
        
        rgb_mask = onehot_to_rgb(mask, COLOR_DICT)
        
        plt.subplot(4, 4, 2*i + 1)
        plt.imshow(img)
        plt.title(f"Input {i+1}")
        plt.axis('off')
        
        plt.subplot(4, 4, 2*i + 2)
        plt.imshow(rgb_mask)
        plt.title(f"GT {i+1}")
        plt.axis('off')
    plt.show()
