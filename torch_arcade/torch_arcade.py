
import os
import os.path
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np

from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets import VisionDataset

from encoding import ENCODING, COLOR_DICT

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
        self.images = [
            os.path.join(image_dir, file_name) 
            for file_name in os.listdir(image_dir) 
            if file_name.endswith('.png')
        ]
        self.file_to_id = {
            os.path.join(image_dir, image['file_name']) : image['id'] 
            for image in self.coco.dataset['images']
        }

    def __len__(self) -> int:
        return len(self.images)


def cached_mask(coco, cache_dir, 
                img_filename, img_id, 
                reduce):
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
        width, height = img.size
        mask = np.zeros((height, width), dtype=np.uint8)
        mask = None
        for ann in annotations:
            category = categories[ann["category_id"]]
            mask = reduce(mask, coco.annToMask(ann), category)
        # np.savez_compressed(mask_file, mask=mask)
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
        super().__init__(root, image_set, ARCADEBinarySegmentation.TASK, download, transform, target_transform, transforms)
        self.mask_dir = os.path.join(self.dataset_dir, ARCADEBinarySegmentation.MASK_CACHE)        
        os.makedirs(self.mask_dir, exist_ok=True)

    @staticmethod
    def reduction(mask, other, _):
        if mask is None: other
        return mask | other

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_filename = self.images[index]
        img = Image.open(img_filename)
        id = self.file_to_id[img_filename]
        mask =  cached_mask(self.coco, self.mask_dir, img_filename, id, ARCADEBinarySegmentation.reduction)
        return img, mask


class ARCADESemanticSegmentation(_ARCADEBase):
    TASK = "segmentation"
    MASK_CACHE = "masks_semantic"
    CAT_TO_ONEHOT = {}

    def __init__(
        self,
        root: Union[str, Path],
        image_set: str = "train",
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, image_set, ARCADESemanticSegmentation.TASK, download, transform, target_transform, transforms)
        self.mask_dir = os.path.join(self.dataset_dir, ARCADESemanticSegmentation.MASK_CACHE)                
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
        return img, mask


class ARCADEStenosisDetection():
    pass


def onehot_to_rgb(onehot, color_dict):
    single_layer = np.argmax(onehot, axis=-1)
    width, height, _ = onehot.shape
    output = np.zeros( (width, height, 3) )
    # print(single_layer)
    # # print(output[single_layer==1].shape)
    for k in range(len(color_dict)):
        output[single_layer==k] = np.array(color_dict[k])
    return np.uint8(output)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ds = ARCADESemanticSegmentation(
        "dataset/",
        image_set="train",
        download="true"
    )
    for img, mask in ds:  
        plt.imshow(img)
        plt.show()
        plt.imshow(onehot_to_rgb(mask, COLOR_DICT))
        plt.show()
