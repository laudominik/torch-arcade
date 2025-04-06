
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

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img_filename = self.images[index]
        img = Image.open(img_filename)
        id = self.file_to_id[img_filename]

        mask_file = os.path.join(
            self.mask_dir,
            f"{os.path.basename(img_filename).replace('.png', '.npz')}"
        )

        mask = None
        if os.path.exists(mask_file):
            mask = np.load(mask_file)["mask"]
        else:
            annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=id))
            width, height = img.size
            mask = np.zeros((height, width), dtype=np.uint8)
            mask = self.coco.annToMask(annotations[0])
            for ann in annotations:
                mask |= self.coco.annToMask(ann)
            np.savez_compressed(mask_file, mask=mask)
        return img, mask

class ARCADESemanticSegmentation():
    pass

class ARCADEStenosisDetection():
    pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ds = ARCADEBinarySegmentation(
        "dataset/",
        image_set="train",
        download="true"
    )
    img, mask = ds[0]
    plt.imshow(img)
    plt.show()
    plt.imshow(mask)
    plt.show()
