
import os
import os.path
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

from PIL import Image

from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets import VisionDataset

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


class _ARCADEBase(VisionDataset):
    """`ARCADE <https://zenodo.org/records/8386059>`_ segmentation/stenosis detection dataset
        root (str or ``pathlib.Path``): Root directory of the ARCADE dataset.
        phase (string, optional): Which phase to use, supports ``"phase_1"``, ``"final_phase"``
        image_set (string, optional): One of ``"train"``, ``"val"``, ``"test"``
        
        note: requires pycocoutils for target loading
    """

    URL="https://zenodo.org/records/8386059/files/arcade_challenge_datasets.zip"
    ZIPNAME="arcade_challenge_datasets.zip"
    FILENAME="arcade_challenge_datasets"

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

        task_dict = DATASET_DICT[task][image_set]
        self.dataset_dir = os.path.join(self.root,  _ARCADEBase.FILENAME, task_dict["path"])
        self.coco = os.path.join(self.dataset_dir, "annotations", task_dict["coco"])
        
        image_dir = os.path.join(self.dataset_dir, "images")
        self.images = [
            os.path.join(image_dir, file_name) 
            for file_name in os.listdir(image_dir) 
            if file_name.endswith('.png')
        ]

    def __len__(self) -> int:
        return len(self.images)

class ARCADEBinarySegmentation():
    pass

class ARCADESemanticSegmentation():
    pass

class ARCADEStenosisDetection():
    pass


if __name__ == "__main__":
    ds = _ARCADEBase(
        "dataset/",
        image_set="train",
        download="true"
    )
    print(len(ds))
