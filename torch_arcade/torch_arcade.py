
import os
import os.path
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

from PIL import Image

from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets import VisionDataset

class _ARCADEBase(VisionDataset):
    """ARCADE <https://zenodo.org/records/8386059> segmentation/stenosis detection dataset
        root (str or ``pathlib.Path``): Root directory of the ARCADE dataset.
        phase (string, optional): Which phase to use, supports ``"phase_1"``, ``"final_phase"``
    """

    URL="https://zenodo.org/records/8386059/files/arcade_challenge_datasets.zip"
    FILENAME="arcade_challenge_datasets.zip"

    def __init__(
        self,
        root: Union[str, Path],
        phase: str = "phase_1",
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)

        self.phase = phase
        if download:
            download_and_extract_archive(_ARCADEBase.URL, self.root, filename=_ARCADEBase.FILENAME)



    def __len__(self) -> int:
        return len(self.images)

class ARCADEBinarySegmentation():
    pass

class ARCADESemanticSegmentation():
    pass

class ARCADEStenosisDetection():
    pass


if __name__ == "__main__":
    _ARCADEBase(
        "dataset/",
        download="true"
    )
