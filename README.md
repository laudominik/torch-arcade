# ARCADE dataset for PyTorch

![](demo.png)

This package contains a PyTorch dataset for [ARCADE](https://zenodo.org/records/8386059) from
"Dataset for Automatic Region-based Coronary Artery Disease Diagnostics Using X-Ray Angiography Images". The dataset can be used for binary and semantic segmentation as well as stenosis detection.

## Installation

```bash
pip3 install git+https://github.com/laudominik/torch-arcade
```

## Usage example

```python
from torch_arcade import ARCADEBinarySegmentation

ds = ARCADEBinarySegmentation(
    "dataset/",
    image_set="train",
    download="true"
)
img, mask = ds[0]
```
