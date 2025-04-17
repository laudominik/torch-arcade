import numpy as np

def onehot(index, num_classes=27):
    vec = [np.uint8(0)] * num_classes
    vec[index] = np.uint8(1)
    return vec

ENCODING = {
    "background": onehot(0),
    "1": onehot(1),
    "2": onehot(2),
    "3": onehot(3),
    "4": onehot(4),
    "5": onehot(5),
    "6": onehot(6),
    "7": onehot(7),
    "8": onehot(8),
    "9": onehot(9),
    "9a": onehot(10),
    "10": onehot(11),
    "10a": onehot(12),
    "11": onehot(13),
    "12": onehot(14),
    "12a": onehot(15),
    "13": onehot(16),
    "14": onehot(17),
    "14a": onehot(18),
    "15": onehot(19),
    "16": onehot(20),
    "16a": onehot(21),
    "16b": onehot(22),
    "16c": onehot(23),
    "12b": onehot(24),
    "14b": onehot(25),
    "stenosis": onehot(26)
}

COLOR_DICT = np.array([
    [0, 0, 0],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [0, 255, 255],
    [255, 0, 255],
    [192, 192, 192],
    [128, 128, 128],
    [128, 0, 0],
    [128, 128, 0],
    [0, 128, 0],
    [0, 0, 128],
    [0, 128, 128],
    [128, 0, 128],
    [255, 165, 0],
    [255, 105, 180],
    [255, 69, 0],
    [60, 179, 113],
    [255, 215, 0],
    [138, 43, 226],
    [255, 105, 180],
    [255, 20, 147],
    [184, 134, 11],
    [255, 140, 0],
    [0, 206, 209],
    [70, 130, 180],
    [255, 215, 0]
])
