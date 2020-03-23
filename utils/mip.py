import torch
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from PIL import Image

data = h5py.File(sys.argv[1], 'r')['unet']
print(data.shape)

data = np.array(data)
data[data >= 0.95] = 1.0
data[data < 0.95] = 0.0
data = np.array(data, np.uint8)
data_max = np.max(data, axis=0)

im = Image.fromarray(np.array(data_max*255, dtype=np.uint8))
im.save("mip.png")
