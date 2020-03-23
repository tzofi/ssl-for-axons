import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2
from skimage.morphology import skeletonize
from PIL import Image
from scipy.ndimage import median_filter
from skimage import io


data = np.array(h5py.File(sys.argv[1], 'r')['dataset1'])[:]
print(data.shape)

clips = np.percentile(data, [0.01, 99.99])
data[data < clips[0]] = clips[0]
data[data > clips[1]] = clips[1]
data = median_filter(data, 3)
data = (data - np.min(data))/(np.max(data) - np.min(data))

#imageio.mimwrite("smi_raw.tif", np.array(data*255, dtype=np.uint8))
#imageio.mimwrite("smi_mask.tif", np.array(truth*255, dtype=np.uint8))
#imageio.mimwrite("smi_center.tif", np.array(center*255, dtype=np.uint8))

hf = h5py.File(sys.argv[2], "w")
hf.create_dataset('data', data=data)
hf.close()
