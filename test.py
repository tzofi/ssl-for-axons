import os
import sys
import json
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import imageio

from torchvision.utils import save_image
from PIL import Image
from collections import OrderedDict
from sklearn.metrics import auc

# internal libraries
from models.model import UNet3D
from datasets.dataset3DTest import AxonDataset

# Load config file
opt_file = open(sys.argv[1], "r")
opt = json.load(opt_file)
opt_file.close()

# Device configuration
device = torch.device('cuda')

# Load model
model = UNet3D(1, 1, final_sigmoid=True).to(device)
model = nn.DataParallel(model)
model.load_state_dict(torch.load(sys.argv[2], map_location=lambda storage, loc: storage))

opt['transforms'] = {}

# Dataset
test_dataset = AxonDataset(opt, val=False, test=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=1, 
                                           shuffle=False)


# Test the model
error = 0
total = 0
predictions = []
binary_preds = []
truth = []
data = []
dices = []
datasets = {}
changes = 0
model.eval()
with torch.no_grad():
    for i, (image, mask, overlap) in enumerate(test_loader):
        if i == 0: fname = test_dataset.fname
        elif test_dataset.fname != fname:
            changes += 1
            fname = test_dataset.fname

        dset = test_dataset.dir + "/" + test_dataset.fname
        if image.squeeze().shape[0] == 4:
            break
        image = image.unsqueeze(1)
        image = image.to(device).float()

        prob = model(image)

        prob = prob.squeeze()
        if mask != 0:
            mask = mask.squeeze()
        if sum(overlap) != 0:
            prob = prob[overlap[0]:, overlap[1]:, overlap[2]:]
            if mask != 0:
                mask = mask[overlap[0]:, overlap[1]:, overlap[2]:]
        prob = prob.cpu().detach().numpy()

        if opt['save_images'] == 1:
            image = image.squeeze()
            for j, x in enumerate(prob[0][0]):
                im = Image.fromarray(np.array(x*255, dtype=np.uint8))
                im.save("cnn" + str(i+j).zfill(3) + ".png")
                im = Image.fromarray(np.array(image[j].cpu().detach().numpy()*255, dtype=np.uint8))
                im.save("raw" + str(i+j).zfill(3) + ".png")
        
        if opt['report_metrics'] == 1:
            prob = prob.flatten()
            true_prob = np.array(mask).flatten()
            binary_preds.extend(prob)
            truth.extend(mask.flatten())

        if opt['save_predictions'] == 1:
            prob[prob >= opt['metric_thresh']] = 1
            prob[prob < opt['metric_thresh']] = 0
            if dset in datasets:
                datasets[dset].append(np.array(prob.squeeze()))
            else:
                datasets[dset] = [np.array(prob.squeeze())]


if opt['report_metrics'] == 1:
    binary = np.array(binary_preds).flatten()
    truth = np.array(truth)
    p = np.sum(truth)

    precisions = []
    recalls = []
    f1s = []
    thresh = 0
    while thresh <= 1:
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for i, d in enumerate(binary):
            t = truth[i]
            if t == 1 and d >= thresh:
                tp += 1.0
            elif t == 1 and d < thresh:
                fn += 1.0
            elif t == 0 and d >= thresh:
                fp += 1.0
            elif t == 0 and d < thresh:
                tn += 1.0

        try:
            prec = tp/(tp+fp)
        except ZeroDivisionError:
            prec = 0
        try:
            rec = tp/(tp+fn)
        except ZeroDivisionError:
            rec = 0
        try:
            f1 = 2 * (prec * rec) / (prec + rec)
        except ZeroDivisionError:
            f1 = 0
        print([rec, prec, f1])
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        thresh += 0.05
    area = auc(recalls, precisions)
    print("Precision: " + str(precisions))
    print("Recall: " + str(recalls))
    print("Best F1 Score: " + str(np.max(f1s)))
    print("Area Under the Curve: " + str(area))
    

if opt['save_predictions'] == 1:
    print("Saving predictions:")
    for i, (filename, predictions) in enumerate(datasets.items()):
        fname = filename.split("/")[-1]
        stitched = test_dataset.stitch(fname, predictions)

        hf = h5py.File("PVG_Full_Prediction.h5", "w")
        hf.create_dataset(opt['output_dataset_name'], data=np.array(stitched))
        hf.close()
        #data = np.array(stitched*255, dtype=np.uint8)
        #imageio.mimwrite("unet_prediction.tif", data)
