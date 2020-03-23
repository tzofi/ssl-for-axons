import os
import sys
import json
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision.utils import save_image
from PIL import Image
from collections import OrderedDict

# internal libraries
from models.model import SlicePredictor
from datasets.datasetSlice import AxonDataset

# Load config file
opt_file = open(sys.argv[1], "r")
opt = json.load(opt_file)
opt_file.close()

# Device configuration
device = torch.device('cuda')

# Select permutations (and save for future evaluation)
opt['permutations'] = sys.argv[2].split("/")[:-1]
opt['permutations'] = np.load('/'.join(opt['permutations']) + "/permutations.npy")

# Load model
model = SlicePredictor(1, len(opt['permutations']))
model = model.to(device)
model = nn.DataParallel(model)
model.load_state_dict(torch.load(sys.argv[2], map_location=lambda storage, loc: storage))

opt['transforms'] = {}
print(opt)

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
correct = 0
incorrect = 0
model.eval()
with torch.no_grad():
    for i, (image, label, mask) in enumerate(test_loader):
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
        prob = prob.cpu().detach().numpy()
        predicted = np.argmax(prob) 
        
        if predicted == label:
            correct += 1
        else:
            incorrect += 1

        if opt['save_images'] == 1:
            image = image.squeeze()
            for j, x in enumerate(prob[0][0]):
                im = Image.fromarray(np.array(x*255, dtype=np.uint8))
                im.save("cnn" + str(i+j).zfill(3) + ".png")
                im = Image.fromarray(np.array(image[j].cpu().detach().numpy()*255, dtype=np.uint8))
                im.save("raw" + str(i+j).zfill(3) + ".png")

        if opt['save_predictions'] == 1:
            prob[prob >= opt['metric_thresh']] = 1
            prob[prob < opt['metric_thresh']] = 0
            if dset in datasets:
                datasets[dset].append(np.array(prob.squeeze()))
            else:
                datasets[dset] = [np.array(prob.squeeze())]
            #predictions.append(np.array(prob.squeeze()))
        
        if opt['report_metrics'] == 1:
            if opt['save_predictions'] != 1:
                prob[prob >= opt['metric_thresh']] = 1
                prob[prob < opt['metric_thresh']] = 0
            prob = prob.flatten()
            true_prob = np.array(mask).flatten()
            #dices.append(dice_coeff(torch.from_numpy(prob), torch.from_numpy(true_prob)))

            ''' Save data for evaluation '''
            binary_preds.extend(prob)
            truth.extend(mask.flatten())

print("Correct: " + str(correct))
print("Incorrect: " + str(incorrect))
