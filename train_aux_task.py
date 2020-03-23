import os
import sys
import json
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""Internal Libraries"""
from models.model import UNet3D, SlicePredictor 
from datasets.datasetSlice import AxonDataset
from utils.permutations import select_permutations

torch.backends.cudnn.enabled = True

# Load config file
opt_file = open(sys.argv[1], "r")
opt = json.load(opt_file)
opt_file.close()
opt['num_epochs']           = 1000
opt['learning_rate']        = 0.001
opt['batch_size']           = 216
opt['validate']             = True
opt['weights_path']         = 'weights/aux_task'
opt['parallel']             = True
opt['transforms']           = {"rotation":0.00}
opt['early_stop_criteria']  = 100
print(opt)

''' Verify weights directory exists, if not create it '''
if not os.path.isdir(opt['weights_path']):
    os.makedirs(opt['weights_path'])

# Select permutations (and save for future evaluation)
opt['permutations'] = select_permutations(opt['z'], opt['number_permutations'])
np.save(opt['weights_path'] + "/permutations", opt['permutations'])
#opt['permutations'] = np.load(opt['weights_path'] + "/permutations.npy")

# Device configuration
device = torch.device('cuda')

# Hyper-parameters
num_epochs = opt['num_epochs']
learning_rate = opt['learning_rate']

#model = UNet3D(1, 1, final_sigmoid=False)
model = SlicePredictor(1, len(opt['permutations']))
model = model.to(device)
model = nn.DataParallel(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# dataset
train_dataset = AxonDataset(opt)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=opt['batch_size'], 
                                           shuffle=True,
                                           pin_memory=True)

opt['repeat_samples'] = 1
val_dataset = AxonDataset(opt, val=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=1, 
                                           shuffle=False,
                                           pin_memory=True)


# Train the model
total_step = len(train_loader)
best_loss = np.inf
early_stop = 0
for epoch in range(num_epochs):
    epoch_loss = 0
    model.train()

    for i, (image, label, mask) in enumerate(train_loader):
        image = image.unsqueeze(1)
        image = image.to(device)
        label = label.to(device)

        # Forward pass
        prediction = model(image)
        loss = 1000 * criterion(prediction, label) * (torch.sum(image)/train_dataset.vol_sum)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}, Epoch Total Loss: {:.4f}"
               .format(epoch+1, num_epochs, i+1, total_step, loss.item()/opt['batch_size'], epoch_loss/((i+1)*opt['batch_size'])))
    early_stop += 1

    if opt['validate']:
        vloss = 0
        model.eval()
        with torch.no_grad():
            for j, (vimage, vlabel, vmask) in enumerate(val_loader):
                vimage = vimage.unsqueeze(1)
                vimage = vimage.to(device)
                vlabel = vlabel.to(device)

                # Forward pass
                vprediction = model(vimage)
                vloss += 1000 * criterion(vprediction, vlabel) * (torch.sum(image)/val_dataset.vol_sum)

        epoch_loss = vloss
        print("Validation Loss: {:.4f}".format(epoch_loss))
        
    if epoch_loss < best_loss:
        early_stop = 0
        best_loss = epoch_loss
        torch.save(model.state_dict(), opt['weights_path'] + "/" + str(epoch) + ".ckpt")

    if early_stop == opt['early_stop_criteria']:
        print("Validation loss has not improved - early stopping!")
        break

torch.save(model.state_dict(), opt['weights_path'] + "/final.ckpt")
