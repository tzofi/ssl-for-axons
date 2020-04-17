import os
import sys
import json
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sklearn.metrics import auc

torch.backends.cudnn.enabled=True

"""Internal Libraries"""
from models.model import UNet3D 
from datasets.dataset3D import AxonDataset

# Load config file
opt_file = open(sys.argv[1], "r")
opt = json.load(opt_file)
opt_file.close()
opt['num_epochs']           = 1000
opt['learning_rate']        = 0.001
opt['batch_size']           = 432
opt['weights_path']         = 'weights/unet'
opt['validate']             = True
opt['parallel']             = True
opt['transforms']           = {"rotation":0.25}
opt['early_stop_criteria']  = 100

# Device configuration
device = torch.device('cuda')

# Hyper-parameters
num_epochs = opt['num_epochs']
learning_rate = opt['learning_rate']
repeat = opt['repeat_samples']
print(opt)

''' Verify weights directory exists, if not create it '''
if not os.path.isdir(opt['weights_path']):
    os.makedirs(opt['weights_path'])

losses = []
f1_scores = []
aucs = []

REPS = 6
for REP in range(REPS):
    print("Running REP " + str(REP))
    print("\n")

    ''' Verify weights directory exists, if not create it '''
    if not os.path.isdir(opt['weights_path'] + "/" + str(REP)):
        os.makedirs(opt['weights_path'] + "/" + str(REP))


    model = UNet3D(1, 1)
    model = model.to(device)
    model = nn.DataParallel(model)
    print("Total parameters: " + str(sum(p.numel() for p in model.parameters())))
    print("Trainable parameters: " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # dataset
    opt['repeat_samples'] = repeat
    train_dataset = AxonDataset(opt)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=opt['batch_size'], 
                                               shuffle=True)

    opt['repeat_samples'] = 1
    val_dataset = AxonDataset(opt, val=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=1, 
                                               shuffle=False)


    # Train the model
    total_step = len(train_loader)
    best_loss = np.inf
    early_stop = 0
    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()

        for i, (image, mask) in enumerate(train_loader):
            image = image.unsqueeze(1)
            image = image.to(device)
            mask = mask.unsqueeze(1)
            mask = mask.to(device)

            # Forward pass
            prediction = model(image)
            loss = criterion(prediction, mask)
            
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
                for j, (vimage, vmask) in enumerate(val_loader):
                    vimage = vimage.unsqueeze(1)
                    vimage = vimage.to(device)
                    vmask = vmask.unsqueeze(1)
                    vmask = vmask.to(device)

                    # Forward pass
                    vprediction = model(vimage)
                    vloss += criterion(vprediction, vmask)

            epoch_loss = vloss/len(val_loader)
            print("Validation Loss: {:.4f}".format(epoch_loss))
            
        if epoch_loss < best_loss:
            early_stop = 0
            best_loss = epoch_loss
            torch.save(model.state_dict(), opt['weights_path'] + "/" + str(REP) + "/best.ckpt")

        if early_stop == opt['early_stop_criteria']:
            print("Validation loss has not improved - early stopping!")
            break

    torch.save(model.state_dict(), opt['weights_path'] + "/" + str(REP) + "/final.ckpt")

    ''' TESTING '''
    # Dataset
    opt['repeat_samples'] = 1
    test_dataset = AxonDataset(opt, val=False, test=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=1, 
                                               shuffle=False)

    model.load_state_dict(torch.load(opt['weights_path'] + "/" + str(REP) + "/best.ckpt", map_location=lambda storage, loc: storage))
    binary_preds = []
    truth = []
    model.eval()
    with torch.no_grad():
        for i, (vimage, vmask) in enumerate(test_loader):
            vimage = vimage.unsqueeze(1)
            vimage = vimage.to(device)

            prob = model(vimage)
            prob = prob.cpu().detach().numpy()
            binary_preds.extend(prob.flatten())
            truth.extend(vmask.flatten())
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
        #print([rec, prec, f1])
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        thresh += 0.05
    area = auc(recalls, precisions)
    print("Best F1 Score: " + str(np.max(f1s)))
    print("Area Under the Curve: " + str(area))
    f1_scores.append(np.max(f1s))
    aucs.append(area)
    losses.append(best_loss.item())

print("DONE. Here are some metrics:")
print("Losses: " + str(losses))
print("Top F1 Scores: " + str(f1_scores))
print("AUC Scores: " + str(aucs))
print("\n")
print("Std Val Loss: " + str(np.std(losses)))
print("Std F1: " + str(np.std(f1_scores)))
print("Std AUC: " + str(np.std(aucs)))
print("\n")
print("Average Val Loss: " + str(np.mean(losses)))
print("Average F1: " + str(np.mean(f1_scores)))
print("Average AUC: " + str(np.mean(aucs)))
