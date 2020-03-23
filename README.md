# Self-Supervised Feature Extraction for 3D Axon Segmentation 

#### Install Our Models

The code has been developed and tested with PyTorch 1.2.0 in a conda environment

```bash
conda activate {Project Environment}
git clone ${repository}
cd ssl-for-axons
conda install --yes --file requirements.txt
```

#### Data

For part of our work, we used the [Janelia dataset from the BigNeuron Project](https://github.com/BigNeuron/Data/releases/tag/Gold166_v1).

#### Training Our Models
We propose first training the 3D U-Net and auxiliary classifier on an auxiliary task. The auxiliary task consists of reordering the slices of each training subvolume and then training the CNN to predict the permutation used to reorder the slices. For example, if there 10 permutations have been generated, then the auxiliary classifier should predict a one-hot encoding of length 10, where argmax of the encoding is the index to the permutation used. Code for sampling permutations is in utils/permutations.py. After training the auxiliary task, the 3D U-Net pre-trained encoder and randomly initialized decoder can be fine tuned on the target segmentation task.

##### To Train

```bash
# 3D U-Net
python train_unet.py configs/train.json

# Auxiliary Task
python train_aux_task.py configs/train_slices.json

# Fine Tuning 3D U-Net with pre-trained encoder 
python transfer_train_unet.py configs/train.json weights/aux_task/best.ckpt
```

All train configuration JSONs can be found in configs/train and modified as needed. Configuration JSON files are available for training a separate airlight model, a separate transmission map model, a DualFastNet model, a FastNet model, or a FastNet50 model. Each configuration JSON file is available for both the 2019 NTIRE Image Dehazing dataset or the He, Zhang dataset. Validation can also be done during training by setting the "validate" flag to "1" in the JSON configuration file and providing a path to the validation JSON configuration file in the "validation\_config" field, as shown below:

All training parameters can be set in the JSON config and/or top of the py file. For example, you should always indicate the location of your training files (we natively support training with data stored in .h5 datasets) in the config:

```bash
"path": "./data/janelia,
"dataset_name_raw": "data",
"dataset_name_truth": "truth",
```

#### Testing Our Models 
As in training, testing can be done using a JSON configuration file. These are located in configs and can be modified as needed. We include support for saving predictions back to an h5 or tiff, as well as reporting voxel-based metrics including precision, recall, AUC, and F1 scores.

```bash
# To test 3D U-Net
python test.py configs/test.json {path_to_weights_file}

# To test auxiliary classifier
python test_aux_task.py configs/test_slices.json {path_to_weights_file}
``` 

#### References
Our implementation draws heavily upon [this 3D U-Net code](https://github.com/wolny/pytorch-3dunet) by Wolny et al. If using this code, please also cite his work.
