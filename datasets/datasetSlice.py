import os
import sys
import h5py
import numpy as np
import torch.utils.data as data
import random
import math
import itertools

from scipy.spatial.distance import cdist
from tqdm import trange
from PIL import Image, ImageEnhance
#from scipy.misc import imresize
from scipy.ndimage import rotate
from skimage.morphology import skeletonize
from scipy.special import expit


class AxonDataset(data.Dataset):
    def __init__(self, opt, val=False, test=False):
        ''' Set up variables  '''
        self.logging = False
        self.test = test
        self.val = val
        self.dir = opt['path']
        self.z = int(opt['z'])
        self.crop_size = int(opt['training_crop_size'])
        self.test_size = int(opt['testing_crop_size'])
        self.log = int(opt['print_log'])
        self.transforms = opt['transforms']
        self.opt = opt
        self.fname = ""
        self.fname_idx = 0
        self.data = None
        self.truth = None
        self.classes = len(opt['permutations'])
        self.vol_sum = 0
        if self.test or self.val:
            opt['repeat_samples'] = 1

        ''' Choose all permutations above hamming distance '''
        self.permutations = opt['permutations'] 

        ''' Append subset (train, val, or test) to dir '''
        if self.test: self.dir = self.dir + "/test"
        elif self.val: self.dir = self.dir + "/val"
        else: self.dir = self.dir + "/train"

        ''' Read data filenames and associate indices  '''
        file_dict = {}
        self.volume_dict = {}
        self.length = 0
        for fname in os.listdir(self.dir):
            if fname[-3:] == ".h5":
                size = h5py.File(self.dir + "/" + fname)['data'].shape
                #size = np.transpose(np.array(h5py.File(self.dir + "/" + fname)['data'])).shape
                if self.test:
                    samples = math.ceil(size[0]/self.z) * math.ceil(size[1]/self.test_size) * math.ceil(size[2]/self.test_size)
                    slices = math.ceil(size[1]/self.test_size) * math.ceil(size[2]/self.test_size)
                else:
                    samples = opt['repeat_samples'] * math.ceil(size[0]/self.z) * math.ceil(size[1]/self.crop_size) * math.ceil(size[2]/self.crop_size)
                    slices = math.ceil(size[1]/self.crop_size) * math.ceil(size[2]/self.crop_size)
                self.length += samples
                file_dict[fname] = samples
                self.volume_dict[fname] = slices


        self.sample_dict = {}
        idx = 0
        for f, c in file_dict.items():
            for i in range(0, c, 1):
                self.sample_dict[idx + i] = f
            idx += c

        assert self.length == idx 


    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        ''' Read the data  '''
        fname = self.sample_dict[idx]

        ''' Monitor the index into the volume being accessed; read new volumes  '''
        if self.fname != fname:
            self.fname = fname
            self.fname_idx = 0
            self.data = np.array(h5py.File(self.dir + "/" + fname, 'r')[self.opt['dataset_name_raw']], dtype=np.float32)
            #self.data = np.transpose(self.data)
            self.vol_sum = np.sum(self.data)
            try:
                self.truth = np.array(h5py.File(self.dir + "/" + fname, 'r')[self.opt['dataset_name_truth']], dtype=np.float32)
                #self.truth = np.transpose(self.truth)
            except:
                ''' No labeled truth '''
                self.truth = None
                pass
        elif len(self.volume_dict) == 1 and self.fname_idx == len(self):
            self.fname_idx = 0

        truth_exists = False
        try:
            exists = self.truth == None
        except:
            truth_exists = True

        if self.val:
            ''' Convert data in NumPy arrays  '''
            image = self.data #np.array(self.data, dtype=np.float32)
            if truth_exists:
                mask = self.truth #np.array(self.truth, dtype=np.float32)
                mask[mask == 255] = 1
            else:
                mask = 0

            ''' Calculate indices into volume for sample  '''
            sample = int((self.fname_idx)/self.volume_dict[self.fname])
            remain = int((self.fname_idx)%self.volume_dict[self.fname])
            length = math.ceil(self.data.shape[1] / self.crop_size)
            x = int(remain % length) * self.crop_size
            y = int(remain / length) * self.crop_size
            if ((sample + 1) * self.z) > len(self.data):
                image = image[sample * self.z:len(self.data), x:x + self.crop_size, y:y + self.crop_size]
                if truth_exists:
                    mask = mask[sample * self.z:len(self.truth), x:x + self.crop_size, y:y + self.crop_size]
            else:
                image = image[sample * self.z:(sample + 1) * self.z, x:x + self.crop_size, y:y + self.crop_size]
                if truth_exists:
                    mask = mask[sample * self.z:(sample + 1) * self.z, x:x + self.crop_size, y:y + self.crop_size]

            #print("Z: {}-{}, X: {}-{}, Y: {}-{}, Shape: {},{},{}, Image: {},{},{}, Mask: {},{},{}, Min: {}, Max: {}.".format(sample*self.z, (sample+1)*self.z, x, x + self.crop_size, y, y + self.crop_size, self.data.shape[0], self.data.shape[1], self.data.shape[2], image.shape[0], image.shape[1], image.shape[2], mask.shape[0], mask.shape[1], mask.shape[2], np.min(image), np.max(image)))

        elif self.test:
            ''' Convert data in NumPy arrays  '''
            image = self.data #np.array(self.data, dtype=np.float32)

            ''' Calculate indices into volume for sample  '''
            sample = int((self.fname_idx)/self.volume_dict[self.fname])
            remain = int((self.fname_idx)%self.volume_dict[self.fname])
            length = math.ceil(self.data.shape[1] / self.test_size)
            x = int(remain % length) * self.test_size
            y = int(remain / length) * self.test_size
            if ((sample + 1) * self.z) > len(self.data):
                image = image[sample * self.z:len(self.data), x:x + self.test_size, y:y + self.test_size]
            else:
                image = image[sample * self.z:(sample + 1) * self.z, x:x + self.test_size, y:y + self.test_size]
            if self.log == 1:
                print("{}:\t Sample: [{}/{}],\t Image Sample: [{}/{}],\t Sub-sample: [{}/{}],\t Crop coordinates: [{}-{}, {}-{}, {}-{}],\t Output shape: {}".format(self.fname, self.fname_idx, len(self), sample+1, math.ceil(self.data.shape[0]/self.z),\
                    remain+1, self.volume_dict[self.fname], sample*self.z, (sample+1)*self.z, x, x+self.test_size, y, y+self.test_size, str(image.shape)))

            ''' Obtain test mask when provided in H5  '''
            try:
                mask = self.truth #np.array(self.truth, dtype=np.float32)
                if ((sample + 1) * self.z) > len(self.data):
                    mask = mask[sample * self.z:len(self.data), x:x + self.test_size, y:y + self.test_size]
                else:
                    mask = mask[sample * self.z:(sample + 1) * self.z, x:x + self.test_size, y:y + self.test_size]
                mask[mask == 255] = 1
            except:
                mask = 0

        else:
            # Old method - iterate over z
            #idx = int(idx/self.num_samples_per_train_slice)
            # New method - randomly slice in z
            idx = int(np.random.randint(0,high=(len(self.data)-self.z)))

            ''' Convert data in NumPy arrays '''
            image = self.data[idx:idx+self.z,:,:] #np.array(self.data[idx:idx + self.z,:,:], dtype=np.float32)
            if truth_exists:
                mask = self.truth[idx:idx+self.z,:,:] #np.array(self.truth[idx:idx + self.z,:,:], dtype=np.float32)
                mask[mask == 255] = 1

                ''' Randomly crop subvolume '''
                inp = np.stack([image, mask])
                out_max = 0
                while out_max == 0:
                    out = self.crop(inp)
                    out_max = np.max(out[0])
                out = self.transform_data(out)

                image = out[0]
                mask = out[1]
            else:
                ''' Randomly crop subvolume '''
                inp = np.stack([image])
                out_max = 0
                while out_max == 0:
                    out = self.crop(inp)
                    out_max = np.max(out[0])
                out = self.transform_data(out)

                image = out[0]
                mask = 0

        if self.logging:
            self.log_image(image[0], "raw", idx)
            if self.truth:
                self.log_image(mask[0], "mask", idx)

        ''' Normalize the image from 0 to 1 '''
        if np.max(image) != 0:
            image = np.array((image - np.min(image))/(np.max(image) - np.min(image)), dtype=np.float32)

        ''' Pad if dimensions are too small '''
        #if not self.test:
        z = self.z - image.shape[0]
        x = self.crop_size - image.shape[1]
        y = self.crop_size - image.shape[2]
        if x + y != 0 or (self.val and z != 0) or (self.test and z != 0 and z < 16):
            if x/2 % 1 == 0:
                x1 = int(x/2)
                x2 = int(x/2)
            else:
                x1 = int(x/2)
                x2 = math.ceil(x/2)
            if y/2 % 1 == 0:
                y1 = int(y/2)
                y2 = int(y/2)
            else:
                y1 = int(y/2)
                y2 = math.ceil(y/2)
            if not self.val and not self.test:
                z1 = 0
                z2 = 0
            elif z/2 % 1 == 0:
                z1 = int(z/2)
                z2 = math.ceil(z/2)
            else:
                z1 = int(z/2)
                z2 = math.ceil(z/2)
            image = np.pad(image, ((z1,z2),(x1,x2),(y1,y2)))
            if truth_exists:
                mask = np.pad(mask, ((z1,z2),(x1,x2),(y1,y2)))

        ''' Increment the index into the current volume  '''
        self.fname_idx += 1

        ''' Randomize the image and mask along the z axis according to permutation label '''
        label = random.randint(0, self.classes-1)
        permutation = self.permutations[label]
        image = image[permutation]
        if truth_exists:
            mask = mask[permutation] 

        return (image, label, mask)


    """ Other internal functions """
    def stitch(self, fname, blocks):
        single = []
        stitched = np.array([])
        samples = self.volume_dict[fname] #self.num_samples_per_slice
        length = math.ceil(self.data.shape[1] / self.test_size)
        for i, block in enumerate(blocks):
            #print("Block: " + str(block.shape))
            single.append(block)
            if (i+1) % samples == 0:
                rows = np.array([])
                r = np.array([])
                for j, b in enumerate(single):
                    #print("Columns: " + str(r.shape))
                    if r.size == 0:
                        r = b
                    else:
                        #print(r.shape)
                        #print(b.shape)
                        r = np.concatenate((r, b), axis=1)
                    if (j + 1) % length == 0:
                        if rows.size == 0:
                            rows = r
                        else:
                            rows = np.concatenate((rows, r), axis=2)
                        #print("Full: " + str(rows.shape))
                        r = np.array([])
                single = []
                if stitched.size == 0:
                    stitched = rows
                else:
                    stitched = np.concatenate((stitched, rows), axis=0)
                #print("Stitched: " + str(stitched.shape))
        return stitched


    def crop(self, images):
        i = 0
        if images[0].shape[1] <= self.crop_size:
            x_crop = [0, images[0].shape[1]]
        else:
            x_crop = int(np.random.randint(0,high=(images[i].shape[1]-self.crop_size)))
            x_crop = [x_crop,int(x_crop+self.crop_size)]
        if images[0].shape[2] <= self.crop_size:
            y_crop = [0, images[0].shape[2]]
        else:
            y_crop = int(np.random.randint(0,high=(images[i].shape[2]-self.crop_size)))
            y_crop = [y_crop,int(y_crop+self.crop_size)]
        new_images = []
        while i < len(images):
            image = images[i][:,x_crop[0]:x_crop[1],y_crop[0]:y_crop[1]]
            new_images.append(image)
            i += 1
        return np.array(new_images)


    def transform_data(self, images):
        rand = np.random.random((3,1))
        if "saturation" in self.transforms.keys() and rand[0] < self.transforms['saturation']:
            ''' Saturation '''
            enhance_factor = np.random.uniform(0.0,1.0)
            i = 0
            while i < len(images):
                enhancer = ImageEnhance.Color(Image.fromarray(images[i]))
                images[i] = np.array(enhancer.enhance(enhance_factor))
                i += 1
        if "rotation" in self.transforms.keys() and rand[1] < self.transforms['rotation']:
            ''' Rotation (with reflective padding) '''
            angle = random.randrange(0,360)
            i = 0
            while i < len(images):
                images[i] = rotate(images[i], angle, reshape=False, mode='reflect')
                i += 1
        if "crop" in self.transforms.keys() and rand[2] < self.transforms['crop']:
            '''crop and resize'''
            i = 0
            x_crop = int(np.random.randint(64,high=(images[i].shape[0]-64)))
            y_crop = int(np.random.randint(64,high=(images[i].shape[1]-64)))
            x_crop = [x_crop,int(x_crop+64)]
            y_crop = [y_crop,int(y_crop+64)]
            while i < len(images):
                size = (images[i].shape[0], images[i].shape[1])
                image = images[i][x_crop[0]:x_crop[1],y_crop[0]:y_crop[1]]
                images[i] = (imresize(image, size, interp='lanczos')/255)
                i += 1
        return images


    def log_image(self, image, name, idx):
        im = Image.fromarray(np.array(image * 255, dtype=np.uint8))
        im.save(name + str(idx).zfill(3) + ".png")
