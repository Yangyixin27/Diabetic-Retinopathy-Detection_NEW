import json
import os
import time
import warnings

import SimpleITK as sitk
import numpy as np
from sklearn.utils import shuffle as sklean_shuffle
from project import utils
from project.settings import intermediate_dir


class NeedleData(object):
    def __init__(self, padding_axis1=0, padding_axis2=0):
        self.numpy_dir = os.path.join(intermediate_dir, "numpy")
        self.numpy_preprocessed_dir = os.path.join(intermediate_dir, "numpy_preprocessed")
        self.SPLIT = 5
        self.padding_axis1 = padding_axis1
        self.padding_axis2 = padding_axis2
        
    
    def balance(self, volumes, labelmaps, balance_ratio, fliplr):
        needle_idx = []
        nonneedle_idx = []
        needle = 0
        for i,_ in enumerate(labelmaps): 
            if 1 in labelmaps[i]:
                needle_idx.append(i)
                needle += 1
            else: 
                nonneedle_idx.append(i)
        print("Number of Slicers Containing Needle: {}".format(needle))
        nonneedle = needle*balance_ratio
        nonneedle_idx = nonneedle_idx[:nonneedle]
        idx = needle_idx + nonneedle_idx
        volumes_balance = volumes[idx]
        labelmaps_balance = labelmaps[idx]
        print("Volumes Balance Shape: ", volumes_balance.shape)
        print("Labelmaps Balance Shape: ", labelmaps_balance.shape)
        
        if fliplr == True:
            needle_flipping_volumes = np.flip(volumes[needle_idx], axis=3)
            needle_flipping_labelmaps = np.flip(labelmaps[needle_idx], axis=3)
            print("Needle Flipping Volumes Shape: ", needle_flipping_volumes.shape)
            print("Needle Flipping Labelmaps Shape: ", needle_flipping_labelmaps.shape)
            volumes_balance = np.concatenate((volumes_balance, needle_flipping_volumes), axis=0)
            labelmaps_balance = np.concatenate((labelmaps_balance, needle_flipping_labelmaps), axis=0)
            print("Volumes Balance Shape After Concatenation: ", volumes_balance.shape)
            print("Labelmaps Balance Shape After Concatenation: ", labelmaps_balance.shape)
            
        return volumes_balance, labelmaps_balance
        
        
    def data_preprocess(self, arrays, mean, std):
        arrays = arrays.astype("float32")
        arrays = (arrays - mean)/std
        return arrays
    
    
    def label_preprocess(self, arrays):
        arrays = arrays.astype("float32")
        for i in range(arrays.shape[0]):
            arrays[i] = arrays[i]/np.amax(arrays[i])
        return arrays
        
        
    def slice_and_padding(self, volume_lists, label_lists, padding_axis1, padding_axis2, shuffle_slices=True):
        rows = volume_lists.shape[1]
        cols = volume_lists.shape[2]
        slices_arr = np.empty((0, rows, cols), dtype=np.float32)
        slices_label_arr = np.empty((0, rows, cols), dtype=np.float32)
        i = 0
        for volume_list, label_list in zip(volume_lists, label_lists):
            if i == 0:
                print(volume_list.shape)
            volume_rolled_arrs = np.rollaxis(volume_list, 2, 0)
            if i == 0:
                print(volume_rolled_arrs.shape)
            label_rolled_arrs = np.rollaxis(label_list, 2, 0)
            assert volume_rolled_arrs.shape == label_rolled_arrs.shape
            
            slices_arr = np.append(slices_arr, volume_rolled_arrs, axis=0)
            slices_label_arr = np.append(slices_label_arr, label_rolled_arrs, axis=0)
                
            if i % 5 == 0:
                print('Done: {0}/{1} volumes'.format(i, len(volume_lists)))
            i += 1
                
        slices_arr = np.lib.pad(slices_arr, ((0,0),(padding_axis1//2,padding_axis1//2),\
                                             (padding_axis2//2,padding_axis2//2)), 'constant', constant_values=(0, 0))

        print('Loading done.')

        if shuffle_slices:
            slices_arr, slices_label_arr = sklean_shuffle(slices_arr, slices_label_arr)

        slices_arr = np.expand_dims(slices_arr, axis=1)
        slices_label_arr = np.expand_dims(slices_label_arr, axis=1)
        return slices_arr, slices_label_arr
    
    
    def preprocess_train_validation_data_per_split(self, split_num):
        train_slices_arrs = np.load(os.path.join(self.numpy_dir, "{}".format(split_num), 'training_data.npy'))
        train_slices_label_arrs = np.load(os.path.join(self.numpy_dir, "{}".format(split_num), 'training_labelmap.npy'))
        val_slices_arrs = np.load(os.path.join(self.numpy_dir, "{}".format(split_num), 'validation_data.npy'))
        val_slices_label_arrs = np.load(os.path.join(self.numpy_dir, "{}".format(split_num), 'validation_labelmap.npy'))
        mean = np.load(os.path.join(self.numpy_dir, "{}".format(split_num), 'training_mean.npy'))
        std = np.load(os.path.join(self.numpy_dir, "{}".format(split_num), 'training_std.npy'))
        
        train_slices_arrs = self.data_preprocess(train_slices_arrs, mean, std)
        val_slices_arrs = self.data_preprocess(val_slices_arrs, mean, std)
        train_slices_label_arrs = self.label_preprocess(train_slices_label_arrs)
        val_slices_label_arrs = self.label_preprocess(val_slices_label_arrs)
        
        assert train_slices_arrs.shape[0] == train_slices_label_arrs.shape[0] 
        assert val_slices_arrs.shape[0] == val_slices_label_arrs.shape[0]
        
        train_vol_arr, train_label_arr = self.slice_and_padding(train_slices_arrs, train_slices_label_arrs, \
                                             self.padding_axis1, self.padding_axis2, shuffle_slices=True)
        val_vol_arr, val_label_arr = self.slice_and_padding(val_slices_arrs, val_slices_label_arrs, \
                                          self.padding_axis1, self.padding_axis2, shuffle_slices=True)
        
        print(train_vol_arr.shape)
        print(train_label_arr.shape)
        print(val_vol_arr.shape)
        print(val_label_arr.shape)
        
        return train_vol_arr, train_label_arr, val_vol_arr, val_label_arr

    
    def save_train_validation_data(self):
        if not os.path.exists(self.numpy_preprocessed_dir):
            os.mkdir(self.numpy_preprocessed_dir)
        for i in range(self.SPLIT):
            i = i + 1
            print("Round: {}".format(i))
            train_vol_arr, train_label_arr, val_vol_arr, val_label_arr = self.preprocess_train_validation_data_per_split(i)
            if not os.path.exists(os.path.join(self.numpy_preprocessed_dir, "{}".format(i))):
                os.mkdir(os.path.join(self.numpy_preprocessed_dir, "{}".format(i)))
            np.save(os.path.join(self.numpy_preprocessed_dir, "{}".format(i), "training_data.npy"), train_vol_arr)
            np.save(os.path.join(self.numpy_preprocessed_dir, "{}".format(i), "training_labelmap.npy"), train_label_arr)
            np.save(os.path.join(self.numpy_preprocessed_dir, "{}".format(i), "validation_data.npy"), val_vol_arr)
            np.save(os.path.join(self.numpy_preprocessed_dir, "{}".format(i), "validation_labelmap.npy"), val_label_arr)
    
    
    def load_train_validation_data(self, split_num, balance = False, balance_ratio = 0, fliplr = False):
        train_slices_arrs = np.load(os.path.join(self.numpy_preprocessed_dir, "{}".format(split_num), 'training_data.npy'))
        train_slices_label_arrs = np.load(os.path.join(self.numpy_preprocessed_dir, "{}".format(split_num), 'training_labelmap.npy'))
        val_slices_arrs = np.load(os.path.join(self.numpy_preprocessed_dir, "{}".format(split_num), 'validation_data.npy'))
        val_slices_label_arrs = np.load(os.path.join(self.numpy_preprocessed_dir, "{}".format(split_num), 'validation_labelmap.npy'))
        if balance == True:
            train_slices_arrs, train_slices_label_arrs = self.balance(train_slices_arrs, train_slices_label_arrs, balance_ratio, fliplr)
        return train_slices_arrs, train_slices_label_arrs, val_slices_arrs, val_slices_label_arrs

    
    def load_test_data(self, split_num):
        test_vol_arrs = np.load(os.path.join(self.numpy_dir, 'testing_data.npy'))
        test_vol_label_arrs = np.load(os.path.join(self.numpy_dir, 'testing_labelmap.npy'))
        mean = np.load(os.path.join(self.numpy_dir, "{}".format(split_num), 'training_mean.npy'))
        std = np.load(os.path.join(self.numpy_dir, "{}".format(split_num), 'training_std.npy'))
        
        test_vol_arrs = self.data_preprocess(test_vol_arrs, mean, std)
        test_vol_label_arrs = self.label_preprocess(test_vol_label_arrs)
        
        test_slices_arrs, test_slices_label_arrs = self.slice_and_padding(test_vol_arrs, test_vol_label_arrs, \
                                             self.padding_axis1, self.padding_axis2, shuffle_slices=False)
        print(test_slices_arrs.shape)
        print(test_slices_label_arrs.shape)
        return test_slices_arrs, test_slices_label_arrs


if __name__ == "__main__":
    nd = NeedleData ()
    a, b, c, d = nd.load_train_validation_data(split_num = 1, balance = True, balance_ratio = 2, fliplr = True)
    print(a.shape)
    print(b.shape)
    print(c.shape)
    print(d.shape)




