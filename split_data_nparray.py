import glob
import os
import sys
import json
import time
import numpy as np
import shutil
import SimpleITK as sitk
from sklearn.model_selection import KFold

from project.settings import cases_root, slicer_dir, intermediate_dir
from project.utils import IntermediateUtil


def loadCases(p):
    f = open(p)
    res = []
    for l in f:
        l = l[:-1]
        if l == "":
            break
        if l[-1] == '\r':
            l = l[:-1]
        res.append(l)
    return res


def store_npy_from_cases_to_outputdir(case_nums, category, output_dir, split_num=0, save = False):
        assert category == "training" or "validation" or "testing", "wrong category"
        all_vol_arrays = []
        all_map_arrays = []
        for case_num in case_nums:
            vols = iu.get_needle_vol_files_with_case(case_num)
            maps = iu.get_needle_map_files_with_case(case_num)
            for vol in vols:
                vol_array = sitk.GetArrayFromImage(sitk.ReadImage(vol))
                vol_array = vol_array/np.amax(vol_array)
                all_vol_arrays.append(vol_array)
            for labelmap in maps:
                map_array = sitk.GetArrayFromImage(sitk.ReadImage(labelmap))
                all_map_arrays.append(map_array)
           
        all_vol_arrays = np.array(all_vol_arrays)
        all_map_arrays = np.array(all_map_arrays)
        all_vol_mean = np.mean(all_vol_arrays)
        all_vol_std = np.std(all_vol_arrays)
        #numpyarray = np.concatenate([np.expand_dims(all_array,axis =0) for all_array in all_vol_arrays], axis=0)
        print(category + " data shape: ", all_vol_arrays.shape)
        print(category + " labelmap shape: ", all_map_arrays.shape)
            
        if save != False:
            if category == "testing":
                np.save(output_dir + category + "_data.npy", all_vol_arrays)
                np.save(output_dir + category + "_labelmap.npy", all_map_arrays)
            else:
                np.save(output_dir + "{}/".format(split_num) + category + "_data.npy", all_vol_arrays)
                np.save(output_dir + "{}/".format(split_num) + category + "_labelmap.npy", all_map_arrays)
            if category == "training":
                np.save(output_dir + "{}/".format(split_num) + category + "_mean.npy", all_vol_mean)
                np.save(output_dir + "{}/".format(split_num) + category + "_std.npy", all_vol_std)
    
   
            
if __name__ == '__main__':
    SAVE = True
    SPLIT = 5
    OUTPUTDIR = intermediate_dir + "numpy/"
    FOLDDIR = intermediate_dir + "fold/"
    if not os.path.isdir(OUTPUTDIR):
        os.mkdir(OUTPUTDIR)
    
    iu = IntermediateUtil(intermediate_dir, "nrrd_resize/")
    test_txt = FOLDDIR + "test.txt"
    testing_nums = loadCases(test_txt)
    store_npy_from_cases_to_outputdir(testing_nums, "testing", OUTPUTDIR, None, SAVE)
    
    for i in range(SPLIT):
        i = i+1
        validation_txt = FOLDDIR + "validation_{}.txt".format(i)
        training_txt = FOLDDIR + "train_{}.txt".format(i)
        validation_nums = loadCases(validation_txt)
        training_nums = loadCases(training_txt)
        print("validation cases: ", validation_nums)
        if not os.path.exists(OUTPUTDIR + "{}".format(i)):
            os.makedirs(OUTPUTDIR + "{}".format(i))
            
        store_npy_from_cases_to_outputdir(validation_nums, "validation", OUTPUTDIR, i, SAVE)
        store_npy_from_cases_to_outputdir(training_nums, "training", OUTPUTDIR, i, SAVE)