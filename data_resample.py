import numpy as np
import os
import glob
import shutil
import SimpleITK as sitk
import matplotlib.pyplot as plt

from project.utils import IntermediateUtil
from project.settings import cases_root, slicer_dir, intermediate_dir


def cast_and_resample(itk_img, spacing):
    itk_img = sitk.Cast(sitk.RescaleIntensity(itk_img), sitk.sitkUInt8)
    itk_img.SetSpacing(spacing)
    return itk_img


if __name__ == '__main__':
    SPACING = [0.75,0.75,3.6]
    iu = IntermediateUtil(intermediate_dir, sub_dir = "nrrd/")   
    volumes = iu.get_needle_vol_files()
    labelmaps = iu.get_needle_map_files()
    masks = iu.get_prostate_mask_files()
    
    nrrd_resample_dir = intermediate_dir + "nrrd_resample/"
    if not os.path.isdir(nrrd_resample_dir):
        os.mkdir(nrrd_resample_dir)
    
    for volume in volumes:
        volume_img = sitk.ReadImage(volume)
        volume_img_ = cast_and_resample(volume_img, SPACING)
        output_file = nrrd_resample_dir + os.path.basename(volume)
        print(output_file)
        sitk.WriteImage(volume_img_, output_file)

    for labelmap in labelmaps:
        label_img = sitk.ReadImage(labelmap)
        label_img_ = cast_and_resample(label_img, SPACING)
        output_file = nrrd_resample_dir + os.path.basename(labelmap)
        print(output_file)
        sitk.WriteImage(label_img_, output_file)
        
    for mask in masks:
        mask_img = sitk.ReadImage(mask)
        mask_img_ = cast_and_resample(mask_img, SPACING)
        output_file = nrrd_resample_dir + os.path.basename(mask_file)
        print(output_file)
        sitk.WriteImage(mask_img_, output_file)
        
        