import numpy as np
import os
import glob
import shutil
import SimpleITK as sitk
import matplotlib.pyplot as plt

from project.utils import IntermediateUtil
from project.settings import cases_root, slicer_dir, intermediate_dir

def get_boundary_from_mask(mask_files):
    x_min = []
    x_max = []
    y_min = []
    y_max = []
    z_min = []
    z_max = []
    for mask_file in mask_files:
        mask_img = sitk.ReadImage(mask_file)
        mask_array = sitk.GetArrayFromImage(mask_img)
        x,y,z = np.where(mask_array==255)
        x_min.append(min(x))
        x_max.append(max(x))
        y_min.append(min(y))
        y_max.append(max(y))
        z_min.append(min(z))
        z_max.append(max(z))
    x_min = min(x_min)
    x_max = max(x_max)
    y_min = min(y_min)
    y_max = max(y_max)
    z_min = min(z_min)
    z_max = max(z_max)
    print("x_min: ", x_min, "x_max: ", x_max)
    print("y_min: ", y_min, "y_max: ", y_max)
    print("z_min: ", z_min, "z_max: ", z_max)
    return x_min, x_max, y_min, y_max, z_min, z_max

if __name__ == '__main__':
    iu = IntermediateUtil(intermediate_dir, sub_dir = "nrrd_resample/")
    volumes = iu.get_needle_vol_files()
    labelmaps = iu.get_needle_map_files()
    masks = iu.get_prostate_mask_files()
    x_min, x_max, y_min, y_max, z_min, z_max = 0, 24, 48, 208, 65, 245   #get_boundary_from_mask(masks)
    
    for i, mask in enumerate(masks):
        vol_img = sitk.ReadImage(volumes[i])
        label_img = sitk.ReadImage(labelmaps[i])
        spacing = vol_img.GetSpacing()
        
        label_array = sitk.GetArrayFromImage(label_img)
        vol_array = sitk.GetArrayFromImage(vol_img)
        
        label_array_crop = label_array[:, y_min:y_max, z_min:z_max]
        vol_array_crop = vol_array[:, y_min:y_max, z_min:z_max]
        
        if vol_array_crop.shape[0] < x_max - x_min:     
            upper_padding = (x_max - x_min - vol_array_crop.shape[0])//2
            lower_padding = x_max - x_min - vol_array_crop.shape[0] - upper_padding
            label_array_crop = np.lib.pad(label_array_crop, ((upper_padding,lower_padding),(0,0),(0,0)), 'constant', constant_values=(0, 0))
            vol_array_crop = np.lib.pad(vol_array_crop, ((upper_padding,lower_padding),(0,0),(0,0)), 'constant', constant_values=(0, 0))
        if vol_array_crop.shape[1] < y_max - y_min:
            left_padding = (y_max - y_min - vol_array_crop.shape[1])//2
            right_padding = y_max - y_min - vol_array_crop.shape[1] - left_padding
            label_array_crop = np.lib.pad(label_array_crop, ((0, 0),(left_padding, right_padding),(0,0)), 'constant', constant_values=(0, 0))
            vol_array_crop = np.lib.pad(vol_array_crop, ((0, 0),(left_padding, right_padding),(0,0)), 'constant', constant_values=(0, 0))
        if vol_array_crop.shape[2] < z_max - z_min:
            front_padding = (z_max - z_min - vol_array_crop.shape[2])//2
            rear_padding = z_max - z_min - vol_array_crop.shape[2] - front_padding
            label_array_crop = np.lib.pad(label_array_crop, ((0,0),(0,0),(front_padding,rear_padding)), 'constant', constant_values=(0, 0))
            vol_array_crop = np.lib.pad(vol_array_crop, ((0,0),(0,0),(front_padding,rear_padding)), 'constant', constant_values=(0, 0))
        
        print(vol_array_crop.shape, label_array_crop.shape)
        vol_img = sitk.GetImageFromArray(vol_array_crop)
        vol_img.SetSpacing(spacing)
        label_img = sitk.GetImageFromArray(label_array_crop)
        label_img.SetSpacing(spacing)
        
        nrrd_resize_dir = os.path.join(intermediate_dir, "nrrd_resize/")
        if not os.path.isdir(nrrd_resize_dir):
            os.mkdir(nrrd_resize_dir)
        vol_file = os.path.join(nrrd_resize_dir, os.path.basename(volumes[i]))
        label_file = os.path.join(nrrd_resize_dir, os.path.basename(labelmaps[i]))

        sitk.WriteImage(vol_img, vol_file)
        sitk.WriteImage(label_img, label_file)