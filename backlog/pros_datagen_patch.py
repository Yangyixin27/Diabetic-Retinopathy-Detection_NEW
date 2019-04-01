import glob
import json
import os
import sys
import time

import numpy as np

for path in ['.', '..']:
    sys.path.append(path)

import SimpleITK as sitk
from sklearn.model_selection import KFold

from project.settings import intermediate_data_folder
from data.pros_resample import get_volume_path

data_specifications = dict()

INPUT_NRRD_DIRECTORY = os.path.join(intermediate_data_folder, 'NRRD-1-5-T-UINT8-DS-2')
data_specifications['input-nrrd-dir'] = INPUT_NRRD_DIRECTORY
SPLIT_RATIO = 0.2
data_specifications['split-ration'] = SPLIT_RATIO
PATCH_SIZE = 32
data_specifications['patch_size'] = PATCH_SIZE
SLICE_LEVEL_NORMALIZATION = True
data_specifications['slice_level_normalization'] = SLICE_LEVEL_NORMALIZATION
OUTPUT_DIRECTORY = os.path.join(intermediate_data_folder, 'npy', time.strftime("%Y_%m_%d_%H_%M"))
data_specifications['output-dir'] = OUTPUT_DIRECTORY

pad = round(PATCH_SIZE / 2)


def bounding_box(arr):
    a = np.where(arr != 0)
    if a[0].size and a[1].size:
        return np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])


def get_pos_neg_counts(label_list):
    pos_count = 0
    neg_count = 0
    for label_path in label_list:
        label = sitk.ReadImage(label_path)
        label_array = sitk.GetArrayFromImage(label)
        for index, slice_arr in enumerate(label_array):
            slice_arr_c = slice_arr[pad:-pad, pad:-pad]
            pos_count += np.sum(slice_arr_c > 0)
            neg_count += np.sum(slice_arr_c == 0)
    return pos_count, neg_count


def create_pos_neg_arrs(label_list):
    pos_count, neg_count = get_pos_neg_counts(label_list)
    print("pos_count: {0}, neg_count: {1}, ratio: {2:0.3f}".format(pos_count, neg_count,
                                                                   pos_count / (pos_count + neg_count)))
    pos_array = np.zeros((pos_count, 1, PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
    neg_array = np.zeros((neg_count, 1, PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
    pos_position = 0
    neg_position = 0
    for label_path in label_list:
        label = sitk.ReadImage(label_path)
        label_array = sitk.GetArrayFromImage(label)
        volume_path = get_volume_path(label_path)
        volume = sitk.ReadImage(volume_path)
        volume_array = sitk.GetArrayFromImage(volume)
        for index, slice_arr in enumerate(label_array):
            slice_arr_c = slice_arr[pad:-pad, pad:-pad]
            volume_arr = volume_array[index, :, :]
            volume_arr_max = np.amax(volume_arr)
            if SLICE_LEVEL_NORMALIZATION == True:
                volume_arr = volume_arr / volume_arr_max * 255
            #
            idx_j = np.where(slice_arr_c > 0)[0]
            idx_i = np.where(slice_arr_c > 0)[1]
            for j, i in zip(idx_j, idx_i):
                pos_array[pos_position, 0] = volume_arr[j:j + PATCH_SIZE, i:i + PATCH_SIZE]
                pos_position += 1
            #
            idx_j = np.where(slice_arr_c == 0)[0]
            idx_i = np.where(slice_arr_c == 0)[1]
            for j, i in zip(idx_j, idx_i):
                neg_array[neg_position, 0] = volume_arr[j:j + PATCH_SIZE, i:i + PATCH_SIZE]
                neg_position += 1
    print(pos_position)
    print(neg_position)
    return pos_array, neg_array


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    labels = sorted(glob.glob(INPUT_NRRD_DIRECTORY + '/*_label.nrrd'))
    n_total = len(labels)
    print(OUTPUT_DIRECTORY)
    print('-' * 100)
    print("no of 1.5T labels: {}".format(n_total))
    '''Selecting test data
    >> np.random.choice(30, 5)
    >> array([2, 11, 8, 16, 21])'''
    # test_index is hard coded.(Generated as above)
    test_index = [2, 11, 8, 16, 21]
    test_labels = [labels[i] for i in test_index]
    with open(os.path.join(OUTPUT_DIRECTORY, 'test.txt'), 'w') as output_file:
        for test_label in test_labels:
            output_file.write("%s\n" % os.path.basename(test_label))

    train_val_labels = list(set(labels) - set(test_labels))
    n_train_val = len(train_val_labels)

    kf = KFold(n_splits=5, shuffle=True)
    x = np.arange(0, n_train_val)
    fold = 0
    for train_index, val_index in kf.split(x):
        print('*' * 100)
        print('fold: {}'.format(fold))
        print("TRAIN:", train_index, "VAL:", val_index)
        train_labels = [train_val_labels[i] for i in train_index]
        val_labels = [train_val_labels[i] for i in val_index]
        fold += 1

        # Training Arrays
        pos_array, neg_array = create_pos_neg_arrs(train_labels)
        np.save(os.path.join(OUTPUT_DIRECTORY, 'pos_train' + str(fold).zfill(2) + '.npy'), pos_array)
        np.save(os.path.join(OUTPUT_DIRECTORY, 'neg_train' + str(fold).zfill(2) + '.npy'), neg_array)

        # Validation Arrays
        pos_array, neg_array = create_pos_neg_arrs(val_labels)
        np.save(os.path.join(OUTPUT_DIRECTORY, 'pos_val' + str(fold).zfill(2) + '.npy'), pos_array)
        np.save(os.path.join(OUTPUT_DIRECTORY, 'neg_val' + str(fold).zfill(2) + '.npy'), neg_array)

        with open(os.path.join(OUTPUT_DIRECTORY, 'training' + '-' + str(fold).zfill(2) + '.txt'), 'w') as output_file:
            for train_label in train_labels:
                output_file.write("%s\n" % os.path.basename(train_label))
        with open(os.path.join(OUTPUT_DIRECTORY, 'validation' + '-' + str(fold).zfill(2) + '.txt'), 'w') as output_file:
            for val_label in val_labels:
                output_file.write("%s\n" % os.path.basename(val_label))

    with open(os.path.join(OUTPUT_DIRECTORY, 'data-specs.json'), 'w') as fp:
        json.dump(data_specifications, fp)
