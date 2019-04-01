from __future__ import division

import csv
import glob
import math
import os
import random
import warnings

import SimpleITK as sitk
import numpy as np
import pandas as pd

module_root = '..'
import sys

sys.path.append(module_root)

from configs.config_1 import *

VOXEL_SIZE_I = config['VOXEL_SIZE_I']
VOXEL_SIZE_J = config['VOXEL_SIZE_J']
VOXEL_SIZE_K = config['VOXEL_SIZE_K']
OUTPUT_SIZE_T2_A = config['OUTPUT_SIZE_T2_A']
OUTPUT_SIZE_T2_S = config['OUTPUT_SIZE_T2_S']
OUTPUT_SIZE_T2_C = config['OUTPUT_SIZE_T2_C']
OUTPUT_SIZE_ADC = config['OUTPUT_SIZE_ADC']
OUTPUT_SIZE_BVAL = config['OUTPUT_SIZE_BVAL']
OUTPUT_SIZE_KTRANS = config['OUTPUT_SIZE_KTRANS']
SERIES_IDS = config['SERIES_IDS']
SECTORS = config['SECTORS']
PERMUTATION_DIRECTORY = config['PERMUTATION_DIRECTORY']
OUTPUT_DIRECTORY = config['OUTPUT_DIRECTORY']
TRAIN_NRRD_FOLDER = config['TRAIN_NRRD_FOLDER']
TEST_NRRD_FOLDER = config['TEST_NRRD_FOLDER']
N_TRAINING = config['N_TRAINING']
N_VALIDATION = config['N_VALIDATION']
CREATE_TRAIN = config['CREATE_TRAIN']
CREATE_VALIDATION = config['CREATE_VALIDATION']
CREATE_VALIDATION_ORIGINAL = config['CREATE_VALIDATION_ORIGINAL']
CREATE_TEST = config['CREATE_TEST']
EXPOSURE_CORRECTION = config['CORRECT_EXPOSURE']
PCLOW = config['PCLOW']
PCHIGH = config['PCHIGH']

# Augmentation parameters
FLIP_AUG = config['FLIP_AUG']
BALANCE_ZONE_DISTRIBUTION = config['BALANCE_ZONE_DISTRIBUTION']

TRANSLATION_X_RANGE = config['TRANSLATION_X_RANGE']
TRANSLATION_Y_RANGE = config['TRANSLATION_Y_RANGE']
TRANSLATION_Z_RANGE = config['TRANSLATION_Z_RANGE']
ROTATION_X_RANGE = config['ROTATION_X_RANGE']
ROTATION_Y_RANGE = config['ROTATION_Y_RANGE']
ROTATION_Z_RANGE = config['ROTATION_Z_RANGE']

# Image data dimension
IMAGE_DATA_DIM = config['IMAGE_DATA_DIM']


def get_roi_parameters(series):
    i, j, k = map(int, series.ijk.values[0].split(' '))
    ijk = [i, j, k]
    si, sj, sk = map(float, series.VoxelSpacing.values[0].split(','))
    index = [i - int(VOXEL_SIZE_I / si / 2), j - int(VOXEL_SIZE_J / sj / 2), k - int(VOXEL_SIZE_K / sk / 2)]
    size = [int(VOXEL_SIZE_I / si), int(VOXEL_SIZE_J / sj), int(VOXEL_SIZE_K / sk)]
    pos = tuple(map(float, series.pos.values[0].split(' ')))
    return index, size, ijk, pos


def get_random_transformation():
    # Todo: Rotation has bugs! for now just we use translation for augmentation.
    translation_x = np.random.random(1) * TRANSLATION_X_RANGE * flip_a_coin()
    translation_y = np.random.random(1) * TRANSLATION_Y_RANGE * flip_a_coin()
    translation_z = np.random.random(1) * TRANSLATION_Z_RANGE * flip_a_coin()
    theta_x = 0
    theta_y = 0
    theta_z = np.random.uniform(-ROTATION_Z_RANGE, ROTATION_Z_RANGE) * flip_a_coin()
    translation = (translation_x[0], translation_y[0], translation_z[0])
    rotation = (theta_x, theta_y, theta_z)
    return translation, rotation


def apply_transformation(image, translation, rotation, pos):
    rotation_center = pos
    rigid_euler = sitk.Euler3DTransform(rotation_center, rotation[0], rotation[1], rotation[2], translation)
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(image)
    similarity = sitk.Similarity3DTransform()
    similarity.SetMatrix(rigid_euler.GetMatrix())
    similarity.SetTranslation(rigid_euler.GetTranslation())
    similarity.SetCenter(rigid_euler.GetCenter())
    resample.SetTransform(similarity)
    transformed_image = resample.Execute(image)
    return transformed_image


def correct_exposure(nda):
    from skimage import exposure
    pl, ph = np.percentile(nda, (PCLOW, PCHIGH))
    nda = exposure.rescale_intensity(nda, in_range=(pl, ph))
    return nda


def zero_pad(image, index, pad):
    filter = sitk.ConstantPadImageFilter()
    filter.SetConstant(0)
    filter.SetPadLowerBound([pad, pad, pad])
    filter.SetPadUpperBound([pad, pad, pad])
    image = filter.Execute(image)
    index = [item + pad for item in index]
    return image, index


def crop_and_resample_image(image, index, size, series_id, uid):
    output_sizes = [OUTPUT_SIZE_T2_A, OUTPUT_SIZE_T2_S, OUTPUT_SIZE_T2_C, OUTPUT_SIZE_ADC, OUTPUT_SIZE_BVAL,
                    OUTPUT_SIZE_KTRANS]
    output_size = output_sizes[SERIES_IDS.index(series_id)]
    bound_u = [i + j for i, j in zip(index, size)]
    diff = [i - j for i, j in zip(bound_u, list(image.GetSize()))]
    if max(diff) > 0:
        pad = max(diff)
        image, index = zero_pad(image, index, pad)

    elif min(index) < 0:
        pad = -min(index)
        image, index = zero_pad(image, index, pad)

    roi_filter = sitk.RegionOfInterestImageFilter()
    roi_filter.SetIndex(index)
    roi_filter.SetSize(size)
    try:
        croped_image = roi_filter.Execute(image)
        '''Rescale Intensity
        It is necessary to rescale intensity if casting is needed!
        If not done there is a chance of data loss.
        '''
        # if series_id in ['Ktrans', 'ADC', ]:
        #    pixel_type = image.GetPixelID()
        pixel_type = sitk.sitkUInt8
        croped_image = sitk.RescaleIntensity(croped_image)
        croped_image = sitk.Cast(croped_image, pixel_type)
        # resampling part
        resample = sitk.ResampleImageFilter()
        input_size = croped_image.GetSize()
        # print("input size:", input_size)
        # print("output size:", output_size)
        input_spacing = croped_image.GetSpacing()
        output_spacing = [input_spacing[0] / output_size[0] * input_size[0],
                          input_spacing[1] / output_size[1] * input_size[1],
                          input_spacing[2] / output_size[2] * input_size[2]]
        resample.SetSize(output_size)
        resample.SetOutputPixelType(pixel_type)
        resample.SetOutputOrigin(croped_image.GetOrigin())
        resample.SetOutputDirection(croped_image.GetDirection())
        resample.SetOutputSpacing(output_spacing)
        doresample = True
        if doresample:
            resampled_image = resample.Execute(croped_image)
        else:
            resampled_image = croped_image
        return [True, resampled_image]
    except Exception as e:
        print('\n' + str(e))
        # warnings.warn('\n'+str(e))
        return [False, uid + '_' + series_id]


def flip_a_coin():
    return np.random.binomial(1, 0.5)


def select_random_sector():
    return random.choice(SECTORS)


def get_finding_ndas(patient_id, fid, images_df, mode):
    # manual for test
    # patient_id = "ProstateX-0227"
    # fid = 1
    nrrd_folder = TRAIN_NRRD_FOLDER if mode != 'test' else TEST_NRRD_FOLDER
    uid = str(patient_id) + '_' + str(fid)

    translation, rotation = get_random_transformation()

    # print(100*'$')
    # print(patient_id +' -'+ str(fid))
    for series_id in SERIES_IDS:
        # print(series_id)
        # TODO: for ktrans and adc do not rescale the intensity!
        if series_id == 'Ktrans':
            filename = glob.glob(nrrd_folder + '/*' + patient_id + '*' + series_id + '*')
            # Ktrans image information is not directly available in images.csv it was however calculated from
            # PD, so we use the ijk of the PD
            series_id = 'tfl_3d_PD'
            series = images_df[(images_df.ProxID == patient_id) &
                               (images_df.fid == fid) & images_df.Name.str.contains(series_id)]
            # print(series_id)
            if patient_id in ['ProstateX-0331', ] :
                filename = [os.path.join(nrrd_folder, 'ProstateX-0330_Ktrans.nrrd')]
            latest_series_no = max(series.DCMSerNum.values)
            selected_image = series[series.DCMSerNum == latest_series_no]
            # Todo: (in the spreadsheet) select the best series for duplicates and only keep that in table
            # print('series number: {}'.format(latest_series_no))
            index, size, ijk, pos = get_roi_parameters(selected_image)
            series_id = 'Ktrans'
        else:
            series = images_df[(images_df.ProxID == patient_id) &
                               (images_df.fid == fid) & images_df.Name.str.contains(series_id)]
            # print(series_id)
            if series_id == 't2_tse_cor' and (patient_id in ['ProstateX-0298', 'ProstateX-0299']) :
               filename = [os.path.join(nrrd_folder, 'ProstateX-0297_t2_tse_cor_6.nrrd')]
            elif series_id == 'BVAL' and (patient_id in ['ProstateX-0227', ]) :
                filename = [os.path.join(nrrd_folder, 'ProstateX-0227_ep2d_DIFF_tra_b50_500_800_1400_alle_spoelen_ADC_7.nrrd')]
            else:
                latest_series_no = max(series.DCMSerNum.values)
                selected_image = series[series.DCMSerNum == latest_series_no]
                # Todo: (in the spreadsheet) select the best series for duplicates and only keep that in table
                # print('series number: {}'.format(latest_series_no))
                index, size, ijk, pos = get_roi_parameters(selected_image)
                filename = glob.glob(nrrd_folder + '/*' + patient_id + '*' + series_id
                                     + '*' + str(latest_series_no) + '*')

        if filename:
            # print(filename)
            filename = filename[0]

        # print(filename)
        image = sitk.ReadImage(filename)

        # reset the image settings
        # nda = sitk.GetArrayFromImage(image)
        # image = sitk.GetImageFromArray(nda)

        if mode not in ['validation_original', 'test']:
            image = apply_transformation(image, translation=translation, rotation=rotation, pos=pos)
        output = crop_and_resample_image(image, index=index, size=size, series_id=series_id, uid=uid)

        if output[0]:
            resampled_image = output[1]
            if series_id == "t2_tse_tra":
                tra_nda = sitk.GetArrayFromImage(resampled_image)
            if series_id == "t2_tse_sag":
                sag_nda = sitk.GetArrayFromImage(resampled_image)
            if series_id == "t2_tse_cor":
                cor_nda = sitk.GetArrayFromImage(resampled_image)
            if series_id == "ADC":
                adc_nda = sitk.GetArrayFromImage(resampled_image)
            if series_id == "BVAL":
                bval_nda = sitk.GetArrayFromImage(resampled_image)
            if series_id == "Ktrans":
                ktrans_nda = sitk.GetArrayFromImage(resampled_image)

    return tra_nda, sag_nda, cor_nda, adc_nda, bval_nda, ktrans_nda, uid, translation, rotation


def fliplr_2d_and_3d(nda):
    nda_f = np.zeros_like(nda)
    if nda.ndim == 2:
        nda_f = np.fliplr(nda)
        return nda_f
    elif nda.ndim == 3:
        for j in range(nda.shape[0]):
            nda_f[j, :, :] = np.fliplr(nda[j, :, :])
        return nda_f
    else:
        warnings.warn('the numpy array dim must be either 2 or 3.')


def get_middle_slice(nda):
    m = math.floor(np.shape(nda)[0] / 2)
    return nda[m, :, :]


def create_empty_arrays(n, dim):
    if dim == 2:
        tra_nda = np.zeros((n, 1, OUTPUT_SIZE_T2_A[1], OUTPUT_SIZE_T2_A[0]), dtype=np.float32)
        sag_nda = np.zeros((n, 1, OUTPUT_SIZE_T2_S[1], OUTPUT_SIZE_T2_S[0]), dtype=np.float32)
        cor_nda = np.zeros((n, 1, OUTPUT_SIZE_T2_C[1], OUTPUT_SIZE_T2_C[0]), dtype=np.float32)
        adc_nda = np.zeros((n, 1, OUTPUT_SIZE_ADC[1], OUTPUT_SIZE_ADC[0]), dtype=np.float32)
        bval_nda = np.zeros((n, 1, OUTPUT_SIZE_BVAL[1], OUTPUT_SIZE_BVAL[0]), dtype=np.float32)
        ktrans_nda = np.zeros((n, 1, OUTPUT_SIZE_KTRANS[1], OUTPUT_SIZE_KTRANS[0]), dtype=np.float32)
        adc_bval_ktrans_nda = np.zeros((n, 3, OUTPUT_SIZE_KTRANS[1], OUTPUT_SIZE_KTRANS[0]), dtype=np.float32)
    elif dim == 3:
        tra_nda = np.zeros((n, 1, OUTPUT_SIZE_T2_A[2], OUTPUT_SIZE_T2_A[1], OUTPUT_SIZE_T2_A[0]), dtype=np.float32)
        sag_nda = np.zeros((n, 1, OUTPUT_SIZE_T2_S[2], OUTPUT_SIZE_T2_S[1], OUTPUT_SIZE_T2_S[0]), dtype=np.float32)
        cor_nda = np.zeros((n, 1, OUTPUT_SIZE_T2_C[2], OUTPUT_SIZE_T2_C[1], OUTPUT_SIZE_T2_C[0]), dtype=np.float32)
        adc_nda = np.zeros((n, 1, OUTPUT_SIZE_ADC[2], OUTPUT_SIZE_ADC[1], OUTPUT_SIZE_ADC[0]), dtype=np.float32)
        bval_nda = np.zeros((n, 1, OUTPUT_SIZE_BVAL[2], OUTPUT_SIZE_BVAL[1], OUTPUT_SIZE_BVAL[0]), dtype=np.float32)
        ktrans_nda = np.zeros((n, 1, OUTPUT_SIZE_KTRANS[2], OUTPUT_SIZE_KTRANS[1], OUTPUT_SIZE_KTRANS[0]),
                              dtype=np.float32)
        adc_bval_ktrans_nda = np.zeros((n, 3, OUTPUT_SIZE_KTRANS[2], OUTPUT_SIZE_KTRANS[1], OUTPUT_SIZE_KTRANS[0]),
                                       dtype=np.float32)
    clinsig_arr = np.zeros((n, 1), dtype=np.bool)
    anatomical_arr = np.zeros((n, 3), dtype=np.float32)
    zone_encoding_arr = np.zeros((n, 3), dtype=np.bool)
    return tra_nda, sag_nda, cor_nda, adc_nda, bval_nda, ktrans_nda, \
           adc_bval_ktrans_nda, clinsig_arr, anatomical_arr, zone_encoding_arr


def get_patient_id_fid(findings_df, subjects):
    sample_clin_sig = flip_a_coin()
    if sample_clin_sig:
        # pick from True ClinSig
        selected_df = findings_df[(findings_df.ProxID.isin(subjects))
                                  & (findings_df.ClinSig == True)]
        if BALANCE_ZONE_DISTRIBUTION:
            zone = select_random_sector()
            selected_df = selected_df[selected_df.zone == zone]
            # print('true selected')
    else:
        # pick from False ClinSig
        selected_df = findings_df[(findings_df.ProxID.isin(subjects))
                                  & (findings_df.ClinSig == False)]
        if BALANCE_ZONE_DISTRIBUTION:
            zone = select_random_sector()
            selected_df = selected_df[selected_df.zone == zone]
            # print('false selected')
    row = np.random.choice(selected_df.index.values, 1)
    sampled_row = selected_df.ix[row]
    patient_id = sampled_row.ProxID.values[0]
    sample_zone = sampled_row.zone.values[0]
    fid = sampled_row.fid.values[0]
    return patient_id, fid, sample_zone, sample_clin_sig


def process_ndas(tra_nda, sag_nda, cor_nda, adc_nda, bval_nda, ktrans_nda):
    if IMAGE_DATA_DIM == 2:
        tra_nda = get_middle_slice(tra_nda)
    if np.amax(tra_nda) > 0:
        tra_nda = tra_nda / 255.
    else:
        warnings.warn('tra all zeors: {}, {}'.format(patient_id, fid))
    if IMAGE_DATA_DIM == 2:
        sag_nda = get_middle_slice(sag_nda)
    if np.amax(sag_nda) > 0:
        sag_nda = sag_nda / 255.
    else:
        warnings.warn('sag all zeors: {}, {}'.format(patient_id, fid))
    if IMAGE_DATA_DIM == 2:
        cor_nda = get_middle_slice(cor_nda)
    if np.amax(cor_nda) > 0:
        cor_nda = cor_nda / 255.
    else:
        warnings.warn('cor all zeors: {}, {}'.format(patient_id, fid))
    if IMAGE_DATA_DIM == 2:
        adc_nda = get_middle_slice(adc_nda)
    if np.amax(adc_nda) > 0:
        adc_nda = adc_nda / 255.
    else:
        warnings.warn('adc all zeors: {}, {}'.format(patient_id, fid))
    if IMAGE_DATA_DIM == 2:
        bval_nda = get_middle_slice(bval_nda)
    if np.amax(bval_nda) > 0:
        bval_nda = bval_nda / 255.
    else:
        warnings.warn('bval all zeors: {}, {}'.format(patient_id, fid))
    if IMAGE_DATA_DIM == 2:
        ktrans_nda = get_middle_slice(ktrans_nda)
    if np.amax(ktrans_nda) > 0:
        ktrans_nda = ktrans_nda / 255.
    else:
        warnings.warn('ktrans all zeors: {}, {}'.format(patient_id, fid))

    return tra_nda, sag_nda, cor_nda, adc_nda, bval_nda, ktrans_nda


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIRECTORY)
    print(OUTPUT_DIRECTORY)

    with open(os.path.join(OUTPUT_DIRECTORY, 'config.csv'), 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in sorted(config.items()):
            writer.writerow([key, value])

    training_file = os.path.join(PERMUTATION_DIRECTORY, "training.txt")
    with open(training_file) as f:
        training_subjects = f.read().splitlines()
    images_table_path = os.path.join(s.sheets_folder, "images.csv")
    images_df = pd.read_csv(images_table_path)

    findings_table_path = os.path.join(s.sheets_folder, "findings.csv")
    findings_df = pd.read_csv(findings_table_path)
    findings_df = findings_df[findings_df.zone.isin(SECTORS)]

    anatomical_table_path = os.path.join(s.sheets_folder, "anatomical.csv")
    anatomical_df = pd.read_csv(anatomical_table_path)

    '''
    #############################################################################################
    TRAINING DATA
    #############################################################################################
    '''
    if CREATE_TRAIN:
        # Theano data shape. Read more: https://keras.io/backend/

        train_tra_nda, train_sag_nda, train_cor_nda, train_bval_nda, train_adc_nda, train_ktrans_nda, \
        adc_bval_ktrans_nda, train_clinsig_arr, train_anatomical_arr, \
        train_zone_encoding_arr = create_empty_arrays(N_TRAINING, IMAGE_DATA_DIM)
        train_uids = []
        train_clin_sigs = []
        train_zones = []
        transformations = []
        failed_cases = []
        if FLIP_AUG:
            n = int(N_TRAINING / 2)
        else:
            n = N_TRAINING
        for i in range(n):
            patient_id, fid, sample_zone, sample_clin_sig = get_patient_id_fid(findings_df=findings_df,
                                                                               subjects=training_subjects)
            tra_nda, sag_nda, cor_nda, adc_nda, bval_nda, ktrans_nda, uid, translation, rotation = get_finding_ndas(
                patient_id, fid, images_df, mode='train')
            tra_nda, sag_nda, cor_nda, adc_nda, bval_nda, ktrans_nda = process_ndas(
                tra_nda, sag_nda, cor_nda, adc_nda, bval_nda, ktrans_nda)
            train_tra_nda[i], train_sag_nda[i], train_cor_nda[i], train_adc_nda[i], train_bval_nda[i], \
            train_ktrans_nda[i] = tra_nda, sag_nda, cor_nda, adc_nda, bval_nda, ktrans_nda
            if FLIP_AUG:
                train_tra_nda[i + n] = fliplr_2d_and_3d(tra_nda)
                train_sag_nda[i + n] = fliplr_2d_and_3d(sag_nda)
                train_cor_nda[i + n] = fliplr_2d_and_3d(cor_nda)
                train_adc_nda[i + n] = fliplr_2d_and_3d(adc_nda)
                train_bval_nda[i + n] = fliplr_2d_and_3d(bval_nda)
                train_ktrans_nda[i + n] = fliplr_2d_and_3d(ktrans_nda)

            if np.amax(adc_nda) > 0 and np.amax(bval_nda) > 0 and np.amax(ktrans_nda) > 0:
                adc_bval_ktrans_nda[i, 0] = adc_nda
                adc_bval_ktrans_nda[i, 1] = bval_nda
                adc_bval_ktrans_nda[i, 2] = ktrans_nda

                if FLIP_AUG:
                    adc_bval_ktrans_nda[i + n, 0] = fliplr_2d_and_3d(adc_nda)
                    adc_bval_ktrans_nda[i + n, 1] = fliplr_2d_and_3d(bval_nda)
                    adc_bval_ktrans_nda[i + n, 2] = fliplr_2d_and_3d(ktrans_nda)
            else:
                warnings.warn('adc_bval_ktrans_nda all zeors: {}, {}'.format(patient_id, fid))

            selected_anatomical_df = anatomical_df[(anatomical_df.ProxID == patient_id) & (anatomical_df.fid == fid)]
            relative_x = selected_anatomical_df.RelativeX.values[0]
            relative_y = selected_anatomical_df.RelativeY.values[0]
            relative_z = selected_anatomical_df.RelativeZ.values[0]
            train_anatomical_arr[i] = np.array([relative_x, relative_y, relative_z])
            if FLIP_AUG:
                train_anatomical_arr[i + n] = np.array([1 - relative_x, relative_y, relative_z])

            # one hot encoding zone information
            if sample_zone == 'PZ':
                zone_encoding = np.array([1, 0, 0])
            if sample_zone == 'TZ':
                zone_encoding = np.array([0, 1, 0])
            if sample_zone == 'AS':
                zone_encoding = np.array([0, 0, 1])
            train_zone_encoding_arr[i] = zone_encoding
            if FLIP_AUG:
                train_zone_encoding_arr[i + n] = zone_encoding

            train_clinsig_arr[i] = bool(sample_clin_sig)
            if FLIP_AUG:
                train_clinsig_arr[i + n] = bool(sample_clin_sig)
            train_uids.append(uid)
            train_zones.append(sample_zone)
            train_clin_sigs.append(str(sample_clin_sig))
            transformation = []
            transformation.extend(translation)
            transformation.extend(rotation)
            transformations.append(transformation)
            if i % 20 == 0:
                if FLIP_AUG:
                    print('Done: {0}/{1} training findings.'.format(2 * i, N_TRAINING))
                else:
                    print('Done: {0}/{1} training findings.'.format(i, N_TRAINING))

        training_output_dir = os.path.join(OUTPUT_DIRECTORY, "train")
        if not os.path.isdir(training_output_dir):
            os.mkdir(training_output_dir)

        if EXPOSURE_CORRECTION:
            train_tra_nda = correct_exposure(train_tra_nda)
            train_sag_nda = correct_exposure(train_sag_nda)
            train_cor_nda = correct_exposure(train_cor_nda)
        np.save(os.path.join(training_output_dir, "tra.npy"), train_tra_nda)
        np.save(os.path.join(training_output_dir, "sag.npy"), train_sag_nda)
        np.save(os.path.join(training_output_dir, "cor.npy"), train_cor_nda)
        np.save(os.path.join(training_output_dir, "adc.npy"), train_adc_nda)
        np.save(os.path.join(training_output_dir, "bval.npy"), train_bval_nda)
        np.save(os.path.join(training_output_dir, "ktrans.npy"), train_ktrans_nda)
        np.save(os.path.join(training_output_dir, "abk.npy"), adc_bval_ktrans_nda)
        np.save(os.path.join(training_output_dir, "clinsig.npy"), train_clinsig_arr)
        np.save(os.path.join(training_output_dir, "anatomical.npy"), train_anatomical_arr)
        np.save(os.path.join(training_output_dir, "zone_encoding.npy"), train_zone_encoding_arr)

        aug_factor = 2 if FLIP_AUG else 1
        with open(os.path.join(training_output_dir, 'uids.txt'), 'w') as output_file:
            for i in range(aug_factor):
                for uid in train_uids:
                    output_file.write("%s\n" % uid)

        with open(os.path.join(training_output_dir, 'clinsig.txt'), 'w') as output_file:
            for i in range(aug_factor):
                for train_clin_sig in train_clin_sigs:
                    output_file.write("%s\n" % str(train_clin_sig))

        with open(os.path.join(training_output_dir, 'anatomical.txt'), 'w') as output_file:
            for i in range(aug_factor):
                for anatomical in train_anatomical_arr:
                    output_file.write("%s\n" % str(anatomical))

        with open(os.path.join(training_output_dir, 'zone_encoding.txt'), 'w') as output_file:
            for i in range(aug_factor):
                for train_zone_encoding in train_zone_encoding_arr:
                    output_file.write("%s\n" % str(train_zone_encoding))

        with open(os.path.join(training_output_dir, 'zones.txt'), 'w') as output_file:
            for i in range(aug_factor):
                for sample_zone in train_zones:
                    output_file.write("%s\n" % str(sample_zone))

        with open(os.path.join(training_output_dir, 'transformations.txt'), 'w') as output_file:
            for i in range(aug_factor):
                for transformation in transformations:
                    output_file.write("%s\n" % transformation)

        # Calculate and save mean of training images
        mean_train_tra = np.mean(train_tra_nda, axis=0)
        std_train_tra = np.std(train_tra_nda, axis=0)
        mean_train_sag = np.mean(train_sag_nda, axis=0)
        std_train_sag = np.std(train_sag_nda, axis=0)
        mean_train_cor = np.mean(train_cor_nda, axis=0)
        std_train_cor = np.std(train_cor_nda, axis=0)
        mean_train_adc = np.mean(train_adc_nda, axis=0)
        std_train_adc = np.std(train_adc_nda, axis=0)
        mean_train_bval = np.mean(train_bval_nda, axis=0)
        std_train_bval = np.std(train_bval_nda, axis=0)
        mean_train_ktrans = np.mean(train_ktrans_nda, axis=0)
        std_train_ktrans = np.std(train_ktrans_nda, axis=0)
        mean_train_abk = np.concatenate((mean_train_adc, mean_train_bval, mean_train_ktrans), 0)
        std_train_abk = np.concatenate((std_train_adc, std_train_bval, std_train_ktrans), 0)

        mean_train_output_dir = os.path.join(OUTPUT_DIRECTORY, "mean_train")
        if not os.path.isdir(mean_train_output_dir):
            os.mkdir(mean_train_output_dir)

        std_train_output_dir = os.path.join(OUTPUT_DIRECTORY, "std_train")
        if not os.path.isdir(std_train_output_dir):
            os.mkdir(std_train_output_dir)

        np.save(os.path.join(mean_train_output_dir, "mean_train_tra.npy"), mean_train_tra)
        np.save(os.path.join(mean_train_output_dir, "mean_train_sag.npy"), mean_train_sag)
        np.save(os.path.join(mean_train_output_dir, "mean_train_cor.npy"), mean_train_cor)
        np.save(os.path.join(mean_train_output_dir, "mean_train_adc.npy"), mean_train_adc)
        np.save(os.path.join(mean_train_output_dir, "mean_train_bval.npy"), mean_train_bval)
        np.save(os.path.join(mean_train_output_dir, "mean_train_ktrans.npy"), mean_train_ktrans)
        np.save(os.path.join(mean_train_output_dir, "mean_train_abk.npy"), mean_train_abk)

        np.save(os.path.join(std_train_output_dir, "std_train_tra.npy"), std_train_tra)
        np.save(os.path.join(std_train_output_dir, "std_train_sag.npy"), std_train_sag)
        np.save(os.path.join(std_train_output_dir, "std_train_cor.npy"), std_train_cor)
        np.save(os.path.join(std_train_output_dir, "std_train_adc.npy"), std_train_adc)
        np.save(os.path.join(std_train_output_dir, "std_train_bval.npy"), std_train_bval)
        np.save(os.path.join(std_train_output_dir, "std_train_ktrans.npy"), std_train_ktrans)
        np.save(os.path.join(std_train_output_dir, "std_train_abk.npy"), std_train_abk)

        print("number of failed cases:", len(failed_cases))
        print(failed_cases)

    '''
    #############################################################################################
    VALIDATION DATA
    #############################################################################################
    '''

    if CREATE_VALIDATION:
        validation_file = os.path.join(PERMUTATION_DIRECTORY, "validation.txt")
        with open(validation_file) as f:
            validation_subjects = f.read().splitlines()
        images_table_path = os.path.join(s.sheets_folder, "images.csv")
        images_df = pd.read_csv(images_table_path)

        findings_table_path = os.path.join(s.sheets_folder, "findings.csv")
        findings_df = pd.read_csv(findings_table_path)
        findings_df = findings_df[findings_df.zone.isin(SECTORS)]
        #
        # keras data shape. Read more: https://keras.io/backend/
        validation_tra_nda, validation_sag_nda, validation_cor_nda, validation_bval_nda, validation_adc_nda, validation_ktrans_nda, \
        validation_adc_bval_ktrans_nda, validation_clinsig_arr, validation_anatomical_arr, \
        validation_zone_encoding_arr = create_empty_arrays(N_VALIDATION, IMAGE_DATA_DIM)

        validation_uids = []
        validation_clin_sigs = []
        validation_zones = []
        transformations = []
        failed_cases = []
        for i in range(N_VALIDATION):
            patient_id, fid, sample_zone, sample_clin_sig = get_patient_id_fid(findings_df=findings_df,
                                                                               subjects=validation_subjects)
            tra_nda, sag_nda, cor_nda, adc_nda, bval_nda, ktrans_nda, uid, translation, rotation = get_finding_ndas(
                patient_id, fid, images_df, mode="validation")

            tra_nda, sag_nda, cor_nda, adc_nda, bval_nda, ktrans_nda = process_ndas(tra_nda, sag_nda, cor_nda, adc_nda, bval_nda, ktrans_nda)
            validation_tra_nda[i], validation_sag_nda[i], validation_cor_nda[i], validation_adc_nda[i], \
            validation_bval_nda[i], \
            validation_ktrans_nda[i] = tra_nda, sag_nda, cor_nda, adc_nda, bval_nda, ktrans_nda

            if np.amax(adc_nda) > 0 and np.amax(bval_nda) > 0 and np.amax(ktrans_nda) > 0:
                validation_adc_bval_ktrans_nda[i, 0] = adc_nda
                validation_adc_bval_ktrans_nda[i, 1] = bval_nda
                validation_adc_bval_ktrans_nda[i, 2] = ktrans_nda
            else:
                warnings.warn('validation_adc_bval_ktrans_nda all zeors: {}, {}'.format(patient_id, fid))

            selected_anatomical_df = anatomical_df[(anatomical_df.ProxID == patient_id) & (anatomical_df.fid == fid)]
            relative_x = selected_anatomical_df.RelativeX.values[0]
            relative_y = selected_anatomical_df.RelativeY.values[0]
            relative_z = selected_anatomical_df.RelativeZ.values[0]
            validation_anatomical_arr[i] = np.array([relative_x, relative_y, relative_z])

            # one hot encoding zone information
            if sample_zone == 'PZ':
                zone_encoding = np.array([1, 0, 0])
            if sample_zone == 'TZ':
                zone_encoding = np.array([0, 1, 0])
            if sample_zone == 'AS':
                zone_encoding = np.array([0, 0, 1])
            validation_zone_encoding_arr[i] = zone_encoding

            validation_clinsig_arr[i] = bool(sample_clin_sig)
            validation_uids.append(uid)
            validation_zones.append(sample_zone)
            validation_clin_sigs.append(str(sample_clin_sig))
            transformation = []
            transformation.extend(translation)
            transformation.extend(rotation)
            transformations.append(transformation)
            if i % 20 == 0:
                print('Done: {0}/{1} validation findings.'.format(i, N_VALIDATION))

        validation_output_dir = os.path.join(OUTPUT_DIRECTORY, "validation")
        if not os.path.isdir(validation_output_dir):
            os.mkdir(validation_output_dir)

        # Todo: unsure that if applying the exposure correction separately for training and validation is a good idea.
        if EXPOSURE_CORRECTION:
            validation_tra_nda = correct_exposure(validation_tra_nda)
            validation_sag_nda = correct_exposure(validation_sag_nda)
            validation_cor_nda = correct_exposure(validation_cor_nda)
        np.save(os.path.join(validation_output_dir, "tra.npy"), validation_tra_nda)
        np.save(os.path.join(validation_output_dir, "sag.npy"), validation_sag_nda)
        np.save(os.path.join(validation_output_dir, "cor.npy"), validation_cor_nda)
        np.save(os.path.join(validation_output_dir, "adc.npy"), validation_adc_nda)
        np.save(os.path.join(validation_output_dir, "bval.npy"), validation_bval_nda)
        np.save(os.path.join(validation_output_dir, "ktrans.npy"), validation_ktrans_nda)
        np.save(os.path.join(validation_output_dir, "abk.npy"), validation_adc_bval_ktrans_nda)
        np.save(os.path.join(validation_output_dir, "clinsig.npy"), validation_clinsig_arr)
        np.save(os.path.join(validation_output_dir, "anatomical.npy"), validation_anatomical_arr)
        np.save(os.path.join(validation_output_dir, "zone_encoding.npy"), validation_zone_encoding_arr)

        with open(os.path.join(validation_output_dir, 'uids.txt'), 'w') as output_file:
            for uid in validation_uids:
                output_file.write("%s\n" % uid)

        with open(os.path.join(validation_output_dir, 'clinsig.txt'), 'w') as output_file:
            for validation_clin_sig in validation_clin_sigs:
                output_file.write("%s\n" % str(validation_clin_sig))

        with open(os.path.join(validation_output_dir, 'anatomical.txt'), 'w') as output_file:
            for anatomical in validation_anatomical_arr:
                output_file.write("%s\n" % str(anatomical))

        with open(os.path.join(validation_output_dir, 'zone_encoding.txt'), 'w') as output_file:
            for validation_zone_encoding in validation_zone_encoding_arr:
                output_file.write("%s\n" % str(validation_zone_encoding))

        with open(os.path.join(validation_output_dir, 'zones.txt'), 'w') as output_file:
            for sample_zone in validation_zones:
                output_file.write("%s\n" % str(sample_zone))

        with open(os.path.join(validation_output_dir, 'transformations.txt'), 'w') as output_file:
            for transformation in transformations:
                output_file.write("%s\n" % transformation)
                # Todo: Add the part for test data generation
    '''
    #############################################################################################
    VALIDATION ORIGINAL DATA
    #############################################################################################
    '''

    if CREATE_VALIDATION_ORIGINAL:
        validation_file = os.path.join(PERMUTATION_DIRECTORY, "validation.txt")
        with open(validation_file) as f:
            validation_subjects = f.read().splitlines()
        images_table_path = os.path.join(s.sheets_folder, "images.csv")
        images_df = pd.read_csv(images_table_path)

        findings_table_path = os.path.join(s.sheets_folder, "findings.csv")
        findings_df = pd.read_csv(findings_table_path)
        validation_finding_df = findings_df[(findings_df.ProxID.isin(validation_subjects))]
        validation_finding_df.sort_values(by=['ProxID', 'fid'], ascending=[True, True])
        n_validation_original = len(validation_finding_df.index)

        validation_tra_nda, validation_sag_nda, validation_cor_nda, validation_bval_nda, validation_adc_nda, validation_ktrans_nda, \
        validation_adc_bval_ktrans_nda, validation_clinsig_arr, validation_anatomical_arr, \
        validation_zone_encoding_arr = create_empty_arrays(n_validation_original, IMAGE_DATA_DIM)

        validation_uids = []
        validation_clin_sigs = []
        validation_zones = []
        transformations = []
        failed_cases = []
        i = 0
        for index, row in validation_finding_df.iterrows():
            patient_id = row.ProxID
            sample_zone = row.zone
            fid = row.fid
            clin_sig = bool(row.ClinSig)
            sample_clin_sig = 1 if clin_sig else 0
            tra_nda, sag_nda, cor_nda, adc_nda, bval_nda, ktrans_nda, uid, translation, rotation = get_finding_ndas(
                patient_id, fid, images_df, mode="validation_original")

            tra_nda, sag_nda, cor_nda, adc_nda, bval_nda, ktrans_nda = process_ndas(tra_nda, sag_nda, cor_nda, adc_nda, bval_nda, ktrans_nda)
            validation_tra_nda[i], validation_sag_nda[i], validation_cor_nda[i], validation_adc_nda[i], \
            validation_bval_nda[i], \
            validation_ktrans_nda[i] = tra_nda, sag_nda, cor_nda, adc_nda, bval_nda, ktrans_nda

            if np.amax(adc_nda) > 0 and np.amax(bval_nda) > 0 and np.amax(ktrans_nda) > 0:
                validation_adc_bval_ktrans_nda[i, 0] = adc_nda
                validation_adc_bval_ktrans_nda[i, 1] = bval_nda
                validation_adc_bval_ktrans_nda[i, 2] = ktrans_nda
            else:
                warnings.warn('validation_adc_bval_ktrans_nda all zeors: {}, {}'.format(patient_id, fid))

            selected_anatomical_df = anatomical_df[(anatomical_df.ProxID == patient_id) & (anatomical_df.fid == fid)]
            relative_x = selected_anatomical_df.RelativeX.values[0]
            relative_y = selected_anatomical_df.RelativeY.values[0]
            relative_z = selected_anatomical_df.RelativeZ.values[0]
            validation_anatomical_arr[i] = np.array([relative_x, relative_y, relative_z])

            # one hot encoding zone information
            if sample_zone == 'PZ':
                zone_encoding = np.array([1, 0, 0])
            if sample_zone == 'TZ':
                zone_encoding = np.array([0, 1, 0])
            if sample_zone == 'AS':
                zone_encoding = np.array([0, 0, 1])
            validation_zone_encoding_arr[i] = zone_encoding

            validation_clinsig_arr[i] = bool(sample_clin_sig)
            validation_uids.append(uid)
            validation_zones.append(sample_zone)
            validation_clin_sigs.append(str(sample_clin_sig))
            transformation = []
            transformation.extend(translation)
            transformation.extend(rotation)
            transformations.append(transformation)
            if i % 20 == 0:
                print('Done: {0}/{1} validation findings.'.format(i, n_validation_original))
            i += 1

        validation_output_dir = os.path.join(OUTPUT_DIRECTORY, "validation_original")
        if not os.path.isdir(validation_output_dir):
            os.mkdir(validation_output_dir)

        # Todo: unsure that if applying the exposure correction separately for training and validation is a good idea.
        if EXPOSURE_CORRECTION:
            validation_tra_nda = correct_exposure(validation_tra_nda)
            validation_sag_nda = correct_exposure(validation_sag_nda)
            validation_cor_nda = correct_exposure(validation_cor_nda)
        np.save(os.path.join(validation_output_dir, "tra.npy"), validation_tra_nda)
        np.save(os.path.join(validation_output_dir, "sag.npy"), validation_sag_nda)
        np.save(os.path.join(validation_output_dir, "cor.npy"), validation_cor_nda)
        np.save(os.path.join(validation_output_dir, "adc.npy"), validation_adc_nda)
        np.save(os.path.join(validation_output_dir, "bval.npy"), validation_bval_nda)
        np.save(os.path.join(validation_output_dir, "ktrans.npy"), validation_ktrans_nda)
        np.save(os.path.join(validation_output_dir, "abk.npy"), validation_adc_bval_ktrans_nda)
        np.save(os.path.join(validation_output_dir, "clinsig.npy"), validation_clinsig_arr)
        np.save(os.path.join(validation_output_dir, "anatomical.npy"), validation_anatomical_arr)
        np.save(os.path.join(validation_output_dir, "zone_encoding.npy"), validation_zone_encoding_arr)

        with open(os.path.join(validation_output_dir, 'uids.txt'), 'w') as output_file:
            for uid in validation_uids:
                output_file.write("%s\n" % uid)

        with open(os.path.join(validation_output_dir, 'clinsig.txt'), 'w') as output_file:
            for validation_clin_sig in validation_clin_sigs:
                output_file.write("%s\n" % str(validation_clin_sig))

        with open(os.path.join(validation_output_dir, 'anatomical.txt'), 'w') as output_file:
            for anatomical in validation_anatomical_arr:
                output_file.write("%s\n" % str(anatomical))

        with open(os.path.join(validation_output_dir, 'zone_encoding.txt'), 'w') as output_file:
            for validation_zone_encoding in validation_zone_encoding_arr:
                output_file.write("%s\n" % str(validation_zone_encoding))

        with open(os.path.join(validation_output_dir, 'zones.txt'), 'w') as output_file:
            for sample_zone in validation_zones:
                output_file.write("%s\n" % str(sample_zone))

        with open(os.path.join(validation_output_dir, 'transformations.txt'), 'w') as output_file:
            for transformation in transformations:
                output_file.write("%s\n" % transformation)
                # Todo: Add the part for test data generation

    '''
    #############################################################################################
    TEST DATA
    #############################################################################################
    '''

    if CREATE_TEST:
        images_table_path = os.path.join(s.sheets_folder, "images_test.csv")
        images_df = pd.read_csv(images_table_path)

        findings_table_path = os.path.join(s.sheets_folder, "findings_test.csv")
        findings_df = pd.read_csv(findings_table_path)
        # findings_df.sort_values(by=['ProxID', 'fid'], ascending=[True, True])
        n_test = len(findings_df.index)

        test_tra_nda, test_sag_nda, test_cor_nda, test_bval_nda, test_adc_nda, test_ktrans_nda, \
        test_adc_bval_ktrans_nda, test_clinsig_arr, test_anatomical_arr, \
        test_zone_encoding_arr = create_empty_arrays(n_test, IMAGE_DATA_DIM)

        test_uids = []
        test_clin_sigs = []
        test_zones = []
        transformations = []
        failed_cases = []
        i = 0
        for index, row in findings_df.iterrows():
            patient_id = row.ProxID
            sample_zone = row.zone
            fid = row.fid
            tra_nda, sag_nda, cor_nda, adc_nda, bval_nda, ktrans_nda, uid, translation, rotation = get_finding_ndas(
                patient_id, fid, images_df, mode="test")

            tra_nda, sag_nda, cor_nda, adc_nda, bval_nda, ktrans_nda = \
                process_ndas(tra_nda, sag_nda, cor_nda, adc_nda, bval_nda, ktrans_nda)
            test_tra_nda[i], test_sag_nda[i], test_cor_nda[i], test_adc_nda[i], \
            test_bval_nda[i], test_ktrans_nda[i] = tra_nda, sag_nda, cor_nda, adc_nda, bval_nda, ktrans_nda

            if np.amax(adc_nda) > 0 and np.amax(bval_nda) > 0 and np.amax(ktrans_nda) > 0:
                test_adc_bval_ktrans_nda[i, 0] = adc_nda
                test_adc_bval_ktrans_nda[i, 1] = bval_nda
                test_adc_bval_ktrans_nda[i, 2] = ktrans_nda
            else:
                warnings.warn('test_adc_bval_ktrans_nda all zeors: {}, {}'.format(patient_id, fid))

            # FIXME: Anatomical information for test is not available at this time. @alireza s
            # selected_anatomical_df = anatomical_df[(anatomical_df.ProxID == patient_id) & (anatomical_df.fid == fid)]
            relative_x = 0
            relative_y = 0
            relative_z = 0
            test_anatomical_arr[i] = np.array([relative_x, relative_y, relative_z])

            # one hot encoding zone information
            if sample_zone == 'PZ':
                zone_encoding = np.array([1, 0, 0])
            if sample_zone == 'TZ':
                zone_encoding = np.array([0, 1, 0])
            if sample_zone == 'AS':
                zone_encoding = np.array([0, 0, 1])
            test_zone_encoding_arr[i] = zone_encoding

            test_uids.append(uid)
            test_zones.append(sample_zone)
            transformation = []
            transformation.extend(translation)
            transformation.extend(rotation)
            transformations.append(transformation)
            if i % 20 == 0:
                print('Done: {0}/{1} test findings.'.format(i, n_test))
            i += 1

        test_output_dir = os.path.join(OUTPUT_DIRECTORY, "test")
        if not os.path.isdir(test_output_dir):
            os.mkdir(test_output_dir)

        # Todo: unsure that if applying the exposure correction separately for training and validation is a good idea.
        if EXPOSURE_CORRECTION:
            test_tra_nda = correct_exposure(test_tra_nda)
            test_sag_nda = correct_exposure(test_sag_nda)
            test_cor_nda = correct_exposure(test_cor_nda)
        np.save(os.path.join(test_output_dir, "tra.npy"), test_tra_nda)
        np.save(os.path.join(test_output_dir, "sag.npy"), test_sag_nda)
        np.save(os.path.join(test_output_dir, "cor.npy"), test_cor_nda)
        np.save(os.path.join(test_output_dir, "adc.npy"), test_adc_nda)
        np.save(os.path.join(test_output_dir, "bval.npy"), test_bval_nda)
        np.save(os.path.join(test_output_dir, "ktrans.npy"), test_ktrans_nda)
        np.save(os.path.join(test_output_dir, "abk.npy"), test_adc_bval_ktrans_nda)
        np.save(os.path.join(test_output_dir, "clinsig.npy"), test_clinsig_arr)
        np.save(os.path.join(test_output_dir, "anatomical.npy"), test_anatomical_arr)
        np.save(os.path.join(test_output_dir, "zone_encoding.npy"), test_zone_encoding_arr)

        with open(os.path.join(test_output_dir, 'uids.txt'), 'w') as output_file:
            for uid in test_uids:
                output_file.write("%s\n" % uid)

        with open(os.path.join(test_output_dir, 'clinsig.txt'), 'w') as output_file:
            for test_clin_sig in test_clin_sigs:
                output_file.write("%s\n" % str(test_clin_sig))

        with open(os.path.join(test_output_dir, 'anatomical.txt'), 'w') as output_file:
            for anatomical in test_anatomical_arr:
                output_file.write("%s\n" % str(anatomical))

        with open(os.path.join(test_output_dir, 'zone_encoding.txt'), 'w') as output_file:
            for test_zone_encoding in test_zone_encoding_arr:
                output_file.write("%s\n" % str(test_zone_encoding))

        with open(os.path.join(test_output_dir, 'zones.txt'), 'w') as output_file:
            for sample_zone in test_zones:
                output_file.write("%s\n" % str(sample_zone))

        with open(os.path.join(test_output_dir, 'transformations.txt'), 'w') as output_file:
            for transformation in transformations:
                output_file.write("%s\n" % transformation)
                # Todo: Add the part for test data generation