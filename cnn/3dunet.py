from keras.layers import Input, merge, Convolution3D, MaxPooling3D, UpSampling3D, Cropping3D  # , Dropout
from keras.layers import Input, merge, BatchNormalization, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
# from module.utils import dice_coef, dice_coef_loss
import os

import cv2
import numpy as np

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, ReduceLROnPlateau, EarlyStopping
from project.settings import cases_root, slicer_dir, intermediate_dir

# choose gpu0/1
import theano.sandbox.cuda

theano.sandbox.cuda.use('gpu1')

# os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu1,floatX=float32"

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

IMGS_TRAIN = os.path.join(intermediate_dir, "patchfour", "2", "train_img_fourped.npy")
IMGS_MASK_TRAIN = os.path.join(intermediate_dir, "patchfour", "2", "train_label_fourped.npy")
#IMGS_TRAIN = os.path.join(intermediate_dir, "patchfour", "1", "train_img_four_flip.npy")
#IMGS_MASK_TRAIN = os.path.join(intermediate_dir, "patchfour", "1", "train_label_four_flip.npy")
IMGS_VAL = os.path.join(intermediate_dir, "patchfour", "2","val_img_fourped.npy")
IMGS_LABEL_VAL = os.path.join(intermediate_dir, "patchfour", "2", "val_label_fourped.npy")

imgs_train_arr = np.load(IMGS_TRAIN)
imgs_mask_train_arr = np.load(IMGS_MASK_TRAIN)
imgs_val_arr = np.load(IMGS_VAL)
imgs_label_val_arr = np.load(IMGS_LABEL_VAL)

imgs_mask_train_arr = imgs_mask_train_arr.astype('float32')
imgs_mask_train_arr /= 255.

imgs_label_val_arr = imgs_label_val_arr.astype('float32')
imgs_label_val_arr /= 255.

image_depth = 24
image_rows = 80 #40 #160
image_columns = 92 #40 #180

n = imgs_train_arr.shape[0]
m = len(imgs_val_arr)
imgs_train = np.empty((n, 1, image_depth + 20, image_rows + 40, image_columns + 40), dtype=np.float32)
imgs_mask_train = np.empty((n, 1, image_depth, image_rows, image_columns), dtype=np.float32)
imgs_val = np.empty((m, 1, image_depth + 20, image_rows + 40, image_columns + 40), dtype=np.float32)
imgs_label_val = np.empty((m, 1, image_depth, image_rows, image_columns), dtype=np.float32)

for i in range(n):
    imgs_train[i, 0, :, :, :] = imgs_train_arr[i, :, :, :]
    imgs_mask_train[i, 0, :, :, :] = imgs_mask_train_arr[i, :, :, :]
for i in range(m):
    imgs_val[i, 0, :, :, :] = imgs_val_arr[i, :, :, :]
    imgs_label_val[i, 0, :, :, :] = imgs_label_val_arr[i, :, :, :]

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def get_unet(lr=1e-6):  # , l2_constant=0.002
    # model_id = "unet3d_same"
    image_depth = 44  # 38
    image_rows = 120 #80 #200  # 140
    image_columns = 132 #80 #220  # 140

    inputs = Input((1, image_depth, image_rows, image_columns))
    conv1 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='valid')(inputs)  # ,W_regularizer=l2(l2_constant)
    conv1 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='valid')(conv1)  # ,,W_regularizer=l2(l2_constant)
    pool1 = MaxPooling3D(pool_size=(1, 2, 2), padding='valid')(conv1)  # strides=(2, 2, 2),

    conv2 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='valid')(pool1)  # W_regularizer=l2(l2_constant),
    conv2 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='valid')(conv2)
    pool2 = MaxPooling3D(pool_size=(1, 2, 2), padding='valid')(conv2)  # strides=(2, 2, 2),

    conv3 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='valid')(pool2)  # W_regularizer=l2(l2_constant),
    conv3 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='valid')(conv3)  # W_regularizer=l2(l2_constant),

    conv2_cropped = Cropping3D(((2, 2), (4, 4), (4, 4)))(conv2)
    up4 = merge([UpSampling3D(size=(1, 2, 2))(conv3), conv2_cropped], mode='concat', concat_axis=1)
    conv4 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='valid')(up4)  # W_regularizer=l2(l2_constant),
    conv4 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='valid')(conv4)  # W_regularizer=l2(l2_constant),

    conv1_cropped = Cropping3D(((6, 6), (16, 16), (16, 16)))(conv1)
    up5 = merge([UpSampling3D(size=(1, 2, 2))(conv4), conv1_cropped], mode='concat', concat_axis=1)
    conv5 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='valid')(up5)  # W_regularizer=l2(l2_constant),
    conv5 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='valid')(conv5)  # W_regularizer=l2(l2_constant),

    conv6 = Convolution3D(1, 1, 1, 1, activation='sigmoid')(conv5)

    model = Model(input=inputs, output=conv6)
    model.compile(optimizer=Adam(lr=lr), loss=dice_coef_loss, metrics=[dice_coef])

    return model  # , model_id


def train_and_predict():
    global imgs_train, imgs_mask_train, imgs_val, imgs_label_val
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    model = get_unet()
    # model.summary()
    model_checkpoint = ModelCheckpoint('unet3d_valid_patch_four22.hdf5', monitor='val_dice_coef', mode='max', save_best_only=True)
    csv_logger = CSVLogger('training_unet3d_valid_patch_four22.log')
    reduce_lr = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.05, patience=5, verbose=0, mode='max', epsilon=0.0001, cooldown=0, min_lr=0)
    early_stopping = EarlyStopping(monitor='val_dice_coef', patience=3, mode='max')
    
    print('-' * 30)
    print('Loading and preprocessing test data...')
    print('-' * 30)

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)
    
    callbacks_list = [model_checkpoint, csv_logger, reduce_lr ,early_stopping]

    model.fit(imgs_train, imgs_mask_train, batch_size=4, nb_epoch=50, verbose=1, shuffle=True,
              callbacks=callbacks_list, validation_data=(imgs_val, imgs_label_val))

    #     print('-'*30)
    #     print('Loading saved weights...')
    #     print('-'*30)
    #     model.load_weights('unet3d_valid_patch_four.hdf5')

    print('-' * 30)
    print('Predicting masks on test data...')
    print('-' * 30)
    imgs_mask_test = model.predict(imgs_val, batch_size=4, verbose=1)  
    np.save('unet3d_valid_patch_four22.npy', imgs_mask_test)

if __name__ == '__main__':
    train_and_predict()
