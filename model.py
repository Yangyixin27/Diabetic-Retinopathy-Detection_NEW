from __future__ import print_function

import json
import os
import time
import sys
import numpy as np
from keras.callbacks import ModelCheckpoint, History
sys.path.append("..")
from project.settings import intermediate_dir


class CNNModel(object):
    def __init__(self, data_streamer, model):
        self.ds = data_streamer
        self.model = model

    def train(self, uid, batch_size, nb_epoch, split_num, balance = False, balance_ratio = 0, fliplr = False):
        if not os.path.exists(os.path.join(intermediate_dir, 'model_checkpoints', uid)):
            os.mkdir(os.path.join(intermediate_dir, 'model_checkpoints', uid))
        
        print('-' * 30)
        print('Loading train and validation data...')
        print('-' * 30)
        train_slices_arrs, train_slices_mask_arrs, val_slices_arrs, val_slices_mask_arrs = self.ds.load_train_validation_data(split_num, balance,                                                                                       balance_ratio, fliplr)
        
        print('-' * 30)
        print('Creating and compiling model...')
        print('-' * 30)
        model_checkpoint_dir = os.path.join(intermediate_dir, 'model_checkpoints', uid, "{}".format(split_num))  
        if not os.path.exists(model_checkpoint_dir):
            os.mkdir(model_checkpoint_dir)
        model_checkpoint = ModelCheckpoint(os.path.join(model_checkpoint_dir, 'model_checkpoint.hdf5'), monitor='loss', save_best_only=True)

        print('-' * 30)
        print('Fitting model...')
        print('-' * 30)
        history = History()
        self.model.fit(train_slices_arrs[:], train_slices_mask_arrs[:],
                  batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
                  shuffle=True, callbacks=[model_checkpoint, history],
                  validation_data= (train_slices_arrs[:], train_slices_mask_arrs[:]))
            
        #self.model.evaluate(val_slices_arr[:], val_slices_mask_arr[:], batch_size=32, verbose=1)

        with open(os.path.join(model_checkpoint_dir, 'model.json'), 'w') as output:
            json.dump(self.model.to_json(), output)

        history_dictionary = dict()
        history_dictionary['past'] = ''
        history_dictionary['epoch'] = history.epoch
        history_dictionary['history'] = history.history
        history_dictionary['params'] = history.params

        with open(os.path.join(model_checkpoint_dir, 'history.json'), 'w') as output:
            json.dump(history_dictionary, output)

    def resume_train(self, uid, batch_size, nb_epoch, split_num):
        print('-' * 30)
        print('Loading train and validation data...')
        print('-' * 30)
        train_slices_arrs, train_slices_mask_arrs, val_slices_arrs, val_slices_mask_arrs = self.ds.load_train_validation_data(split_num)

        model_checkpoint_dir = os.path.join(intermediate_dir, 'model_checkpoints', uid, "{}".format(split_num))
        model_checkpoint_file = os.path.join(model_checkpoint_dir, 'model_checkpoint.hdf5')
        model_checkpoint = ModelCheckpoint(os.path.join(model_checkpoint_dir, 'model_checkpoint.hdf5'), monitor='loss', save_best_only=True)
        self.model.load_weights(model_checkpoint_file)

        print('-' * 30)
        print('Resume Previous Training... Fitting model...')
        print('-' * 30)
        history = History()
        self.model.fit(train_slices_arrs[:], train_slices_mask_arrs[:],
                  batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
                  shuffle=True, callbacks=[model_checkpoint, history],
                  validation_data=(val_slices_arrs[:], val_slices_mask_arrs[:]))

        history_dictionary = dict()
        history_dictionary['past'] = uid
        history_dictionary['epoch'] = history.epoch
        history_dictionary['history'] = history.history
        history_dictionary['params'] = history.params
        
        with open(os.path.join(model_checkpoint_dir, 'model.json'), 'w') as output:
            json.dump(self.model.to_json(), output)
        
        uid = time.strftime("%Y_%m_%d_%H_%M_%S")
        with open(os.path.join(model_checkpoint_dir, uid + '_history.json'), 'w') as output:
            json.dump(history_dictionary, output)

    def predict_validation(self, uid, split_num, batch_size=32):    ##need to be modified
        _, _, val_slices_arrs, val_slices_mask_arrs = self.ds.load_train_validation_data(split_num)
        model_checkpoint_dir = os.path.join(intermediate_dir, 'model_checkpoints', uid, "{}".format(split_num))
        model_checkpoint_file = os.path.join(model_checkpoint_dir, 'model_checkpoint.hdf5')
        print("Checkpoint Path: ", model_checkpoint_file)
        self.model.load_weights(model_checkpoint_file)
        val_slices_prediction = self.model.predict(val_slices_arrs, verbose=1)
        dice_loss, dice_coef = self.model.evaluate(val_slices_arrs, val_slices_mask_arrs, batch_size, verbose=1)
        print("Validation Prediction Shape: {}".format(val_slices_prediction.shape))
        return dice_loss, dice_coef, val_slices_prediction, val_slices_mask_arrs

    def predict_test(self, uid, split_num, batch_size=32):
        test_slices_arrs, test_slices_mask_arrs = self.ds.load_test_data(split_num)
        model_checkpoint_dir = os.path.join(intermediate_dir, 'model_checkpoints', uid, "{}".format(split_num))
        model_checkpoint_file = os.path.join(model_checkpoint_dir, 'model_checkpoint.hdf5')
        print("Checkpoint Path: ", model_checkpoint_file)
        self.model.load_weights(model_checkpoint_file)
        test_slices_prediction = self.model.predict(test_slices_arrs, verbose=1)
        dice_loss, dice_coef = self.model.evaluate(test_slices_arrs, test_slices_mask_arrs, batch_size, verbose=1)
        print("Testing Prediction Shape: {}".format(test_slices_prediction.shape))
        return dice_loss, dice_coef, test_slices_prediction, test_slices_mask_arrs
