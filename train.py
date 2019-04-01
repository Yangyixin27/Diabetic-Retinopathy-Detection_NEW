import os
import sys
import time

from data_generator import NeedleData
from model import CNNModel
from project.settings import intermediate_dir
from cnn import unet_1
import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu0')

if __name__ == '__main__':
    SPLIT=5
    data_dir = os.path.join(intermediate_dir, "numpy")
    
    for i in range(SPLIT):
        i += 1
        model, model_id, padding_axis1, padding_axis2 = unet_1.model(l2_constant=0, lr=1e-4)
        ds = NeedleData(padding_axis1, padding_axis2)
        #ds.save_train_validation_data()    #coment this later
        cnn = CNNModel(data_streamer= ds, model= model)
        uid = model_id + "br=4_lr=1e-4_bs=128_flipping"
        cnn.train(uid=uid, batch_size=128, nb_epoch=1000, split_num=i, balance = True, balance_ratio=4, fliplr = True)