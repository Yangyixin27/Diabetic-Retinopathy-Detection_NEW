from keras.layers import Input, merge, Cropping2D, Convolution2D, MaxPooling2D, UpSampling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
K.set_image_dim_ordering('th')    #Theano Dimension Order
import sys

sys.path.append("..")
from cnn.loss import dice_coef, dice_coef_category, dice_coef_loss



def model(lr=1e-5, l2_constant=0):
    model_id = "unet_1"
    volume_rows = 44
    volume_cols = 200
    labelmap_rows = 24
    labelmap_cols = 160
    padding_axis1 = volume_rows - labelmap_rows
    padding_axis2 = volume_cols - labelmap_cols

    inputs = Input((1, volume_rows, volume_cols))
    conv1 = Convolution2D(32, 3, 3, activation='relu', W_regularizer=l2(l2_constant), border_mode='valid')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', W_regularizer=l2(l2_constant), border_mode='valid')(conv1)
    pool1 = MaxPooling2D(pool_size=(1, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', W_regularizer=l2(l2_constant), border_mode='valid')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', W_regularizer=l2(l2_constant), border_mode='valid')(conv2)
    pool2 = MaxPooling2D(pool_size=(1, 2))(conv2)
    
    conv3 = Convolution2D(128, 3, 3, activation='relu', W_regularizer=l2(l2_constant), border_mode='valid')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', W_regularizer=l2(l2_constant), border_mode='valid')(conv3)
    
    conv2_cropped = Cropping2D(((2, 2), (4, 4)))(conv2)
    up4 = merge([UpSampling2D(size=(1, 2))(conv3), conv2_cropped], mode='concat', concat_axis=1)
    conv4 = Convolution2D(64, 3, 3, activation='relu', W_regularizer=l2(l2_constant), border_mode='valid')(up4)
    conv4 = Convolution2D(64, 3, 3, activation='relu', W_regularizer=l2(l2_constant), border_mode='valid')(conv4)

    conv1_cropped = Cropping2D(((6, 6), (16, 16)))(conv1)
    up5= merge([UpSampling2D(size=(1, 2))(conv4), conv1_cropped], mode='concat', concat_axis=1)
    conv5= Convolution2D(32, 3, 3, activation='relu', W_regularizer=l2(l2_constant), border_mode='valid')(up5)
    conv5 = Dropout(0.5)(conv5)
    conv5= Convolution2D(32, 3, 3, activation='relu', W_regularizer=l2(l2_constant), border_mode='valid')(conv5)
    
    conv6 = Convolution2D(1, 1, 1, activation='sigmoid')(conv5)

    model = Model(input=inputs, output=conv6)
    print(model.summary())
    model.compile(optimizer=Adam(lr=lr), loss= dice_coef_loss, metrics=[dice_coef_category])

    return model, model_id, padding_axis1, padding_axis2

if __name__ == "__main__":
    model, model_id, padding_axis1, padding_axis2 = model()
    print(model.metrics_names)