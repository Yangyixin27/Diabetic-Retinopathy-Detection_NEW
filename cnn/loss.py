def dice_coef(y_true, y_pred):
    import keras.backend as K
    K.set_image_dim_ordering('th')
    
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_category(y_true, y_pred):
    import keras.backend as K
    K.set_image_dim_ordering('th')
    
    y_pred_category = y_pred >= 0.5
    y_pred_category = K.cast(y_pred_category, "float32")
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_category_f = K.flatten(y_pred_category)
    intersection = K.sum(y_true_f * y_pred_category_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_category_f) + smooth)


def dice_coef_category_numpy_collectionwise(y_true, y_pred):
    import numpy as np
    
    y_pred_category = y_pred >= 0.5
    y_pred_category = y_pred_category.astype(float)
    smooth = 1.
    y_true_f = y_true.flatten()
    y_pred_category_f = y_pred_category.flatten()
    intersection = np.sum(y_true_f * y_pred_category_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_category_f) + smooth)


def dice_coef_category_numpy_slicewise(y_true, y_pred):
    import numpy as np
    
    y_pred_category = y_pred >= 0.5
    y_pred_category = y_pred_category.astype(float)
    smooth = 1.
    dice_sum = 0
    num_slicers = y_true.shape[0]
    for i in range(num_slicers):
        y_true_f = y_true[i].flatten()
        y_pred_category_f = y_pred_category[i].flatten()
        intersection = np.sum(y_true_f * y_pred_category_f)
        dice_sum += (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_category_f) + smooth)
    return dice_sum/num_slicers
    
        
def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)