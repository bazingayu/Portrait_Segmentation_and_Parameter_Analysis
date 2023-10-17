'''
@author: Junwei Yu
@contact: yuju@tcd.ie
@file : losses.py
@time: 2023/07/28 
'''
from tensorflow.keras import losses
from tensorflow.image import resize
import tensorflow as tf
import tensorflow.keras.backend as K

def get_loss(loss_type='cross_entropy'):
    if loss_type == 'cross_entropy':
        # return the cross entropy loss
        return losses.CategoricalCrossentropy()
    else:
        raise ValueError('Unsupported loss function: {}'.format(loss_type))


def dice_loss(target, pred):

    pred = tf.transpose(pred, (0, 3, 1, 2))
    target = tf.transpose(target, (0, 3, 1, 2))

    smooth = 1.0
    iflat = tf.reshape(pred, (tf.shape(pred)[0], tf.shape(pred)[1], -1))  # batch, channel, -1
    tflat = tf.reshape(target, (tf.shape(target)[0], tf.shape(target)[1], -1))
    intersection = K.sum(iflat * tflat, axis=-1)
    return 1 - ((2.0 * intersection + smooth)) / (K.sum(iflat, axis=-1) + K.sum(tflat, axis=-1) + smooth)



def combine_loss(y_true, y_pred, weights):
    main_output, output1, output2, output3, output4 = y_pred

    main_loss_ce = losses.CategoricalCrossentropy()(y_true, main_output, sample_weight=weights)
    main_loss = main_loss_ce

    # downsample
    y_true1 = resize(y_true, size=tf.shape(output1)[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    y_true2 = resize(y_true, size=tf.shape(output2)[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    y_true3 = resize(y_true, size=tf.shape(output3)[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    y_true4 = resize(y_true, size=tf.shape(output4)[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    weights = tf.expand_dims(weights, axis=-1)  # Add a new axis at the end
    weights1 = resize(weights, size=tf.shape(output1)[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    weights2 = resize(weights, size=tf.shape(output2)[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    weights3 = resize(weights, size=tf.shape(output3)[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    weights4 = resize(weights, size=tf.shape(output4)[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # additional loss
    loss1 = losses.CategoricalCrossentropy()(y_true1, output1, sample_weight=weights1)
    loss2 = losses.CategoricalCrossentropy()(y_true2, output2, sample_weight=weights2)
    loss3 = losses.CategoricalCrossentropy()(y_true3, output3, sample_weight=weights3)
    loss4 = losses.CategoricalCrossentropy()(y_true4, output4, sample_weight=weights4)

    return main_loss+ 0.1 * loss1 + 0.1 * loss2 + 0.1 * loss3 + 0.1 * loss4
