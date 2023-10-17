'''
@author: Junwei Yu
@contact : yuju@tcd.ie
@file: unet.py
@time: 2023/7/28 12:33
'''

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

def build_unet(input_shape=(256,256,3)):
    # transfer learning, use the pretrained model as the backbone.
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)

    # get the skip connection layers.
    block1 = base_model.get_layer('conv1_relu').output          #64   * 256 * 256
    block2 = base_model.get_layer('conv2_block2_out').output    #256  * 128 * 128
    block3 = base_model.get_layer('conv3_block4_out').output    #512  * 64  * 64
    block4 = base_model.get_layer('conv4_block6_out').output    #1024 * 32  * 32
    block5 = base_model.get_layer('conv5_block3_out').output    #2048 * 16  * 16

    x = UpSampling2D((2, 2))(block5)
    x = Conv2D(1024, (1, 1), activation='relu')(x)  # add 1x1 conv to adjust channel number
    x = Concatenate()([x, block4])
    x1 = Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
    output1 = Conv2D(3, (1, 1), activation='softmax')(x1)

    x = UpSampling2D((2, 2))(x1)
    x = Conv2D(512, (1, 1), activation='relu')(x)  # add 1x1 conv to adjust channel number
    x = Concatenate()([x, block3])
    x2 = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    output2 = Conv2D(3, (1, 1), activation='softmax')(x2)

    x = UpSampling2D((2, 2))(x2)
    x = Conv2D(256, (1, 1), activation='relu')(x)  # add 1x1 conv to adjust channel number
    x = Concatenate()([x, block2])
    x3 = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    output3 = Conv2D(3, (1, 1), activation='softmax')(x3)

    x = UpSampling2D((2, 2))(x3)
    x = Conv2D(64, (1, 1), activation='relu')(x)  # add 1x1 conv to adjust channel number
    x = Concatenate()([x, block1])
    x4 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    output4 = Conv2D(3, (1, 1), activation='softmax')(x4)

    x = UpSampling2D((2, 2))(x4)
    output = Conv2D(3, (1, 1), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=[output, output1, output2, output3, output4])

    return model

if __name__ == "__main__":
    build_unet()