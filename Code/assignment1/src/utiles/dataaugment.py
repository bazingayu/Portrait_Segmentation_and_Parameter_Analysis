'''
@author: Junwei Yu
@contact: yuju@tcd.ie
@file : dataaugment.py
@time: 2023/07/29 
'''
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.image import resize
from tensorflow.keras.utils import to_categorical

class DataAugmenter:
    def __init__(self, rotation_range, brightness_range, contrast_range, noise_stddev):
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_stddev = noise_stddev

    def random_rotate(self, img, mask):
        """Randomly rotate image and mask."""
        # Convert one-hot mask to class labels
        # mask_labels = tf.argmax(mask, axis=-1)

        random_angles = tf.random.uniform(shape=(), minval=-self.rotation_range, maxval=self.rotation_range)
        img = tfa.image.rotate(img, random_angles, fill_value=0)

        mask = tfa.image.rotate(mask, random_angles, fill_value=3)  # Assume the last class label is 3

        return img, mask


    def random_crop_and_resize(self, img, mask):
        """Randomly crop and resize image and mask."""
        crop_size = tf.random.uniform(shape=(), minval=180, maxval=256, dtype=tf.int32)  # 随机选择裁剪大小

        masktype = mask.dtype
        # match the type
        mask = tf.cast(mask, img.dtype)

        # add one channel to match img
        mask = tf.expand_dims(mask, axis=-1)

        # merge
        combined = tf.concat([img, mask], axis=-1)

        # random crop
        crop_size_combined = tf.convert_to_tensor([crop_size, crop_size, 4])
        combined_crop = tf.image.random_crop(combined, size=crop_size_combined)

        img_cropped = combined_crop[:, :, :3]
        mask_cropped = combined_crop[:, :, 3]
        mask_cropped = tf.expand_dims(mask_cropped, axis=-1)

        # adjust size
        img = tf.image.resize(img_cropped, [256, 256])

        # adjust size with nearest neighbor and transfer the data type back
        mask = tf.image.resize(mask_cropped, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mask = tf.squeeze(mask, axis=-1)
        mask = tf.cast(mask, masktype)

        return img, mask

    def color_jitter(self, img):
        img_hsv = tf.image.rgb_to_hsv(img)
        hue_delta = tf.random.uniform(shape=(), minval=-0.01, maxval=0.01)  # hue
        saturation_factor = tf.random.uniform(shape=(), minval=0.9, maxval=1.1)  # saturation
        img_hsv = tf.image.adjust_hue(img_hsv, hue_delta)
        img_hsv = tf.image.adjust_saturation(img_hsv, saturation_factor)
        img_rgb = tf.image.hsv_to_rgb(img_hsv)
        return img_rgb

    def color_enhance(self, img):
        """Enhance the brightness and contrast of the image."""
        img = tf.image.random_brightness(img, max_delta=self.brightness_range)
        img = tf.image.random_contrast(img, lower=1-self.contrast_range, upper=1+self.contrast_range)
        return img

    def add_noise(self, img):
        """Add random noise to the image."""
        noise = tf.random.normal(shape=tf.shape(img), stddev=self.noise_stddev)
        img = img + noise
        img = tf.clip_by_value(img, 0., 1.)  # Ensure pixel values stay in range
        return img

    def __call__(self, img, mask):
        """Apply data augmentation."""
        if tf.random.uniform([]) < 0.5:  # 50% probability
            img, mask = self.random_rotate(img, mask)

        # Perform random crop and resize
        if tf.random.uniform([]) < 0.5:  # 50% probability
            img, mask = self.random_crop_and_resize(img, mask)

        # Perform color enhancement on the image only
        # if tf.random.uniform([]) < 0.0:  # 0% probability
        #     img = self.color_enhance(img)
        #
        # # Add random noise to the image only
        # if tf.random.uniform([]) < 0.0:  # 0% probability
        #     img = self.add_noise(img)
        #
        #
        # if tf.random.uniform([]) < 0.0:  # 0% probability
        #     img = self.color_jitter(img)

        return img, mask


