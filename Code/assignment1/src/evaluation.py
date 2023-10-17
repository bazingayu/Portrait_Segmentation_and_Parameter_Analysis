'''
@author: Junwei Yu
@contact: yuju@tcd.ie
@file : evaluation.py
@time: 2023/07/31 
'''
import sys
sys.path.append('../')
sys.path.append('./')
from tensorflow.keras.models import load_model
from src.utiles.losses import combine_loss
import tensorflow as tf
import numpy as np
import cv2
import os
from src.utiles.losses import combine_loss
from src.utiles.metrics import WeightedMeanIoU
from src.utiles.metrics import calculate_evalutation_metrics



import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Provide the path to the model.")
    parser.add_argument("--model_path", type=str, default='./src/saved_models/model_epoch_90_best_miou_0.9621.h5', help="Path to the model file.")
    args = parser.parse_args()
    return args.model_path

color_to_label = {
    (75, 25, 230): 0,    # Red, 1, glomerulus
    (0, 255, 0): 1,    # Green, 2, normal tissue
    (255, 255, 0): 2,    # Blue, 3, background
    (0, 0, 0): 3       # Black, 4, unlabeled data
}

mean_value = [125.40095372, 194.61185261, 173.29846314]
std_value = [69.33723314, 41.24116595, 46.72013978]

def main():
    model_path = parse_arguments()
    test_images_path = './data/test/images'
    test_mask_path = "./data/test/mask_ori"
    img_size = (256, 256)

    pred = []
    true = []

    with tf.device('/cpu:0'):
        model = tf.keras.models.load_model(model_path, custom_objects={'combine_loss': combine_loss, 'WeightedMeanIoU': WeightedMeanIoU})

        image_files = [os.path.join(test_images_path, f) for f in os.listdir(test_images_path) if
                       os.path.isfile(os.path.join(test_images_path, f))]

        for image_file in image_files:

            mask_file = os.path.join(test_mask_path, os.path.basename(image_file)[:-5] + "A.png")
            mask = cv2.imread(mask_file)
            mask_resized = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)

            mask_label = np.zeros(mask_resized.shape[:2], dtype=np.int32)
            for color, label in color_to_label.items():
                mask_label[(mask_resized == color).all(axis=-1)] = label

            img = cv2.imread(image_file)
            img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img1, img_size)
            # Normalize and expand the dimensions
            img_tensor = tf.convert_to_tensor(img_resized, dtype=tf.float32)

            # 使用tf.split将图像分解为三个单通道图像
            channels = tf.split(img_tensor, num_or_size_splits=3, axis=-1)

            for i in range(3):
                channels[i] = (channels[i] - mean_value[i]) / std_value[i]

            # 使用tf.stack将通道重新组合为归一化的图像
            img_tensor = tf.concat(channels, axis=-1)

            img_tensor = tf.expand_dims(img_tensor, 0)

            prediction = model.predict(img_tensor)[0][0]
            # pred_class_map = tf.argmax(prediction, axis=-1)
            # pred_rgb_map = map_class_to_rgb(pred_class_map)
            # concatenated_img = cv2.hconcat([img_resized, pred_rgb_map, mask_resized])
            # cv2.imshow("win", concatenated_img)
            # cv2.waitKey(0)

            # Convert the prediction to class labels
            pred_class_map = tf.argmax(prediction, axis=-1).numpy()
            pred.append(pred_class_map)
            true.append(mask_label)
        calculate_evalutation_metrics(pred, true)



def map_class_to_rgb(class_map):
    rgb_map = np.zeros((*class_map.shape, 3), dtype=np.uint8)

    # Define the mapping from class to RGB color
    label_to_color = {
        0: (75, 25, 230),  # Red, class 1
        1: (0, 255, 0),    # Green, class 2
        2: (255, 255, 0)   # Blue, class 3
    }

    # Map each class in the class_map to its corresponding color
    for class_id, color in label_to_color.items():
        rgb_map[class_map == class_id] = color

    return rgb_map


if __name__ == "__main__":
    main()
