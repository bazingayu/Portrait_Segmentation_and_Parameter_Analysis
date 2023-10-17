'''
@author: Junwei Yu
@contact: yuju@tcd.ie
@file : metrics.py
@time: 2023/07/28 
'''

from tensorflow.keras import metrics
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
def get_metrics(metric_type='MeanIoU', num_classes=3):
    if metric_type == 'MeanIoU':
        return metrics.MeanIoU(num_classes=num_classes)
    else:
        raise ValueError('Unsupported metric: {}'.format(metric_type))

def class_wise_iou(y_true, y_pred, num_classes):
    class_wise_iou_list = []
    smoothening_factor = 0.00001
    for i in range(num_classes):
        intersection = tf.reduce_sum(tf.cast((y_true == i) & (y_pred == i), dtype=tf.float32))
        union = tf.reduce_sum(tf.cast((y_true == i) | (y_pred == i), dtype=tf.float32))
        iou = (intersection + smoothening_factor) / (union + smoothening_factor)
        class_wise_iou_list.append(iou)
    return class_wise_iou_list


class WeightedMeanIoU(tf.keras.metrics.Metric):

    def __init__(self, num_classes=3, name="weighted_mean_iou", **kwargs):
        super(WeightedMeanIoU, self).__init__(name=name, **kwargs)
        self.mean_iou = tf.keras.metrics.MeanIoU(num_classes=num_classes)
        self.total_iou = self.add_weight(name="total_iou", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Only compute IoU where sample_weight != 0
        mask = tf.cast(sample_weight, tf.bool)

        y_true_indices = tf.argmax(y_true, axis=-1)
        y_pred_indices = tf.argmax(y_pred, axis=-1)

        y_true = tf.boolean_mask(y_true_indices, mask)
        y_pred = tf.boolean_mask(y_pred_indices, mask)

        self.mean_iou.update_state(y_true, y_pred)
        self.total_iou.assign_add(self.mean_iou.result())
        self.count.assign_add(1.0)

    def result(self):
        return self.total_iou / self.count

    def reset_states(self):
        self.mean_iou.reset_states()  # reset the states of the inner MeanIoU metric
        self.total_iou.assign(0.0)  # reset the total IoU
        self.count.assign(0.0)  # reset the count



def calculate_evalutation_metrics(predicted_masks, true_masks):
    """
    Calculates the mean Intersection over Union (IoU) for classes 0, 1, 2 and various metrics for class 0.
    """
    total_iou = [0, 0, 0]
    inter = [0, 0, 0]
    uni = [0, 0, 0]
    tn, fp, fn, tp = 0, 0, 0, 0
    smoothening_factor = 0.00001

    for predicted_mask, true_mask in zip(predicted_masks, true_masks):

        # Calculate IoU for each class

        for i in range(3):
            intersection = 0
            union = 0
            for m in range(true_mask.shape[0]):
                for n in range(true_mask.shape[1]):
                    if (true_mask[m, n] == 3):
                        continue
                    if(true_mask[m, n] == i and predicted_mask[m, n] == i):
                        intersection += 1
                    if (true_mask[m, n] == i or predicted_mask[m, n] == i):
                        union += 1

            #
            inter[i] += intersection
            uni[i] += union



        # Calculate other metrics for class 0

        binary_true = []
        binary_pred = []
        for i in range(true_mask.shape[0]):
            for j in range(true_mask.shape[1]):
                if(true_mask[i, j] == 3):
                    continue
                binary_pred.append(predicted_mask[i, j] == 0)
                binary_true.append(true_mask[i, j] == 0)

        TN, FP, FN, TP = confusion_matrix(binary_true, binary_pred, labels=[0, 1]).ravel()

        tn += TN; fp+= FP; fn+=FN; tp+=TP

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0


    iou = [inter[i] / uni[i] for i in range(3)]

    result = {
        'iou': iou,
        'mean_iou': sum(iou)/3,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    print(result)

    return result

# {'iou': [0.7726709594530521, 0.9889613793494276, 0.9977415732451711], 'mean_iou': 0.9197913040158836, 'mean_precision': 0.8020767178966086, 'mean_recall': 0.9547009354625342, 'mean_f1': 0.8717590315705915}
