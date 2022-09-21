import numpy as np
import cv2
import albumentations as A
import matplotlib.pyplot as plt

from trafficlight import TrafficLight

import glob
import os
import random

def test_infer():
    img_dir = "../test_tflight_2_eval/bbox/"
    img_list = glob.glob(os.path.join(img_dir, '*'))
    # random.shuffle(img_list)

    for img in img_list:
        input = cv2.imread(img)

        trafficlight = TrafficLight()
        trafficlight.load_model('trafficlight.onnx')

        bbox_coords = [[0, 0, input.shape[1] - 1, input.shape[0] - 1]]

        labels, confs = trafficlight.predict(input, bbox_coords)

        plot_img = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)[
            bbox_coords[0][0]:bbox_coords[0][3], 
            bbox_coords[0][1]:bbox_coords[0][2]
        ]

        _confs = confs[0]
        print(img)
        plt.imshow(plot_img)
        plt.title("Predict: {}, Confidences: {:.4f} {:.4f} {:.4f}".format(labels[0], _confs[0], _confs[1], _confs[2]))
        plt.show()

test_infer()