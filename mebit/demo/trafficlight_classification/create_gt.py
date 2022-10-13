from trafficlight import TrafficLight

import sys
import os
import glob

import cv2

if __name__ == "__main__":
    image_dir = sys.argv[1]
    
    image_list = glob.glob(os.path.join(image_dir, '*'))

    gt_dir = os.path.join(image_dir, '../gt')
    if not os.path.isdir(gt_dir):
        os.mkdir(gt_dir)

    trafficlight = TrafficLight()
    trafficlight.load_model('trafficlight.onnx')

    for i, image_path in enumerate(image_list):
        image = cv2.imread(image_path)

        bbox_coords = [[0, 0, image.shape[1] - 1, image.shape[0] - 1]]
        
        labels, confs = trafficlight.predict(image, bbox_coords)

        image_name = os.path.splitext(os.path.split(image_path)[-1])[0]
        gt_name = image_name + '.txt'
        # gt_name = os.path.split(image_path)[-1][:-len(image_type)] + 'txt'
        with open(os.path.join(gt_dir, gt_name), 'w') as file:
            file.write(labels[0])