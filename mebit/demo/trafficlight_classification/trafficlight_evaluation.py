import pandas as pd

import glob
import os
import sys
import json

from .trafficlight import TrafficLight
from ...classification import ClsfEvaluation

option = ["blurring", 
          "increasing_brightness", 
          "increasing_contrast",
          "decreasing_brightness", 
          "decreasing_contrast", 
          "down_scale",
          "crop"]

def inference_function(input):
    bbox_coords = [[0, 0, input.shape[1] - 1, input.shape[0] - 1]]
    labels, confs = trafficlight.predict(input, bbox_coords)
    result_storage[image][opt] = {
        'label': labels[0], 
        'confs': list(confs[0].astype(float))
        }
    return (labels, confs)

def convert_function(predicted_sample):
    return predicted_sample[0][0]

if __name__ == "__main__":
    image_dir = sys.argv[1]
    gt_dir = sys.argv[2]
    image_type = sys.argv[3]
    result_folder = sys.argv[4]

    trafficlight = TrafficLight()
    trafficlight.load_model('mebit/demo/trafficlight_classification/model/trafficlight.onnx')
    df = pd.DataFrame(columns = ["image"] + option)

    image_list = glob.glob(os.path.join(image_dir, "*." + image_type))

    result_storage = {}

    for i, image in enumerate(image_list):
        result_storage[image] = {}
        file_name_ = os.path.split(image)[-1]
        gt_name = os.path.join(gt_dir, 
                               os.path.splitext(file_name_)[0] + '.txt')
        evaluation = ClsfEvaluation.from_input_path(img_path=image, gt_path=gt_name, image_color='bgr')


        new_record = {"image": image}
        for opt in option:
            result = evaluation.stats(inference_function, 
                                           convert_function,
                                           opt,
                                           "accuracy",
                                           0.5,
                                           result_folder,
                                           verbose=False)
            new_record[opt] = result['value']

        df = pd.concat([df, pd.DataFrame.from_records([new_record])], ignore_index=True)

    df.to_csv(os.path.join(result_folder, "result.csv"))
    json_data = json.dumps(result_storage, indent=4)
    with open(os.path.join(result_folder, 'json_result.json'), 'w') as file:
        file.write(json_data)
    print(df)


