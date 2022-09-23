import os

from mmocr.utils.ocr import MMOCR
import cv2

from ..text_detection import TDetEvaluation

def inference_function(input):
    img = cv2.cvtColor(input, cv2.COLOR_RGB2BGR)
    cv2.imwrite(temp_img_path, img)
    
    predicted_sample = ocr.readtext(temp_img_path, output='demo/det_out.jpg')
    return predicted_sample

def convert_output_function(predicted_sample):
    result = predicted_sample[0]['boundary_result']
    predicted_bbox = [i[:8] for i in result]
    predicted_confidence = [i[8] for i in result]
    formated_sample = {'boxes': predicted_bbox,
                       'confidences': predicted_confidence}
    return formated_sample

img = 'MEBIT/data/text_detection/img_16.jpg'
gt = 'MEBIT/data/text_detection/gt_img_16.txt'

save_dir_ = 'demo'
temp_img_path = os.path.join(save_dir_, 'my_image.jpg')
if os.path.isdir(save_dir_) is False:
    os.makedirs(save_dir_)

option = ["blurring", 
          "increasing_brightness", 
          "increasing_contrast",
          "decreasing_brightness", 
          "decreasing_contrast", 
          "down_scale",
          "crop"]

# Load models into memory
ocr = MMOCR(recog=None)

foo = TDetEvaluation(img, gt)
# foo.tdet_stats(inference_function, 
#                convert_output_function, 
#                option[6],
#                'ap',
#                0.5,
#                verbose=True)
foo.tdet_stats(inference_function, 
               convert_output_function, 
               option[0],
               'precision',
               0.5,
               verbose=True)