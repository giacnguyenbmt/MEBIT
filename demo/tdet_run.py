from mmocr.utils.ocr import MMOCR
import cv2

from ..mebit.metrics import Evaluation


def inference_function(input):
    img = cv2.cvtColor(input, cv2.COLOR_RGB2BGR)
    cv2.imwrite('demo/my_image.jpg', img)
    predicted_sample = ocr.readtext('demo/my_image.jpg', output='demo/det_out.jpg')
    return predicted_sample

def convert_output_function(predicted_sample):
    result = predicted_sample[0]['boundary_result']
    predicted_bbox = [i[:8] for i in result]
    predicted_confidence = [i[8] for i in result]
    formated_sample = {'boxes': predicted_bbox,
                       'confidences': predicted_confidence}
    return formated_sample


img = 'data/img_16.jpg'
gt = 'data/gt_img_16.txt'
option = ["blurring", 
          "increasing_brightness", 
          "increasing_contrast",
          "decreasing_brightness", 
          "decreasing_contrast", 
          "down_scale",
          "crop"]

# Load models into memory
ocr = MMOCR(recog=None)

foo = Evaluation(img, gt)
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