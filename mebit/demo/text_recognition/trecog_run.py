import os

from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import cv2

from ...text_recognition import TRecogEvaluation

def inference_function(input):
    img = cv2.cvtColor(input, cv2.COLOR_RGB2BGR)
    temp_path = 'temp.jpg'
    cv2.imwrite(temp_path, img)
    img = Image.open(temp_path)
    predicted_sample = detector.predict(img)
    if os.path.isfile(temp_path):
        os.remove(temp_path)
    return predicted_sample

def convert_output_function(predicted_sample):
    # print(predicted_sample)
    return predicted_sample

img = 'data/text_recognition/word_1.png'
gt = 'data/text_recognition/treg_gt.txt'
option = ["blurring",
          "increasing_brightness",
          "increasing_contrast",
          "decreasing_brightness",
          "decreasing_contrast",
          "down_scale",
          "crop",
          "rotate90",]

config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
config['cnn']['pretrained']=False
config['device'] = 'cuda:0'
config['predictor']['beamsearch']=False

detector = Predictor(config)

foo = TRecogEvaluation.from_input_path(img, gt)

for i in range(len(option)):
    foo.stats(inference_function,
              convert_output_function,
              option[i],
              'accuracy',
              0.5,
              "result_folder",
              True,
              verbose=True)