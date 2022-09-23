import os
import re

from .base import BaseEvaluation
from .metrics import trecog_metric as text_metric
from .utils import util

class TRecogEvaluation(BaseEvaluation):
    valid_option_list = [
        "blurring",
        "increasing_brightness",
        "increasing_contrast",
        "decreasing_brightness",
        "decreasing_contrast",
        "down_scale",
        "crop",
        "left_rotation",
        "right_rotation",
        "compactness",
    ]

    @classmethod
    def from_input_path(cls, 
                        img_path, 
                        gt_path, 
                        image_color='rgb'):
        # Read image
        data = util.read_image(img_path, image_color)

        # Read ground truth
        gt = []
        evaluationParams = {
            'SAMPLE_NAME_2_ID':'(?:word_)?([0-9]+).png',
            'CRLF':False,
            'DOUBLE_QUOTES':True
            }
        with open(gt_path, 'r') as file:
            gt_file = file.read()
        gtLines = gt_file.split("\r\n" if evaluationParams['CRLF'] else "\n")
        for line in gtLines:
            line = line.replace("\r","").replace("\n","")
            if(line != ""):
                if (evaluationParams['DOUBLE_QUOTES']):
                    m = re.search(r'"(.+)"',line)
                else:
                    m = re.search(r"'(.+)'",line)
                gt.append(m.group()[1:-1])

        instance = cls(data, gt, image_color)

        instance.img_path = img_path
        instance.gt_path = gt_path

        return instance

    @classmethod
    def from_extended_input_path(cls, 
                                 img_path, 
                                 gt_path, 
                                 image_color='rgb'):
        # Read image
        data = util.read_image(img_path, image_color)

        # Read ground truth
        ...
        instance = None

        instance.img_path = img_path
        instance.gt_path = gt_path

        return instance

    def save_gt(self, gt, img_names):
        for i, img_name in enumerate(img_names):
            txt_name = os.path.split(self.img_path)[-1]
            txt_name = os.path.splitext(txt_name)[0]

            # get type: lastpoint or deadpoint
            type_status = os.path.splitext(img_name)[0][-9:]
            txt_path = '{}_{}_{}.txt'.format(
                txt_name,
                self.option,
                type_status
            )

            _path = os.path.join(
                self.result_image_path,
                'gt',
                txt_path
            )
            with open(_path, 'w') as f:
                f.write('{}, "{}"'.format(img_name, gt[i]))
    
    def save_dt(self, dt, img_names):
        for i, img_name in enumerate(img_names):
            txt_name = os.path.split(self.img_path)[-1]
            txt_name = os.path.splitext(txt_name)[0]

            # get type: lastpoint or deadpoint
            type_status = os.path.splitext(img_name)[0][-9:]
            txt_path = '{}_{}_{}.txt'.format(
                txt_name,
                self.option,
                type_status
            )

            _path = os.path.join(
                self.result_image_path,
                'dt',
                txt_path
            )
            with open(_path, 'w') as f:
                f.write('{}, "{}"'.format(img_name, dt[i]))

    def evaluate(self, gt, dt):
        acc = text_metric.compute_accuracy(gt, dt)
        levenshtein_distance = text_metric.compute_levenshtein(gt, dt)
        metric = {
            "accuracy": acc,
            "levenshtein": levenshtein_distance
        }
        return metric

    def create_original_input(self):
        data = [self.data]
        gt = self.gt
        return data, gt

    def format_transformed_gt(self, *args, **kwargs):
        if 'data' in kwargs:
            num_record = len(kwargs['data'])
        else:
            num_record = 1
        gt = self.gt * num_record
        return gt

    def format_dt(self, *args, **kwargs):
        if 'results' in kwargs:
            dt = kwargs['results']
        else:
            dt = None
        return dt
