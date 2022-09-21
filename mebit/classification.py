import os

import cv2
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score

from .base import BaseEvaluation
from .utils.util import HiddenPrints

class ClsfEvaluation(BaseEvaluation):
    valid_option_list = [
        "blurring", 
        "increasing_brightness", 
        "increasing_contrast", 
        "decreasing_brightness", 
        "decreasing_contrast", 
        "down_scale", 
        "crop",
    ]

    @classmethod
    def from_input_path(cls, 
                        img_path, 
                        gt_path, 
                        image_color='rgb'):
        # Read image
        BGR_img = cv2.imread(img_path)
        if image_color == 'rgb':
            img = cv2.cvtColor(BGR_img, cv2.COLOR_BGR2RGB)
        elif image_color == 'bgr':
            img = BGR_img
        data = img

        # Read ground truth
        with open(gt_path, 'r') as file:
            gt_file = file.read().replace('\n', '')
            transcriptions_list = [gt_file]
            gt = transcriptions_list

        return cls(data, gt, image_color)

    @classmethod
    def from_coco_input_path(cls, 
                             img_path, 
                             gt_path, 
                             image_color='rgb'):
        # Read image
        BGR_img = cv2.imread(img_path)
        if image_color == 'rgb':
            img = cv2.cvtColor(BGR_img, cv2.COLOR_BGR2RGB)
        elif image_color == 'bgr':
            img = BGR_img
        data = img

        # Read ground truth
        with open(gt_path, 'r') as file:
            gt_file = file.read().replace('\n', '')
            transcriptions_list = [gt_file]
            gt = transcriptions_list

        return cls(data, gt, image_color)

    def save_gt(self, gt, img_names):
        for i, img_name in enumerate(img_names):
            _name, _ = os.path.splitext(img_name)
            _new_name = _name + '.txt'
            _path = os.path.join(
                self.result_image_path,
                'gt',
                _new_name
            )
            with open(_path, 'w') as f:
                f.write(gt[i])
    
    def save_dt(self, dt, img_names):
        for i, img_name in enumerate(img_names):
            _name, _ = os.path.splitext(img_name)
            _new_name = _name + '.txt'
            _path = os.path.join(
                self.result_image_path,
                'dt',
                _new_name
            )
            with open(_path, 'w') as f:
                f.write(dt[i])

    def evaluate(self, gt, dt):
        metric = {
            "accuracy": accuracy_score(gt, dt),
            "precision": precision_score(gt, dt, average='micro'),
            "recall": recall_score(gt, dt, average='micro'),
            "f1": f1_score(gt, dt, average='micro')
        }
        return metric

    # def read_groundtruth(self):
    #     with open(self.gt_path, 'r') as file:
    #         gt_file = file.read().replace('\n', '')
    #         self.transcriptions_list = [gt_file]

    def format_original_gt(self, *args, **kwargs):
        gt = self.gt
        return gt

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
