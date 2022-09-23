import os

from pycocotools.coco import COCO
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score

from .base import BaseEvaluation
from .utils.util import HiddenPrints
from .utils import util

class ClsfEvaluation(BaseEvaluation):
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
        with open(gt_path, 'r') as file:
            gt_file = file.read().replace('\n', '')
            transcriptions_list = [gt_file]
            gt = transcriptions_list

        instance = cls(data, gt, image_color)

        instance.img_path = img_path
        instance.gt_path = gt_path

        return instance

    @classmethod
    def from_coco_input_path(cls, 
                             img_path, 
                             gt_path, 
                             image_color='rgb'):
        # Read image
        data = util.read_image(img_path, image_color)

        # Read ground truth
        with HiddenPrints():
            gt = COCO(gt_path)

        instance = cls(data, gt, image_color)
        key = list(instance.gt.anns.keys())[0]
        ann = instance.gt.anns[key]
        category_id = ann['category_id']
        cat_name = gt.cats[category_id]['name']
        instance.bboxes.append(ann['bbox'] + [cat_name])


        instance.img_path = img_path
        instance.gt_path = gt_path

        return instance

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

    def create_original_input(self):
        if self.option in ['left_rotation', 'right_rotation', 'compactness']:
            x, y, w, h, cat_ = self.bboxes[0]
            x, y, w, h = map(int, [x, y, w, h])
            data = [self.data[y:y + h, x:x + w]]
            gt = [cat_]
        else:
            data = [self.data]
            gt = self.gt
        return data, gt

    def format_transformed_gt(self, *args, **kwargs):
        if 'data' in kwargs:
            num_record = len(kwargs['data'])
        else:
            num_record = 1
        if self.option in ['left_rotation', 'right_rotation', 'compactness']:
            gt = self.bboxes[0][-1:] * num_record
        else:
            gt = self.gt * num_record
        return gt

    def format_dt(self, *args, **kwargs):
        if 'results' in kwargs:
            dt = kwargs['results']
        else:
            dt = None
        return dt
