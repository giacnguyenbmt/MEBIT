# file name
# function
# author, version
import os
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .base import BaseEvaluation
from .utils import util, coco_util
from .utils.util import HiddenPrints

class ODetEvaluation(BaseEvaluation):
    valid_option_list = [
        "blurring",
        "increasing_brightness",
        "increasing_contrast",
        "decreasing_brightness",
        "decreasing_contrast",
        "down_scale",
        "crop",
        "rotate90",
        # "left_rotation", # To-do option
        # "right_rotation",
        # "compactness",
    ]

    @classmethod
    def from_input_path(cls, 
                        img_path, 
                        gt_path, 
                        image_color='rgb'):
        # Read image
        data = util.read_image(img_path, image_color)

        # Read ground truth
        with HiddenPrints():
            gt = COCO(gt_path)

        instance = cls(data, gt, image_color)

        instance.bboxes = []
        for key in list(instance.gt.anns.keys()):
            ann = instance.gt.anns[key]
            category_id = ann['category_id']
            cat_name = gt.cats[category_id]['name']
            instance.bboxes.append(
                ann['bbox'] + [(category_id, cat_name)]
            )

        instance.img_path = img_path
        instance.gt_path = gt_path

        return instance

    """
    # This func is used in order to read input for 
    # left/right rotation and compactness option
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
    """

    def save_gt(self, gt, img_names):
        dataset = gt.dataset
        for i, img_name in enumerate(img_names):
            dataset['images'][i]['file_name'] = img_name

        json_file = os.path.split(self.img_path)[-1]
        json_name = os.path.splitext(json_file)[0]

        # get type: lastpoint or deadpoint
        type_status = os.path.splitext(img_name)[0][-9:]
        json_path = '{}_{}_{}.json'.format(
            json_name,
            self.option,
            type_status
        )

        _path = os.path.join(
            self.result_image_path,
            'gt',
            json_path
        )

        json_content = json.dumps(dataset)
        with open(_path, 'w') as file:
            file.write(json_content)
    
    def save_dt(self, dt, img_names):
        json_file = os.path.split(self.img_path)[-1]
        json_name = os.path.splitext(json_file)[0]

        # get type: lastpoint or deadpoint
        img_name = img_names[0]
        type_status = os.path.splitext(img_name)[0][-9:]
        
        json_path = '{}_{}_{}.json'.format(
            json_name,
            self.option,
            type_status
        )
        
        _path = os.path.join(
            self.result_image_path,
            'dt',
            json_path
        )

        dt_list = dt.dt_list
        json_content = json.dumps(dt_list)
        with open(_path, 'w') as file:
            file.write(json_content)

    def evaluate(self, gt, dt):
        # run evaluation
        with HiddenPrints():
            cocoeval = COCOeval(gt, dt, iouType="bbox")
            cocoeval.evaluate()
            cocoeval.accumulate()
            cocoeval.summarize()
        metric = {
            'ap' : cocoeval.stats[0],
            'ap_50' : cocoeval.stats[1],
            'ap_75' : cocoeval.stats[2]
        }
        return metric

    def create_original_input(self):
        data = [self.data]
        gt = self.gt

        return data, gt

    def format_transformed_gt(self, *args, **kwargs):
        raw_gt = args[0]
        data = kwargs.get('data', None)
        assert (data is not None), "Missing data argument"

        if self.report[self.option]['type'] == 1:
            gt = self.gt
        else:
            coco_format = coco_util.albu_to_coco_dict(data, raw_gt)
            gt = coco_util.create_cocogt(coco_format)

        return gt

    # 
    def format_dt(self, *args, **kwargs):
        """
        
        """
        results = kwargs.get('results', None)
        gt = kwargs.get('gt', None)
        assert (None not in [results, gt]), "Lack arguments"

        # if option is crop
        dt_list = []
        for i, rs in enumerate(results):
            img_infos = gt.dataset['images'][i]
            img_id = img_infos['id']
            for j, poly in enumerate(rs['boxes']):
                instance_det = {
                    "image_id": img_id,
                    "category_id": int(rs['classes'][j]),
                    "bbox": poly.astype(float).tolist(),
                    "score": float(rs['scores'][j])
                }
                dt_list.append(instance_det)
        with HiddenPrints():
            dt = gt.loadRes(dt_list)
            dt.dt_list = dt_list

        return dt
