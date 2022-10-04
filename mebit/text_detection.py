import os
import json

from pycocotools import mask
from pycocotools.cocoeval import COCOeval

from .base import BaseEvaluation
from .utils import util, coco_util
from .utils.util import HiddenPrints
from .metrics import rrc_evaluation_funcs_1_1 as rrc_evaluation_funcs
from .metrics import script

read_gt = rrc_evaluation_funcs.get_tl_line_values_from_file_contents

class TDetEvaluation(BaseEvaluation):
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
        with open(gt_path, 'r') as file:
            gt_file = file.read()
            points_list, _, transcriptions_list = read_gt(
                gt_file,
                CRLF=False,
                LTRB=False,
                withTranscription=True,
                withConfidence=False,
            )
        keypoints = [(points_list[j][i], points_list[j][i + 1])
                     for j in range(len(points_list)) 
                     for i in range(0, 8, 2)]
        gt = {
            'boxes': points_list,
            'texts': transcriptions_list
        }

        instance = cls(data, gt, image_color)

        instance.img_path = img_path
        instance.gt_path = gt_path
        instance.keypoints = keypoints

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
        if self.option == 'crop':
            dataset = gt.dataset
            for i, img_name in enumerate(img_names):
                dataset['images'][i]['file_name'] = img_name
            for i, ann in enumerate(dataset['annotations']):
                try:
                    ann = ann['segmentation']['counts'].decode('ascii')
                    dataset['annotations'][i] = ann
                except:
                    pass

            json_name = os.path.split(self.img_path)[-1]
            json_name = os.path.splitext(json_name)[0]

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

        else:
            for i, img_name in enumerate(img_names):
                _name, _ = os.path.splitext(img_name)
                _new_name = _name + '.txt'
                _path = os.path.join(
                    self.result_image_path,
                    'gt',
                    _new_name
                )

                contents = ''
                for i, poly in enumerate(gt['boxes']):
                    int_poly = [int(i) for i in poly]
                    oneline = '{},' * 8 + '{}\n'
                    contents += oneline.format(
                        *int_poly,
                        gt['texts'][i]
                    )
                
                with open(_path, 'w') as f:
                    f.write(contents)
    
    def save_dt(self, dt, img_names):
        if self.option == 'crop':
            img_name = img_names[0]
            dt_list = dt.dt_list
            new_dt_list = []
            for i, ann in enumerate(dt_list):
                seg = ann['segmentation']
                try:
                    seg["counts"] = seg["counts"].decode('ascii')
                except:
                    pass

                instance_seg = {
                    "image_id": ann["image_id"],
                    "category_id": ann["category_id"],
                    "segmentation": seg,
                    "score": ann["score"]
                }
                new_dt_list.append(instance_seg)
            
            json_name = os.path.split(self.img_path)[-1]
            json_name = os.path.splitext(json_name)[0]

            # get type: lastpoint or deadpoint
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

            json_content = json.dumps(new_dt_list)
            with open(_path, 'w') as file:
                file.write(json_content)

        else:
            for i, img_name in enumerate(img_names):
                _name, _ = os.path.splitext(img_name)
                _new_name = _name + '.txt'
                _path = os.path.join(
                    self.result_image_path,
                    'dt',
                    _new_name
                )

                contents = ''
                for i, poly in enumerate(dt['boxes']):
                    int_poly = [int(i) for i in poly]
                    oneline = '{},' * 7 + '{}\n'
                    contents += oneline.format(
                        *int_poly,
                    )
                
                with open(_path, 'w') as f:
                    f.write(contents)


    def evaluate(self, gt, dt):
        # if option is crop
        if self.report[self.option]['type'] == 2:
            # run evaluation
            with HiddenPrints():
                cocoeval = COCOeval(gt, dt)
                cocoeval.evaluate()
                cocoeval.accumulate()
                cocoeval.summarize()
            metric = {
                'ap' : cocoeval.stats[0],
                'ap_50' : cocoeval.stats[1],
                'ap_75' : cocoeval.stats[2]
            }
        else:
            alias_func = script.evaluate_method_per_sample
            metric = alias_func(gt, dt)
        return metric

    def create_original_input(self):
        data = [self.data]

        # if option is crop, convert bbox annotation to mask annotation
        if self.report[self.option]['type'] == 2:
            coco_format = coco_util.text_infos_to_coco_dict(
                self.img_path,
                self.gt,
                self.width,
                self.height
            )
            gt = coco_util.create_cocogt(coco_format)
            # create masks from corresponding polygons
            for id in gt.getAnnIds(imgIds=1):
                self.masks.append(gt.annToMask(gt.loadAnns(id)[0]))
        else:
            gt = self.gt

        return data, gt

    def format_transformed_gt(self, *args, **kwargs):
        raw_gt = args[0]
        data = kwargs.get('data', None)
        assert (data is not None), "Missing data argument"

        if self.option == 'crop':
            coco_format = coco_util.tdet_albu_to_coco_dict(
                    data, 
                    raw_gt
                )
            gt = coco_util.create_cocogt(coco_format)

        elif (self.option == "down_scale" 
        or self.report[self.option]['type'] == 4):
            keypoints = raw_gt[0]['keypoints']
            new_points_list = [[keypoints[i][0], 
                                keypoints[i][1],
                                keypoints[i + 1][0],
                                keypoints[i + 1][1],
                                keypoints[i + 2][0],
                                keypoints[i + 2][1],
                                keypoints[i + 3][0],
                                keypoints[i + 3][1]] 
                                for i in range(0, len(keypoints), 4)]
            gt = {
                'boxes': new_points_list,
                'texts': self.gt['texts']
            }

        else:
            gt = self.gt

        return gt

    def format_dt(self, *args, **kwargs):
        results = kwargs.get('results', None)
        gt = kwargs.get('gt', None)
        assert (None not in [results, gt]), "Lack arguments"

        # if option is crop
        if self.report[self.option]['type'] == 2:
            dt_list = []
            for i, rs in enumerate(results):
                img_infos = gt.dataset['images'][i]
                img_id = img_infos['id']
                h, w = img_infos['height'], img_infos['width']
                for j, poly in enumerate(rs['boxes']):
                    instance_seg = {
                        "image_id": img_id,
                        "category_id": 1,
                        "segmentation": mask.frPyObjects(
                            [poly], 
                            h, w)[0],
                        "score": rs['confidences'][j]
                    }
                    dt_list.append(instance_seg)
            with HiddenPrints():
                dt = gt.loadRes(dt_list)
                dt.dt_list = dt_list
        else:
            dt = results[0]

        return dt
