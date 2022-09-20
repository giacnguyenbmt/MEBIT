import os
import json

import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO

from .util import HiddenPrints


def find_bbox(polygon):
    x = min([polygon[i] for i in range(0, len(polygon), 2)])
    y = min([polygon[i] for i in range(1, len(polygon), 2)])
    width = max([polygon[i] for i in range(0, len(polygon), 2)]) - x
    height = max([polygon[i] for i in range(1, len(polygon), 2)]) - y
    return [x, y, width, height]

def find_area(segmentation, height, width):
    objs = mask.frPyObjects(segmentation, height, width)
    area = mask.area(objs)
    return float(area[0])

def text_infos_to_coco_dict(img_path, gt, width, height):
    image_infos = {}

    image_infos['images'] = [
        {
            'id': 1,
            'width': width,
            'height': height,
            'file_name': img_path
        }
    ]

    points_list = gt['points_list']
    transcriptions_list = gt['transcriptions_list']
    anns = []
    id = 1
    for index, polygon in enumerate(points_list): 
        if transcriptions_list[index] != '###':
            ann = {
                "id": id, 
                "image_id": 1, 
                "category_id": 1, 
                "segmentation": [polygon], 
                "area": find_area([polygon], height, width),
                "bbox": find_bbox(polygon), 
                "iscrowd": 0,
            }
            id += 1
            anns.append(ann)
    image_infos['annotations'] = anns

    image_infos['categories'] = [
        {'id': 1, 'name': 'text'}
    ]
    return image_infos

def albu_to_coco_dict(data, raw_gt):
    image_infos = {
        'images': [],
        'annotations': [],
        'categories': []
    }

    ann_id = 1
    for img_id, img in enumerate(data):
        image_infos['images'].append(
            {
                'id': img_id + 1,
                'width': img.shape[1],
                'height': img.shape[0],
                'file_name': "data/alb_{}.jpg".format(img_id + 1)
            }
        )

        for transformed_mask in raw_gt[img_id]['masks']:
            mask_rle = mask.encode(np.asfortranarray(transformed_mask))
            mask_rle['counts'] = mask_rle['counts'].decode('ascii')
            ann = {
                "id": ann_id,
                "image_id": img_id + 1,
                "category_id": 1,
                "segmentation": mask_rle,
                "area": float(mask.area(mask_rle)),
                "bbox": mask.toBbox(mask_rle).tolist(),
                "iscrowd": 0,
            }
            image_infos['annotations'].append(ann)
            ann_id += 1

    image_infos['categories'] = [
        {'id': 1, 'name': 'text'}
    ]
    return image_infos

def create_cocogt(coco_format):
    # convert to COCO class
    annotation_file = 'temp_gt.json'
    json_content = json.dumps(coco_format)
    with open(annotation_file, 'w') as file:
        file.write(json_content)
    with HiddenPrints():
        cocoGt=COCO(annotation_file)
    os.remove(annotation_file)
    return cocoGt