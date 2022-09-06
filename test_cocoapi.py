from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask
import cv2

import json
import sys
import os

import skimage.io as io
import matplotlib.pyplot as plt

import utils.rrc_evaluation_funcs_1_1 as rrc_evaluation_funcs

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# # Disable
# def blockPrint():
#     sys.stdout = open(os.devnull, 'w')

# # Restore
# def enablePrint():
#     sys.stdout = sys.__stdout__


def find_bbox(polygon):
    x = min([polygon[i] for i in range(0, len(polygon), 2)])
    y = min([polygon[i] for i in range(1, len(polygon), 2)])
    width = max([polygon[i] for i in range(0, len(polygon), 2)]) - x
    height = max([polygon[i] for i in range(1, len(polygon), 2)]) - y
    return [x, y, width, height]

def find_area(segmentation):
    objs = mask.frPyObjects(segmentation, height, width)
    area = mask.area(objs)
    return float(area[0])

def text_infos_to_coco_dict():
    image_infos = {}

    image_infos['images'] = [
        {
            'id': 1,
            'width': width,
            'height': height,
            'file_name': img_path
        }
    ]

    anns = []
    id = 1
    for index, polygon in enumerate(points_list): 
        if transcriptions_list[index] != '###':
            ann = {
                "id": id, 
                "image_id": 1, 
                "category_id": 1, 
                "segmentation": [polygon], 
                "area": find_area([polygon]),
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

def mmocr_result2coco(gt, res_file):
    h, w = gt['images'][0]['height'], gt['images'][0]['width']
    boundary_result = res_file['boundary_result']
    results = []
    for i, rs in enumerate(boundary_result):
        predict_seg = {
            "image_id": 1, 
            "category_id": 1, 
            "segmentation": mask.frPyObjects(
                [rs[:-1]], 
                h, w
            )[0], 
            "score": round(rs[-1], 3),
        }
        results.append(predict_seg)
    return results

if __name__ == "__main__":
    with HiddenPrints():
        # Define image and groundtruth path
        img_path = 'data/img_16.jpg'
        gt_path = 'data/gt_img_16.txt'

        # Read groundtruth file
        points_list = []
        transcriptions_list = []
        with open(gt_path, 'r') as file:
            gt_file = file.read()
            points_list, _, transcriptions_list = rrc_evaluation_funcs.get_tl_line_values_from_file_contents(
                gt_file,
                CRLF=False,
                LTRB=False,
                withTranscription=True,
                withConfidence=False,
            )

        # Read image and its information
        img = cv2.imread(img_path)
        height, width, _ = img.shape

        # Create dict
        coco_format = text_infos_to_coco_dict()
        # print(coco_format)

        # Read model output
        res_path = 'data/out_img_16.json'
        with open(res_path, 'r') as file:
            res_file = json.load(file)
        prediction = mmocr_result2coco(coco_format, res_file)
        # print(prediction)
        # print(mask.toBbox(prediction[0]['segmentation']))


        # EVALUATION WITH COCOAPI
        annType = 'segm'

        # Read anns
        annotation_file = 'temp_gt.json'
        json_content = json.dumps(coco_format)
        with open(annotation_file, 'w') as file:
            file.write(json_content)
        cocoGt=COCO(annotation_file)
        os.remove(annotation_file)
        # print(cocoGt.dataset)

        # read res
        cocoDt=cocoGt.loadRes(prediction)
        # print(cocoDt.dataset)

        # # running evaluation
        cocoEval = COCOeval(cocoGt,cocoDt)
        # # cocoEval.params.imgIds  = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

    # print(cocoEval.stats)

    # # Show gt
    # img = cocoGt.loadImgs(1)[0]
    # I = io.imread(img['file_name'])
    # plt.imshow(I); plt.axis('off')
    # anns = cocoGt.loadAnns(range(1, 8))
    # cocoGt.showAnns(anns)
    # plt.show()