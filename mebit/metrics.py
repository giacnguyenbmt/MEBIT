import os
import re
import sys
import json

import cv2
import Levenshtein
import numpy as np
import albumentations as A
from pycocotools import mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score

from .utils import rrc_evaluation_funcs_1_1 as rrc_evaluation_funcs
from .utils import script

read_gt = rrc_evaluation_funcs.get_tl_line_values_from_file_contents

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class Evaluation:
    def __init__(self,
                 img_path,
                 gt_path,
                 image_color='rgb') -> None:
        self.img_path = img_path
        self.gt_path = gt_path
        self.image_color = image_color

        self.keypoints = []
        self.masks = []
        self.bboxes = []
        
        self.report = {
            "blurring": {
                'message': 'blur_limit', 
                'value': 0, 
                'note': 'higher is better',
                'generator': self.test_blurring()},
            "increasing_brightness": {
                'message': 'brightness_limit', 
                'value': 0.0, 
                'note': 'higher is better',
                'generator': self.test_increasing_brightness()},
            "increasing_contrast": {
                'message': 'contrast_limit', 
                'value': 0.0, 
                'note': 'higher is better',
                'generator': self.test_increasing_contrast()},
            "decreasing_brightness": {
                'message': 'brightness_limit', 
                'value': 0.0, 
                'note': 'lower is better',
                'generator': self.test_decreasing_brightness()},
            "decreasing_contrast": {
                'message': 'contrast_limit', 
                'value': 0.0, 
                'note': 'lower is better',
                'generator': self.test_decreasing_contrast()},
            "down_scale": {
                'message': 'max_ratio', 
                'value': 1.0, 
                'note': 'lower is better',
                'generator': self.test_scale()},
            "crop": {
                'message': 'alpha', 
                'value': 1.0, 
                'note': 'lower is better',
                'generator': self.test_crop()},
        }

    def preprocess_input(self):
        # set param
        self.stop_generator = False

        # check option
        assert (self.option in self.report.keys()), 'Invalid option'

        # Create a folder to store result
        if self.result_image_path is not None:
            if not os.path.exists(self.result_image_path):
                os.makedirs(os.path.join(self.result_image_path, 'images'))
                os.makedirs(os.path.join(self.result_image_path, 'gt'))
                os.makedirs(os.path.join(self.result_image_path, 'dt'))

        # Read image
        BGR_img = cv2.imread(self.img_path)
        if self.image_color == 'rgb':
            self.img = cv2.cvtColor(BGR_img, cv2.COLOR_BGR2RGB)
        elif self.image_color == 'bgr':
            self.img = BGR_img

        self.height, self.width, _ = self.img.shape

        # Read ground truth
        if self.model_type == 'tdet':
            with open(self.gt_path, 'r') as file:
                gt_file = file.read()
                self.points_list, _, self.transcriptions_list = read_gt(
                    gt_file,
                    CRLF=False,
                    LTRB=False,
                    withTranscription=True,
                    withConfidence=False,
                )
            self.keypoints = [(self.points_list[j][i], 
                                self.points_list[j][i + 1])
                                for j in range(len(self.points_list)) 
                                for i in range(0, 8, 2)]

        elif self.model_type == 'trecog':
            self.transcriptions_list = []
            evaluationParams = {
                'SAMPLE_NAME_2_ID':'(?:word_)?([0-9]+).png',
                'CRLF':False,
                'DOUBLE_QUOTES':True
                }

            with open(self.gt_path, 'r') as file:
                gt_file = file.read()

            gtLines = gt_file.split("\r\n" if evaluationParams['CRLF'] else "\n")
            for line in gtLines:
                line = line.replace("\r","").replace("\n","")
                if(line != ""):
                    if (evaluationParams['DOUBLE_QUOTES']):
                        m = re.search(r'"(.+)"',line)
                    else:
                        m = re.search(r"'(.+)'",line)
                    self.transcriptions_list.append(m.group()[1:-1])

        elif self.model_type == 'clsf':
            with open(self.gt_path, 'r') as file:
                gt_file = file.read().replace('\n', '')
                self.transcriptions_list = [gt_file]
    
    def get_generator(self, option):
        return self.report.get(option).get('generator')

    def save_image(self, name, image):
        if self.image_color == 'rgb':
            new_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            new_img = image
        cv2.imwrite(name, new_img)

    def save_images(self, data, type_data='deadpoint'):
        if self.result_image_path is not None:
            if self.test_failed:
                file_name = os.path.split(self.img_path)[-1]
                _name, _extension = os.path.splitext(file_name)

                if len(data) == 1:
                    _path = os.path.join(
                        self.result_image_path,
                        'images',
                        _name \
                        + "_{}_{}".format(self.option, type_data) \
                        + _extension
                    )
                    self.save_image(_path, data[0])
                else:
                    for i, img in enumerate(data):
                        _path = os.path.join(
                            self.result_image_path,
                            'images',
                            _name
                            + "_{}_{}_{}".format(
                                self.option, 
                                i + 1,
                                type_data
                            ) \
                            + _extension
                        )
                        self.save_image(_path, img)

    # =====================================================
    # ==============define transformation==================
    def blur(self, blur_limit):
        transform = A.Compose([
            A.Blur(blur_limit=(blur_limit, blur_limit + 1), p=1.0),
        ])
        transformed = transform(image=self.img)
        return transformed

    def brightness(self, brightness_limit):
        transform = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=[brightness_limit, 
                                  brightness_limit+0.001],
                contrast_limit=0, 
                brightness_by_max=True, 
                always_apply=False, 
                p=1.0),
        ])
        transformed = transform(image=self.img)
        return transformed

    def contrast(self, contrast_limit):
        transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0, 
                contrast_limit=[contrast_limit, contrast_limit + 0.001], 
                brightness_by_max=True, 
                always_apply=False, 
                p=1.0),
        ])
        transformed = transform(image=self.img)
        return transformed

    def crop(self, x_min, y_min, x_max, y_max):
        transform = A.Compose(
            [A.Crop(x_min, y_min, x_max, y_max)], 
            keypoint_params=A.KeypointParams(format='xy', 
                                            remove_invisible=False),
            bbox_params=A.BboxParams(format='coco')
        )
        transformed = transform(image=self.img, 
                                masks=self.masks,
                                keypoints=self.keypoints,
                                bboxes=self.bboxes)
        return transformed

    def resize(self, ratio):
        h = int(self.height * ratio)
        w = int(self.width * ratio)

        transform = A.Compose(
            [A.Resize(
                height=h, 
                width=w, 
                interpolation=1, 
                always_apply=False, 
                p=1)], 
            keypoint_params=A.KeypointParams(format='xy', 
                                             remove_invisible=False),
            bbox_params=A.BboxParams(format='coco')
        )
        transformed = transform(image=self.img, 
                                masks=self.masks,
                                keypoints=self.keypoints,
                                bboxes=self.bboxes)
        return transformed

    #======================================================
    #==============split transformation test===============
    def test_blurring(self):
        blur_limit = 3
        while True:
            transformed = self.blur(blur_limit)
            data = [transformed['image']]
            del transformed['image']
            raw_gt = [transformed]

            self.limit = blur_limit
            blur_limit += 2

            yield data, raw_gt
            
            if blur_limit > 2 * max(self.height, self.width):
                print("Reached the limit of the blurring test!")
                self.stop_generator = True

    def test_increasing_brightness(self):
        brightness_limit = 0
        while True:
            transformed = self.brightness(brightness_limit)
            data = [transformed['image']]
            del transformed['image']
            raw_gt = [transformed]

            self.limit = brightness_limit
            brightness_limit = round(brightness_limit + 0.1, 1)

            yield data, raw_gt

            if brightness_limit >= 1.0:
                print("Reached the limit of the brightness test!")
                self.stop_generator = True

    def test_increasing_contrast(self):
        contrast_limit = 0
        amout = 0.1
        while True:
            transformed = self.contrast(contrast_limit)
            data = [transformed['image']]
            del transformed['image']
            raw_gt = [transformed]

            self.limit = contrast_limit
            amout += amout * 0.1
            contrast_limit += amout

            yield data, raw_gt
            
            if contrast_limit > 255:
                contrast_limit = 255.0
                print("Reached the limit of the contrast test!")
                self.stop_generator = True

    def test_decreasing_brightness(self):
        brightness_limit = 0
        while True:
            transformed = self.brightness(brightness_limit)
            data = [transformed['image']]
            del transformed['image']
            raw_gt = [transformed]

            self.limit = brightness_limit
            brightness_limit  = round(brightness_limit - 0.1, 1)

            yield data, raw_gt

            if brightness_limit <= -1.0:
                print("Reached the limit of the brightness test!")
                self.stop_generator = True

    def test_decreasing_contrast(self):
        contrast_limit = 0
        amout = 0.1
        while True:
            transformed = self.contrast(contrast_limit)
            data = [transformed['image']]
            del transformed['image']
            raw_gt = [transformed]

            self.limit = contrast_limit
            amout += amout * 0.1
            contrast_limit -= amout

            yield data, raw_gt
            
            if contrast_limit < -255:
                contrast_limit = -255.0
                print("Reached the limit of the contrast test!")
                self.stop_generator = True

    def test_scale(self):
        ratio = 0.9
        while True:
            transformed = self.resize(ratio)
            data = [transformed['image']]
            del transformed['image']
            raw_gt = [transformed]

            self.limit = ratio
            ratio = round(ratio - 0.1, 1)

            yield data, raw_gt

            if ratio <= 0.1 or min(data[0].shape[:2]) < 3:
                print("Reached the limit of the down-scale test!")
                self.stop_generator = True

    def test_crop(self):
        # crop 9 parts of image according alpha
        numerator = 5
        denominator = 6

        while True:
            # 1/3 <= alpha < 1.0
            alpha = numerator / denominator

            # Create a matrix which represents 9 parts of image
            # [[top-left]     [top-center]      [top-right]
            #  [midle-left]     [center]      [midle-right]
            #  [bottom-left] [bottom-center] [bottom-right]]
            alpha_matrix = np.array([
                [            0,             0,                 alpha,                 alpha],
                [(1 - alpha)/2,             0, (1 - alpha)/2 + alpha,                 alpha],
                [    1 - alpha,             0,                     1,                 alpha],
                [            0, (1 - alpha)/2,                 alpha, (1 - alpha)/2 + alpha],
                [(1 - alpha)/2, (1 - alpha)/2, (1 - alpha)/2 + alpha, (1 - alpha)/2 + alpha],
                [    1 - alpha, (1 - alpha)/2,                     1, (1 - alpha)/2 + alpha],
                [            0,     1 - alpha,                 alpha,                     1],
                [(1 - alpha)/2,     1 - alpha, (1 - alpha)/2 + alpha,                     1],
                [    1 - alpha,     1 - alpha,                     1,                     1],
            ])
            self.limit = alpha

            data = []
            raw_gt = []
            new_coords = alpha_matrix * np.array([self.width, self.height, self.width, self.height])
            new_coords = new_coords.astype(int)    

            for _, coord in enumerate(new_coords):
                transformed = self.crop(*coord)
                data.append(transformed['image'])
                del transformed['image']
                raw_gt.append(transformed)
            
            denominator += 1

            yield data, raw_gt

            if denominator >= 15:
                print("Reached the limit of the crop test!")
                self.stop_generator = True

    #======================================================
    #==================COCO DATASET TOOL=================== 
    def find_bbox(self, polygon):
        x = min([polygon[i] for i in range(0, len(polygon), 2)])
        y = min([polygon[i] for i in range(1, len(polygon), 2)])
        width = max([polygon[i] for i in range(0, len(polygon), 2)]) - x
        height = max([polygon[i] for i in range(1, len(polygon), 2)]) - y
        return [x, y, width, height]

    def find_area(self, segmentation, height, width):
        objs = mask.frPyObjects(segmentation, height, width)
        area = mask.area(objs)
        return float(area[0])

    def text_infos_to_coco_dict(self, img_path, gt, width, height):
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
                    "area": self.find_area([polygon], height, width),
                    "bbox": self.find_bbox(polygon), 
                    "iscrowd": 0,
                }
                id += 1
                anns.append(ann)
        image_infos['annotations'] = anns

        image_infos['categories'] = [
            {'id': 1, 'name': 'text'}
        ]
        return image_infos

    def albu_to_coco_dict(self, data, raw_gt):
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

    def create_cocogt(self, coco_format):
        # convert to COCO class
        annotation_file = 'temp_gt.json'
        json_content = json.dumps(coco_format)
        with open(annotation_file, 'w') as file:
            file.write(json_content)
        with HiddenPrints():
            cocoGt=COCO(annotation_file)
        os.remove(annotation_file)
        return cocoGt

    def tdet_result2coco(self, cocogt, res_file):
        img_id = list(cocogt.imgs.keys())[0]
        h, w = cocogt.imgs[img_id]['height'], cocogt.imgs[img_id]['width']
        boundary_result = res_file['boxes']
        score = res_file['confidences']
        results = []
        for i, poly in enumerate(boundary_result):
            predict_seg = {
                "image_id": img_id, 
                "category_id": 1, 
                "segmentation": mask.frPyObjects(
                    [poly], 
                    h, w
                )[0], 
                "score": score[i],
            }
            results.append(predict_seg)

        with HiddenPrints():
            cocodt=cocogt.loadRes(results)
        return cocodt

    #======================================================
    def compute_accuracy(self, ground_truth, predictions, mode='per_char'):
        """
        Computes accuracy for text recognition
        :param ground_truth:
        :param predictions:
        :param display: Whether to print values to stdout
        :param mode: if 'per_char' is selected then
                    single_label_accuracy = correct_predicted_char_nums_of_single_sample / single_label_char_nums
                    avg_label_accuracy = sum(single_label_accuracy) / label_nums
                    if 'full_sequence' is selected then
                    single_label_accuracy = 1 if the prediction result is exactly the same as label else 0
                    avg_label_accuracy = sum(single_label_accuracy) / label_nums
        :return: avg_label_accuracy
        """
        if mode == 'per_char':

            accuracy = []

            for index, label in enumerate(ground_truth):
                prediction = predictions[index]
                total_count = len(label)
                correct_count = 0
                try:
                    for i, tmp in enumerate(label):
                        if tmp == prediction[i]:
                            correct_count += 1
                except IndexError:
                    continue
                finally:
                    try:
                        accuracy.append(correct_count / total_count)
                    except ZeroDivisionError:
                        if len(prediction) == 0:
                            accuracy.append(1)
                        else:
                            accuracy.append(0)
            avg_accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
        elif mode == 'full_sequence':
            try:
                correct_count = 0
                for index, label in enumerate(ground_truth):
                    prediction = predictions[index]
                    if prediction == label:
                        correct_count += 1
                avg_accuracy = correct_count / len(ground_truth)
            except ZeroDivisionError:
                if not predictions:
                    avg_accuracy = 1
                else:
                    avg_accuracy = 0
        else:
            raise NotImplementedError('Other accuracy compute mode has not been implemented')

        return avg_accuracy

    def compute_levenshtein(self, ground_truth, predictions):
        accuracy = []
        for i, gt in enumerate(ground_truth):
            accuracy.append(Levenshtein.distance(gt, predictions[i]))
        return np.mean(np.array(accuracy).astype(np.float32), axis=0)

    def evaluate(self, gt, dt):
        if self.model_type == 'tdet':
            if self.option == 'crop':
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
                
        elif self.model_type == 'trecog':
            acc = self.compute_accuracy(gt, dt)
            levenshtein_distance = self.compute_levenshtein(gt, dt)
            metric = {
                "accuracy": acc,
                "levenshtein": levenshtein_distance
            }
        elif self.model_type == 'clsf':
            metric = {
                "accuracy": accuracy_score(gt, dt),
                "precision": precision_score(gt, dt, average='micro'),
                "recall": recall_score(gt, dt, average='micro'),
                "f1": f1_score(gt, dt, average='micro')
            }

        return metric
    
    def evaluate_clsf(self, predicted_sample):
        acc = 1 if self.transcriptions_list==predicted_sample else 0
        return {
            "accuracy": acc,
        }
        
    def create_original_input(self):
        data = [self.img]
        if self.model_type == 'tdet':
            if self.option == 'crop':
                tdet_gt = {
                    'points_list': self.points_list,
                    'transcriptions_list': self.transcriptions_list
                }
                coco_format = self.text_infos_to_coco_dict(
                    self.img_path, 
                    tdet_gt, 
                    self.width,
                    self.height
                )
                gt = self.create_cocogt(coco_format)
                # create masks from corresponding polygons
                for id in gt.getAnnIds(imgIds=1):
                    self.masks.append(gt.annToMask(gt.loadAnns(id)[0]))
            else:
                gt = {
                    'boxes': self.points_list,
                    'texts': self.transcriptions_list
                }

        elif self.model_type == 'trecog':
            if self.option == 'crop':
                gt = self.transcriptions_list
            else:
                gt = self.transcriptions_list
        
        elif self.model_type == 'clsf':
            gt = self.transcriptions_list

        return data, gt

    def create_input(self, image_generator):
        """
        format data according to model_type and option
        """
        
        # Get data from generator
        data, raw_gt = next(image_generator)

        # Format data
        if self.model_type == 'tdet':
            if self.option == 'crop':
                coco_format = self.albu_to_coco_dict(data, raw_gt)
                gt = self.create_cocogt(coco_format)
            elif self.option == "down_scale":
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
                    'texts': self.transcriptions_list
                }
            else:
                gt = {
                    'boxes': self.points_list,
                    'texts': self.transcriptions_list
                }
        elif self.model_type  == 'trecog':
            if self.option == 'crop':
                gt = self.transcriptions_list * 9
            else:
                gt = self.transcriptions_list
        elif self.model_type == 'clsf':
            if self.option == 'crop':
                gt = self.transcriptions_list * 9
            else:
                gt = self.transcriptions_list

        return data, gt

    def make_report(self, option, verbose=True):
        message = "{}: \n{} = {} \n({})".format(option,
                                                self.report[option]['message'],
                                                self.report[option]['value'],
                                                self.report[option]['note'])
        if verbose is True:
            print(message)
        return self.report[option]

    def check(self, metrics, threshold, criterion="precision"):
        self.test_failed = True
        if criterion == 'levenshtein':
            if metrics[criterion] > threshold:
                return False
        elif metrics[criterion] < threshold:
            return False
        self.test_failed = False

        if self.stop_generator is True:
            return False

        return True

    def update_report(self, option):
        self.report[option]['value'] = self.limit

    def fit(self, inference_function, convert_output_function, data, gt):
        # get result from model
        results = []
        for _, img in enumerate(data):
            predicted_result = inference_function(img)
            converted_result = convert_output_function(predicted_result)
            results.append(converted_result)

        # format result
        if self.model_type == 'tdet':
            if self.option == 'crop':
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
            else:
                dt = results[0]
        elif self.model_type == 'trecog':
            if self.option == 'crop':
                dt = results
            else:
                dt = results
        elif self.model_type == 'clsf':
            dt = results
        return dt

    def backup_data(self, data, gt, dt):
        self.penultimate_data = data
        self.penultimate_gt = gt
        self.penultimate_dt = dt

    def transformation_test(self,
                            inference_function,
                            convert_output_function,
                            option,
                            criterion,
                            threshold):
        # STEP 1: PREPROCESSING INPUT
        # Read orginal image and its groundtruth
        self.preprocess_input()

        ##############################################################
        # STEP 2: CHECK WHETHER MODEL FAIL WITH ORIGINAL IMAGE OR NOT
        ##############################################################

        # Create data which has format corresponding option
        data, gt  = self.create_original_input()

        # Conduct inference and format model result
        dt = self.fit(inference_function, convert_output_function, data, gt)

        metric = self.evaluate(gt, dt)

        # STEP 3: RUN TEST WITH CORRESPONDING OPTION
        if self.check(metric, threshold, criterion):
            # Get corresponding generator
            image_generator = self.get_generator(option)

            while True:
                self.backup_data(data, gt, dt)

                # Create data which has format corresponding option
                data, gt = self.create_input(image_generator)
                
                # Conduct inference and format model result
                dt = self.fit(inference_function, convert_output_function, data, gt)

                metric = self.evaluate(gt, dt)

                # Check end condition
                if self.check(metric, threshold, criterion) is False:
                    self.update_report(option)
                    # save log point at which model dectected incorrect
                    self.save_images(data, type_data='deadpoint')
                    self.save_images(self.penultimate_data, type_data='lastpoint')
                    break

    def tdet_stats(self,
                   inference_function,
                   convert_output_function,
                   option,
                   criterion="precision",
                   threshold=0.5,
                   result_image_path=None,
                   verbose=False):
        """
        Parameter:
        inference_function: a function receive our test input and give
        coresponding predicted sample.
        convert_output_function: this function converts your model 
        output according our format.
        option: ["blurring", 
                 "increasing_brightness", 
                 "increasing_contrast", 
                 "decreasing_brightness", 
                 "decreasing_contrast", 
                 "down_scale", 
                 "crop"]
        """

        self.model_type = 'tdet'
        self.option = option
        self.result_image_path = result_image_path

        self.transformation_test(
            inference_function,
            convert_output_function,
            option,
            criterion,
            threshold
        )
                
        return self.make_report(option, verbose)

    def trecog_stats(self,
                     inference_function,
                     convert_output_function,
                     option,
                     criterion="precision",
                     threshold=0.5,
                     result_image_path=None,
                     verbose=False):
        """
        Parameter:
        inference_function: a function receive our test input and give
        coresponding predicted sample.
        convert_output_function: this function converts your model 
        output according our format.
        option: ["blurring", 
                 "increasing_brightness", 
                 "increasing_contrast", 
                 "decreasing_brightness", 
                 "decreasing_contrast", 
                 "down_scale", 
                 "crop"]
        """

        self.model_type = 'trecog'
        self.option = option
        self.result_image_path = result_image_path

        self.transformation_test(
            inference_function,
            convert_output_function,
            option,
            criterion,
            threshold
        )
                
        return self.make_report(option, verbose)

    def clsf_stats(self,
                   inference_function,
                   convert_output_function,
                   option,
                   criterion="precision",
                   threshold=0.5,
                   result_image_path=None,
                   verbose=False):
        """
        Parameter:
        inference_function: a function receive our test input and give
        coresponding predicted sample.
        convert_output_function: this function converts your model 
        output according our format.
        option: ["blurring", 
                 "increasing_brightness", 
                 "increasing_contrast", 
                 "decreasing_brightness", 
                 "decreasing_contrast", 
                 "down_scale", 
                 "crop"]
        """

        self.model_type = 'clsf'
        self.option = option
        self.result_image_path = result_image_path

        self.transformation_test(
            inference_function,
            convert_output_function,
            option,
            criterion,
            threshold
        )
                
        return self.make_report(option, verbose)