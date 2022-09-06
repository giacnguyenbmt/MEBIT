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

from utils import rrc_evaluation_funcs_1_1 as rrc_evaluation_funcs
from utils import script

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
        self.stop_generator = False
        self.image_color = image_color
        
        self.report = {
            "blurring": {
                'message': 'blur_limit', 
                'value': 0, 
                'note': 'higher is better'},
            "increasing_brightness": {
                'message': 'brightness_limit', 
                'value': 0.0, 
                'note': 'higher is better'},
            "increasing_contrast": {
                'message': 'contrast_limit', 
                'value': 0.0, 
                'note': 'higher is better'},
            "decreasing_brightness": {
                'message': 'brightness_limit', 
                'value': 0.0, 
                'note': 'lower is better'},
            "decreasing_contrast": {
                'message': 'contrast_limit', 
                'value': 0.0, 
                'note': 'lower is better'},
            "down_scale": {
                'message': 'max_ratio', 
                'value': 1.0, 
                'note': 'lower is better'},
            "crop": {
                'message': 'alpha', 
                'value': 1.0, 
                'note': 'lower is better'},
        }

    def preprocess_input(self):
        # Create a folder to store result
        if (not os.path.exists(self.result_image_path) 
            and self.result_image_path is not None):
            os.makedirs(self.result_image_path)

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

        elif self.model_type == 'treg':
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
                self.transcriptions_list = gt_file

    # def save_images(self):
    #     for i, img in enumerate(self.transformed_image):
    #         new_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #         cv2.imwrite("./output/{}.jpg".format(i), new_img)

    def save_image(self, name, image):
        if self.image_color == 'rgb':
            new_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            new_img = image
        cv2.imwrite(name, new_img)

    # =====================================================
    # ==============define transformation==================
    def blur(self, blur_limit):
        transform = A.Compose([
            A.Blur(blur_limit=(blur_limit, blur_limit + 1), p=1.0),
        ])
        transformed_image = transform(image=self.img)["image"]
        return transformed_image

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
        transformed_image = transform(image=self.img)["image"]
        return transformed_image

    def contrast(self, contrast_limit):
        transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0, 
                contrast_limit=[contrast_limit, contrast_limit + 0.001], 
                brightness_by_max=True, 
                always_apply=False, 
                p=1.0),
        ])
        transformed_image = transform(image=self.img)["image"]
        return transformed_image

    def crop(self, h, w):
        # RandomCrop
        transform = A.Compose([
            A.RandomCrop(height=h, 
                width=w, 
                p=1),
        ], keypoint_params=A.KeypointParams(format='xy', 
                                            remove_invisible=False))
        transformed = transform(image=self.img, 
                                keypoints=self.keypoints)
        transformed_image = transformed['image']
        transformed_groundtruth = transformed['keypoints']

        return transformed_image, transformed_groundtruth

    def resize(self, ratio):
        h = int(self.height * ratio)
        w = int(self.width * ratio)

        transform = A.Compose([
            A.Resize(height=h, 
                width=w, 
                interpolation=1, 
                always_apply=False, 
                p=1),
        ], keypoint_params=A.KeypointParams(format='xy', 
                                            remove_invisible=False))
        transformed = transform(image=self.img, 
                                keypoints=self.keypoints)
        transformed_image = transformed['image']
        transformed_groundtruth = transformed['keypoints']

        return transformed_image, transformed_groundtruth

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

    def create_cocogt(self, img_path, gt, width, height):
        # Create dict
        coco_format = self.text_infos_to_coco_dict(img_path, gt, width, height)
        
        # convert to COCO class
        annotation_file = 'temp_gt.json'
        json_content = json.dumps(coco_format)
        with open(annotation_file, 'w') as file:
            file.write(json_content)
        cocoGt=COCO(annotation_file)
        os.remove(annotation_file)

        return cocoGt

    def tdet_result2coco(self, cocogt, res_file):
        # Create COCO-format result
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

        cocodt=cocogt.loadRes(results)
        return cocodt
    #======================================================
    #==============split transformation test===============       
    def test_original_image(self):
        data = [self.img]
        if self.model_type == 'tdet':
            tdet_gt = {
                'points_list': self.points_list,
                'transcriptions_list': self.transcriptions_list
            }
            gt = self.create_cocogt(self.img_path, 
                                    tdet_gt, 
                                    self.height, 
                                    self.width)

        return data, gt
        
    def test_blurring(self):
        blur_limit = 3
        while True:
            img = self.blur(blur_limit)

            # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # unique_value = np.unique(gray)

            self.current_img = img
            self.limit = blur_limit
            blur_limit += 2
            if self.model_type == 'tdet':
                self.current_points_list = self.points_list
                yield self.current_img, self.current_points_list
            elif self.model_type in ['treg', 'clsf']:
                yield self.current_img
            
            # if len(unique_value)  < 3:
            #     print("Reached the limit of blurring test!")
            #     self.stop_generator = True
            if blur_limit > 2 * max(self.height, self.width):
                print("Reached the limit of blurring test!")
                self.stop_generator = True

    def test_increasing_brightness(self):
        brightness_limit = 0
        while True:
            img = self.brightness(brightness_limit)
            self.current_img = img
            self.limit = brightness_limit
            brightness_limit += 0.1
            if self.model_type == 'tdet':
                self.current_points_list = self.points_list
                yield self.current_img, self.current_points_list
            elif self.model_type in ['treg', 'clsf']:
                yield self.current_img

            if brightness_limit > 1.0:
                print("Reached the limit of the brightness test!")
                self.stop_generator = True

    def test_increasing_contrast(self):
        contrast_limit = 0
        amout = 0.1
        while True:
            img = self.contrast(contrast_limit)

            self.current_img = img
            self.limit = contrast_limit
            amout += amout * 0.1
            contrast_limit += amout
            if self.model_type == 'tdet':
                self.current_points_list = self.points_list
                yield self.current_img, self.current_points_list
            elif self.model_type in ['treg', 'clsf']:
                yield self.current_img
            
            if contrast_limit > 255:
                print("Reached the limit of the contrast test!")
                self.stop_generator = True

    def test_decreasing_brightness(self):
        brightness_limit = 0
        while True:
            img = self.brightness(brightness_limit)
            self.current_img = img
            self.limit = brightness_limit
            brightness_limit -= 0.1
            if self.model_type == 'tdet':
                self.current_points_list = self.points_list
                yield self.current_img, self.current_points_list
            elif self.model_type in ['treg', 'clsf']:
                yield self.current_img

            if brightness_limit < -1.0:
                print("Reached the limit of the brightness test!")
                self.stop_generator = True

    def test_decreasing_contrast(self):
        contrast_limit = 0
        amout = 0.1
        while True:
            img = self.contrast(contrast_limit)

            self.current_img = img
            self.limit = contrast_limit
            amout += amout * 0.1
            contrast_limit -= amout
            if self.model_type == 'tdet':
                self.current_points_list = self.points_list
                yield self.current_img, self.current_points_list
            elif self.model_type in ['treg', 'clsf']:
                yield self.current_img
            
            if contrast_limit < -255:
                print("Reached the limit of the contrast test!")
                self.stop_generator = True

    def test_scale(self):
        ratio = 0.9
        while True:
            if ratio <= 0:
                break

            img, new_keypoints = self.resize(ratio)
            new_gt = [[new_keypoints[i][0], 
                       new_keypoints[i][1],
                       new_keypoints[i + 1][0],
                       new_keypoints[i + 1][1],
                       new_keypoints[i + 2][0],
                       new_keypoints[i + 2][1],
                       new_keypoints[i + 3][0],
                       new_keypoints[i + 3][1]] 
                       for i in range(0, len(new_keypoints), 4)]
            self.current_img = img
            self.current_points_list = new_gt
            self.limit = ratio

            ratio -= 0.1

            yield self.current_img, self.current_points_list

    def test_treg_scale(self):
        ratio = 0.9
        while True:

            img, _ = self.resize(ratio)
            self.current_img = img
            self.limit = ratio

            ratio -= 0.1

            yield self.current_img

            if ratio <= 0.11 or min(img.shape[:2]) < 3:
                self.stop_generator = True

    def test_tdet_crop(self):
        ...

    #======================================================
    def compute_accuracy(self, ground_truth, predictions, mode='per_char'):
        """
        Computes accuracy
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

    def evaluate_tdet(self, predicted_sample, option):
        if option == 'crop':
            predicted_sample.evaluate()
            predicted_sample.accumulate()
            predicted_sample.summarize()
            sample_metric = {
                'AP' : predicted_sample.stats[0],
                'AP.50' : predicted_sample.stats[1]
            }
        else:
            temp_func = script.evaluate_method_per_sample
            sample_metric = temp_func(self.gt_sample, 
                                    predicted_sample
                            )
            # message = 'Metrics:\nprecision = {}\nrecall = {}\nhmean = {}'
            # print(message.format(sample_metric['precision'], 
            #       sample_metric['recall'], 
            #       sample_metric['hmean'])
            # )
        return sample_metric

    def evaluate_treg(self, predicted_sample):
        acc = self.compute_accuracy(self.transcriptions_list, 
                                    [predicted_sample])
        levenshtein_distance = self.compute_levenshtein(self.transcriptions_list,
                                                        [predicted_sample])
        
        message = 'Metrics:\naccuracy = {}\nlevenshtein = {}'
        print(message.format(acc, levenshtein_distance))
        
        return {
            "accuracy": acc,
            "levenshtein": levenshtein_distance
        }
    
    def evaluate_clsf(self, predicted_sample):
        acc = 1 if self.transcriptions_list==predicted_sample else 0
        return {
            "accuracy": acc,
        }
        

    def create_original_input(self):
        if self.model_type == 'tdet':
            self.test_original_image()
            self.gt_sample = {
                'boxes': self.current_points_list,
                'texts': self.transcriptions_list
            }
        elif self.model_type in ['treg', 'clsf']:
            self.test_original_image()

    def create_input(self, image_generator):
        if self.model_type == 'tdet':
            next(image_generator)
            self.gt_sample = {
                'boxes': self.current_points_list,
                'texts': self.transcriptions_list
            }
        elif self.model_type in ['treg', 'clsf']:
            next(image_generator)

    def make_report(self, option, verbose=1):
        message = "{}: \n{} = {} \n({})".format(option,
                                                self.report[option]['message'],
                                                self.report[option]['value'],
                                                self.report[option]['note'])
        if verbose == 1:
            print(message)
        return self.report[option]

    def check(self, metrics, threshold, criterion="precision"):
        if self.stop_generator is True:
            return False
        elif criterion == 'levenshtein':
            if metrics[criterion] > threshold:
                return False
        elif metrics[criterion] < threshold:
            return False
        return True

    def update_report(self, option):
        self.report[option]['value'] = self.limit

    def fit(self, inference_function, convert_output_function, data):
        predicted_result = inference_function(data)
        converted_result = convert_output_function(predicted_result)

        if self.model_type == 'tdet':
            if self.option == 'crop':
                


    def tdet_stats(self,
                   inference_function,
                   convert_output_function,
                   option,
                   criterion="precision",
                   threshold=0.5,
                   result_image_path=None):
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
        self.result_image_path = result_image_path
        ## Step 1: check input image:
        # Read orginal image and groundtruth
        self.preprocess_input()

        # infer original image
        self.create_original_input()
        # them mask vao gt
        predicted_sample = inference_function(self.current_img)
        formated_sample = convert_output_function(predicted_sample)
        # them mask vao det
        if option == 'crop':
            gt = {
                'points_list' : self.points_list,
                'transcriptions_list': self.transcriptions_list
            }
            self.cocogt = self.create_cocogt(self.img_path, gt, self.height, self.width)
            self.cocodt = self.tdet_result2coco(self.cocogt, formated_sample)
            formated_sample = COCOeval(self.cocogt, self.cocodt)
            
        # get metric
        metrics = self.evaluate_tdet(formated_sample, option)

        # check condition
        if self.check(metrics, threshold, criterion) is True:

            ## Step 2: use corresponding generator
            option_list = {"blurring": self.test_blurring(), 
                           "increasing_brightness": self.test_increasing_brightness(), 
                           "increasing_contrast": self.test_increasing_contrast(), 
                           "decreasing_brightness": self.test_decreasing_brightness(), 
                           "decreasing_contrast": self.test_decreasing_contrast(), 
                           "down_scale": self.test_treg_scale(),
                           "crop": ...}
            
            # Get corresponding generator
            image_generator = option_list.get(option, None)
            if image_generator is None:
                print("Invalid option")
                return None

            while True:
                self.create_input(image_generator)
                # def pass through model
                predicted_sample = inference_function(self.current_img)
                formated_sample = convert_output_function(predicted_sample)
                
                # coco for tdet
                metrics = self.evaluate(formated_sample)

                # update check func
                if self.check(metrics, threshold, criterion) is False:
                    self.update_report(option)
                    if result_image_path is not None and self.stop_generator is False:
                        self.save_image(os.path.join(result_image_path,
                                                     option + os.path.split(self.img_path)[-1]), 
                                        self.current_img)
                    break

        ## Step 3
        return self.make_report(option, verbose)

    def treg_stats(self,
                   inference_function,
                   convert_output_function,
                   option,
                   criterion="accuracy",
                   threshold=0.5,
                   result_image_path=None):

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
                 "down_scale"]
        """
        
        self.model_type = 'treg'
        self.keypoints = []
        self.result_image_path = result_image_path
        ## Step 1: check input image:
        # Read orginal image and groundtruth
        self.preprocess_input()

        # infer original image
        self.create_original_input()
        predicted_sample = inference_function(self.current_img)
        formated_sample = convert_output_function(predicted_sample)
        # get metric
        metrics = self.evaluate_treg(formated_sample)

        # check condition
        if self.check(metrics, threshold, criterion) is True:

            ## Step 2: use corresponding generator
            option_list = {"blurring": self.test_blurring(), 
                           "increasing_brightness": self.test_increasing_brightness(), 
                           "increasing_contrast": self.test_increasing_contrast(), 
                           "decreasing_brightness": self.test_decreasing_brightness(), 
                           "decreasing_contrast": self.test_decreasing_contrast(), 
                           "down_scale": self.test_treg_scale()}
            
            # Get corresponding generator
            image_generator = option_list.get(option, None)
            if image_generator is None:
                print("Invalid option")
                return None

            while True:
                self.create_input(image_generator)

                predicted_sample = inference_function(self.current_img)
                formated_sample = convert_output_function(predicted_sample)
                
                metrics = self.evaluate_treg(formated_sample)

                if self.check(metrics, threshold, criterion) is False:
                    self.update_report(option)
                    if result_image_path is not None and self.stop_generator is False:
                        self.save_image(os.path.join(result_image_path,
                                                     option + os.path.split(self.img_path)[-1]), 
                                        self.current_img)
                    break

        ## Step 3
        return self.make_report(option, verbose)

    def clsf_stats(self,
                   inference_function,
                   convert_output_function,
                   option,
                   result_image_path=None,
                   verbose=0):
        
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
                 "down_scale"]
        """

        self.model_type = 'clsf'
        self.keypoints = []
        self.result_image_path = result_image_path
        ## Step 1: check input image:
        # Read orginal image and groundtruth
        self.preprocess_input()

        # infer original image
        self.create_original_input()
        predicted_sample = inference_function(self.current_img)
        formated_sample = convert_output_function(predicted_sample)
        # get metric
        metrics = self.evaluate_clsf(formated_sample)

        # check condition
        if self.check(metrics, 0.5, "accuracy") is True:

            ## Step 2: use corresponding generator
            option_list = {"blurring": self.test_blurring(), 
                           "increasing_brightness": self.test_increasing_brightness(), 
                           "increasing_contrast": self.test_increasing_contrast(), 
                           "decreasing_brightness": self.test_decreasing_brightness(), 
                           "decreasing_contrast": self.test_decreasing_contrast(), 
                           "down_scale": self.test_treg_scale()}
            
            # Get corresponding generator
            image_generator = option_list.get(option, None)
            if image_generator is None:
                print("Invalid option")
                return None

            while True:
                self.create_input(image_generator)
                
                predicted_sample = inference_function(self.current_img)
                formated_sample = convert_output_function(predicted_sample)
                
                metrics = self.evaluate_clsf(formated_sample)

                if self.check(metrics, 0.5, "accuracy") is False:
                    self.update_report(option)
                    if result_image_path is not None and self.stop_generator is False:
                        self.save_image(os.path.join(result_image_path,
                                                     option + os.path.split(self.img_path)[-1]), 
                                        self.current_img)
                    break

        ## Step 3
        return self.make_report(option, verbose)