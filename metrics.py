from operator import gt
import re
import os

import albumentations as A
import Levenshtein
import cv2
import numpy as np

from utils import rrc_evaluation_funcs_1_1 as rrc_evaluation_funcs
from utils import script

read_gt = rrc_evaluation_funcs.get_tl_line_values_from_file_contents

class Evaluation:
    def __init__(self,
                 img_path,
                 gt_path,
                 result_path='./result') -> None:
        self.img_path = img_path
        self.gt_path = gt_path
        self.result_path = result_path
        self.stop_generator = False
        
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
        }

    def preprocess_input(self):
        # Create a folder to store result
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        # Read image
        BGR_img = cv2.imread(self.img_path)
        self.img = cv2.cvtColor(BGR_img, cv2.COLOR_BGR2RGB)
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

    def save_images(self):
        for i, img in enumerate(self.transformed_image):
            new_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite("./output/{}.jpg".format(i), new_img)

    def save_image(self, name, image):
        new_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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

    def random_crop(self):
        ...
        pass

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
    #==============split transformation test===============       
    def test_original_image(self):
        self.current_img = self.img
        if self.model_type == 'tdet':
            self.current_points_list = self.points_list
        
    def test_blurring(self):
        blur_limit = 3
        while True:
            img = self.blur(blur_limit)
            self.current_img = img
            self.limit = blur_limit
            blur_limit += 2
            if self.model_type == 'tdet':
                self.current_points_list = self.points_list
                yield self.current_img, self.current_points_list
            elif self.model_type == 'treg':
                yield self.current_img

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
            elif self.model_type == 'treg':
                yield self.current_img

            if brightness_limit > 1.0:
                print("Reached the limit of brightness test!")
                self.stop_generator = True

    def test_increasing_contrast(self):
        contrast_limit = 0
        amout = 0.1
        while True:
            img = self.contrast(contrast_limit)

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            unique_value = np.unique(gray)

            self.current_img = img
            self.limit = contrast_limit
            amout += amout * 0.1
            contrast_limit += amout
            if self.model_type == 'tdet':
                self.current_points_list = self.points_list
                yield self.current_img, self.current_points_list
            elif self.model_type == 'treg':
                yield self.current_img
            
            if len(unique_value)  < 3:
                print("Reached the limit of contrast test!")
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
            elif self.model_type == 'treg':
                yield self.current_img

            if brightness_limit > 1.0:
                print("Reached the limit of brightness test!")
                self.stop_generator = True

    def test_decreasing_contrast(self):
        contrast_limit = 0
        amout = 0.1
        while True:
            img = self.contrast(contrast_limit)

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            unique_value = np.unique(gray)

            self.current_img = img
            self.limit = contrast_limit
            amout += amout * 0.1
            contrast_limit -= amout
            if self.model_type == 'tdet':
                self.current_points_list = self.points_list
                yield self.current_img, self.current_points_list
            elif self.model_type == 'treg':
                yield self.current_img
            
            if len(unique_value)  < 3:
                print("Reached the limit of contrast test!")
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
            if ratio <= 0:
                break

            img, _ = self.resize(ratio)
            self.current_img = img

            self.limit = ratio
            ratio -= 0.1

            yield self.current_img

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

    def evaluate(self, predicted_sample):
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
        return message

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

    def tdet_stats(self,
                   inference_function,
                   convert_output_function,
                   option,
                   criterion="precision",
                   threshold=0.5,
                   save_image=False):
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

        self.model_type = 'tdet'
        ## Step 1: check input image:
        # Read orginal image and groundtruth
        self.preprocess_input()

        # infer original image
        self.create_original_input()
        predicted_sample = inference_function(self.current_img)
        formated_sample = convert_output_function(predicted_sample)
        # get metric
        metrics = self.evaluate(formated_sample)

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
                
                metrics = self.evaluate(formated_sample)

                if self.check(metrics, threshold, criterion) is False:
                    self.update_report(option)
                    if save_image == True:
                        self.save_image('./result/' + option + '.jpg', self.current_img)
                    break

        ## Step 3
        self.make_report(option)

    def treg_stats(self,
                   inference_function,
                   convert_output_function,
                   option,
                   criterion="accuracy",
                   threshold=0.5,
                   save_image=False):

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
                    if save_image == True:
                        self.save_image('./result/' + option + '.jpg', self.current_img)
                    break

        ## Step 3
        self.make_report(option)

    def clsf_stats(self,
                   inference_function,
                   convert_output_function,
                   option,
                   save_image=False,
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
                    if save_image == True:
                        self.save_image('./result/' + option + '.jpg', self.current_img)
                    break

        ## Step 3
        return self.make_report(option, verbose)