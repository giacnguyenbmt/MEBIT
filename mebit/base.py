import os
import abc

import cv2
import numpy as np

from .metrics import rrc_evaluation_funcs_1_1 as rrc_evaluation_funcs
from .transforms import definded_transforms as trans

read_gt = rrc_evaluation_funcs.get_tl_line_values_from_file_contents

class BaseEvaluation(metaclass=abc.ABCMeta):
    keypoints = []
    masks = []
    bboxes = []
    # valid option list is detailed in subclass
    valid_option_list = []

    def __init__(self,
                 data,
                 gt,
                 image_color='rgb') -> None:
        self.data = data
        self.gt = gt
        self.image_color = image_color
        self.height, self.width, _ = self.data.shape
        self.report = {
            "blurring": {
                'message': 'blur_limit',
                'storage': self._init_store_option_data(0, 0),
                'note': 'higher is better',
                'generator': self.test_blurring()
            },
            "increasing_brightness": {
                'message': 'brightness_limit',
                'storage': self._init_store_option_data(0., 0),
                'note': 'higher is better',
                'generator': self.test_increasing_brightness()
            },
            "increasing_contrast": {
                'message': 'contrast_limit',
                'storage': self._init_store_option_data(0., 0),
                'note': 'higher is better',
                'generator': self.test_increasing_contrast()
            },
            "decreasing_brightness": {
                'message': 'brightness_limit',
                'storage': self._init_store_option_data(0., 0),
                'note': 'lower is better',
                'generator': self.test_decreasing_brightness()
            },
            "decreasing_contrast": {
                'message': 'contrast_limit',
                'storage': self._init_store_option_data(0., 0),
                'note': 'lower is better',
                'generator': self.test_decreasing_contrast()
            },
            "down_scale": {
                'message': 'max_ratio',
                'storage': self._init_store_option_data(1., 0),
                'note': 'lower is better',
                'generator': self.test_scale()
            },
            "crop": {
                'message': 'alpha',
                'storage': self._init_store_option_data(1., 0),
                'note': 'lower is better',
                'generator': self.test_crop()
            },
            "rotate90": {
                'message': 'num_image',
                'storage': self._init_store_option_data(0, 0),
                'note': 'higher is better',
                'generator': self.test_rotate90()
            },
        }

    def _init_store_option_data(self, init_value=0, init_score=0):
        _data = {
            'penultimate': {
                'data': None,
                'gt': None,
                'dt': None,
                'score': init_score,
                'value': init_value
            },
            'last': {
                'data': None,
                'gt': None,
                'dt': None,
                'score': init_score,
                'value': init_value
            },
        }
        return _data
    
    #======================================================
    #==============split transformation test===============
    def test_blurring(self):
        blur_limit = 3
        while True:
            transformed = trans.blur(
                blur_limit, image=self.data
            )
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
            transformed = trans.brightness(
                brightness_limit, image=self.data
            )
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
            transformed = trans.contrast(
                contrast_limit, image=self.data
            )
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
            transformed = trans.brightness(
                brightness_limit, image=self.data
            )
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
            transformed = trans.contrast(
                contrast_limit, image=self.data
            )
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
            transformed = trans.resize(
                ratio,
                image=self.image,
                masks=self.masks,
                keypoints=self.keypoints,
                bboxes=self.bboxes
            )
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
                transformed = trans.crop(
                    *coord,
                    image=self.image,
                    masks=self.masks,
                    keypoints=self.keypoints,
                    bboxes=self.bboxes
                )
                data.append(transformed['image'])
                del transformed['image']
                raw_gt.append(transformed)
            
            denominator += 1

            yield data, raw_gt

            if denominator >= 15:
                print("Reached the limit of the crop test!")
                self.stop_generator = True
    
    def test_rotate90(self):
        ...

    #======================================================
    #==================Log and report======================
    def backup_data(self, data, gt, dt):
        self.penultimate_data = data
        self.penultimate_gt = gt
        self.penultimate_dt = dt

    def save_image(self, name, image):
        if self.image_color == 'rgb':
            new_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            new_img = image
        cv2.imwrite(name, new_img)

    def save_images(self, data, type_data='deadpoint'):
        # store name of saved images
        img_names = []

        file_name = os.path.split(self.img_path)[-1]
        _name, _extension = os.path.splitext(file_name)

        if len(data) == 1:
            _new_name = _name \
                + "_{}_{}".format(self.option, type_data) \
                + _extension
            _path = os.path.join(
                self.result_image_path,
                'images',
                _new_name
            )
            self.save_image(_path, data[0])
            img_names.append(_new_name)
        else:
            for i, img in enumerate(data):
                _new_name = _name \
                    + "_{}_{}_{}".format(
                        self.option, 
                        i + 1,
                        type_data
                    ) \
                    + _extension
                _path = os.path.join(
                    self.result_image_path,
                    'images',
                    _new_name
                )
                self.save_image(_path, img)
                img_names.append(_new_name)
        return img_names

    @abc.abstractmethod
    def save_gt(self, gt, img_names):
        ...
    
    @abc.abstractmethod
    def save_dt(self, dt, img_names):
        ...
    
    def log(self, data, gt, dt, metric=None):
        if (self.result_image_path is None 
            or self.test_failed is False):
            return

        # last point
        img_names = self.save_images(
            self.penultimate_data, 
            type_data='lastpoint'
        )
        self.save_gt(self.penultimate_gt, img_names)
        self.save_dt(self.penultimate_dt, img_names)

        # dead point
        img_names = self.save_images(data, type_data='deadpoint')
        self.save_gt(gt, img_names)
        self.save_dt(dt, img_names)

    def update_report(self, option):
        self.report[option]['storage']['value'] = self.limit

    def make_report(self, option, verbose=True):
        message = "{}: \n{} = {} \n({})".format(option,
                                                self.report[option]['message'],
                                                self.report[option]['value'],
                                                self.report[option]['note'])
        if verbose is True:
            print(message)
        
        return None
        # return self.report[option]

    #======================================================
    #===============Metrics and condition==================
    @abc.abstractmethod
    def evaluate(self, gt, dt):
        metric = None
        return metric

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

    #======================================================
    #========================Process=======================
    def preprocess_input(self):
        # set param
        self.stop_generator = False

        # check option
        assert (self.option in self.valid_option_list), 'Invalid option'

        # Create a folder to store result
        if self.result_image_path is not None:
            if not os.path.exists(self.result_image_path):
                os.makedirs(os.path.join(self.result_image_path, 'images'))
                os.makedirs(os.path.join(self.result_image_path, 'gt'))
                os.makedirs(os.path.join(self.result_image_path, 'dt'))

        self.height, self.width, _ = self.data.shape

        # Read ground truth
        # self.read_groundtruth()
        
    # @abc.abstractmethod
    # def read_groundtruth(self):
    #     ...
    
    def get_generator(self, option):
        return self.report.get(option).get('generator')
    
    def create_original_input(self):
        data = [self.data]
        # format gt
        gt = self.format_original_gt()
        return data, gt

    @abc.abstractmethod
    def format_original_gt(self, *args, **kwargs):
        gt = None
        return gt

    def create_input(self, image_generator):
        """
        format data according to model_type and option
        """
        
        # Get data from generator
        data, raw_gt = next(image_generator)
        # Format data
        gt = self.format_transformed_gt(raw_gt, data=data)
        return data, gt

    @abc.abstractmethod
    def format_transformed_gt(self, *args, **kwargs):
        gt = None
        return gt

    def fit(self, inference_function, convert_output_function, data, gt):
        # get result from model
        results = []
        for _, img in enumerate(data):
            predicted_result = inference_function(img)
            converted_result = convert_output_function(predicted_result)
            results.append(converted_result)

        # format predicted result
        dt = self.format_dt(results=results)

        return dt

    @abc.abstractmethod
    def format_dt(self, *args, **kwargs):
        dt = None
        return dt

    def test_transformation(self,
                            inference_function,
                            convert_output_function,
                            option,
                            criterion,
                            threshold):
        # STEP 1: PREPROCESSING INPUT
        # Read orginal image and its groundtruth
        self.preprocess_input()

        # STEP 2: CHECK WHETHER MODEL FAIL WITH ORIGINAL IMAGE OR NOT
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
                    self.log(data, gt, dt)
                    break

    #======================================================
    #====================Main function=====================
    def stats(self,
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
        """

        self.option = option
        self.result_image_path = result_image_path

        self.test_transformation(
            inference_function,
            convert_output_function,
            option,
            criterion,
            threshold
        )
                
        return self.make_report(option, verbose)