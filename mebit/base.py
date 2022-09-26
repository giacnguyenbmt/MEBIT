from genericpath import isdir
import os
import abc
import random
import string

import cv2
import numpy as np

from .metrics import rrc_evaluation_funcs_1_1 as rrc_evaluation_funcs
from .transforms import definded_transforms as trans

read_gt = rrc_evaluation_funcs.get_tl_line_values_from_file_contents

class BaseEvaluation(metaclass=abc.ABCMeta):
    # count number of images which model fits
    counter = 0
    
    length_of_random_name = 5
    img_path = None
    gt_path = None
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
                'generator': self.test_blurring(),
                'type': 1
            },
            "increasing_brightness": {
                'message': 'brightness_limit',
                'storage': self._init_store_option_data(0., 0),
                'note': 'higher is better',
                'generator': self.test_increasing_brightness(),
                'type': 1
            },
            "increasing_contrast": {
                'message': 'contrast_limit',
                'storage': self._init_store_option_data(0., 0),
                'note': 'higher is better',
                'generator': self.test_increasing_contrast(),
                'type': 1
            },
            "decreasing_brightness": {
                'message': 'brightness_limit',
                'storage': self._init_store_option_data(0., 0),
                'note': 'lower is better',
                'generator': self.test_decreasing_brightness(),
                'type': 1
            },
            "decreasing_contrast": {
                'message': 'contrast_limit',
                'storage': self._init_store_option_data(0., 0),
                'note': 'lower is better',
                'generator': self.test_decreasing_contrast(),
                'type': 1
            },
            "down_scale": {
                'message': 'max_ratio',
                'storage': self._init_store_option_data(1., 0),
                'note': 'lower is better',
                'generator': self.test_scale(),
                'type': 1
            },
            "crop": {
                'message': 'alpha',
                'storage': self._init_store_option_data(1., 0),
                'note': 'lower is better',
                'generator': self.test_crop(),
                'type': 2
            },
            "rotate90": {
                'message': 'num_image',
                'storage': self._init_store_option_data(0, 0),
                'note': 'higher is better',
                'generator': self.test_rotate_90(),
                'type': 4
            },
            "left_rotation": {
                'message': 'rotation_limit',
                'storage': self._init_store_option_data(0, 0),
                'note': 'higher is better',
                'generator': self.test_left_rotation(),
                'type': 3
            },
            "right_rotation": {
                'message': 'rotation_limit',
                'storage': self._init_store_option_data(0, 0),
                'note': 'lower is better',
                'generator': self.test_right_rotation(),
                'type': 3
            },
            "compactness": {
                'message': 'compacness_limit',
                'storage': self._init_store_option_data(0, 0),
                'note': 'lower is better',
                'generator': self.test_compactness(),
                'type': 3
            }
        }

    @classmethod
    def get_available_option(cls):
        return cls.valid_option_list

    def _init_store_option_data(self, init_value=0, init_score=0):
        option_data = {
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
        return option_data
    
    #======================================================
    #==============split transformation test===============
    def test_blurring(self, *args, **kwargs):
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

    def test_increasing_brightness(self, *args, **kwargs):
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

    def test_increasing_contrast(self, *args, **kwargs):
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

    def test_decreasing_brightness(self, *args, **kwargs):
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

    def test_decreasing_contrast(self, *args, **kwargs):
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

    def test_scale(self, *args, **kwargs):
        ratio = 0.9
        while True:
            transformed = trans.resize(
                ratio,
                image=self.data,
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

    def test_crop(self, *args, **kwargs):
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
                    image=self.data,
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
    
    def test_rotate_90(self, flip=False, *args, **kwargs):
        while True:
            data = []
            raw_gt = []
            
            if flip is True:
                for k in range(0, 4):
                    transformed = trans.flip_rorate90(
                        k=k, 
                        flip=True, 
                        image=self.data,
                        masks=self.masks,
                        keypoints=self.keypoints,
                        bboxes=self.bboxes
                    )
                    data.append(transformed['image'])
                    del transformed['image']
                    raw_gt.append(transformed)

                    yield data, raw_gt

            for k in range(1, 4):
                transformed = trans.flip_rorate90(
                    k=k, 
                    flip=False, 
                    image=self.data,
                    masks=self.masks,
                    keypoints=self.keypoints,
                    bboxes=self.bboxes
                )
                data.append(transformed['image'])
                del transformed['image']
                raw_gt.append(transformed)
                
                # stop condition
                self.stop_type_4 = (k == 3)

                yield data, raw_gt

    def test_left_rotation(self, color=None, *args, **kwargs):
        rotation_limit = 0
        while True:
            transformed = trans.rotate(
                limit=rotation_limit,
                image=self.data,
                masks=self.masks,
                keypoints=self.keypoints,
                bboxes=self.bboxes
            )
            
            raw_data = transformed['image']
            x, y, w, h, _ = transformed['bboxes'][0]
            x, y, w, h = np.array([x, y, w, h]).round().astype(int)
            data = [raw_data[y:y + h, x:x + w]]

            del transformed['image']
            raw_gt = [transformed]

            self.limit = rotation_limit
            rotation_limit += 1

            yield data, raw_gt

            if rotation_limit >= 45:
                print("Reached the limit of the left rotation test!")
                self.stop_generator = True
    
    def test_right_rotation(self, *args, **kwargs):
        rotation_limit = 0
        while True:
            transformed = trans.rotate(
                limit=rotation_limit,
                image=self.data,
                masks=self.masks,
                keypoints=self.keypoints,
                bboxes=self.bboxes
            )
            
            raw_data = transformed['image']
            x, y, w, h, _ = transformed['bboxes'][0]
            x, y, w, h = np.array([x, y, w, h]).round().astype(int)
            data = [raw_data[y:y + h, x:x + w]]

            del transformed['image']
            raw_gt = [transformed]

            self.limit = rotation_limit
            rotation_limit -= 1

            yield data, raw_gt

            if rotation_limit <= -45:
                print("Reached the limit of the right rotation test!")
                self.stop_generator = True

    def test_compactness(self, *args, **kwargs):
        compacness_limit = 1.0
        raw_data = self.data
        x, y, w, h, cat_ = self.bboxes[0]
        while True:
            alpha = np.sqrt(1 / compacness_limit)
            new_w = round(w * alpha)
            new_h = round(h * alpha)
            new_x = int(x - round((new_w - w) / 2))
            new_y = int(y - round((new_h - h) / 2))
            data = [raw_data[new_y:new_y + new_h, new_x:new_x + new_w]]

            self.limit = compacness_limit
            compacness_limit = round(compacness_limit - 0.05, 2)

            raw_gt = cat_

            yield data, raw_gt

            if compacness_limit <= 0.05:
                print("Reached the limit of the compactness test!")
                self.stop_generator = True

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

        if self.img_path is not None:
            file_name = os.path.split(self.img_path)[-1]
            _name, _extension = os.path.splitext(file_name)
        else:
            _name = ''.join(random.choices(
                string.ascii_uppercase + string.digits, 
                k=self.length_of_random_name
            ))
            _extension = '.jpg'

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

    def save_tested_image(self, img):
        if self.save_all_tested_images is False:
            return None

        if self.result_image_path is not None:
            dir_ = os.path.join(
                self.result_image_path, 
                'tested_images'
            )
        else:
            dir_ = 'tested_images'
        
        if os.path.isdir(dir_) is False:
            os.makedirs(dir_)
        
        _name = os.path.join(
            dir_,
            "{}.jpg".format(self.counter)
        )
        self.save_image(_name, img)
        self.counter += 1

        return True

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
        
        if self.report[self.option]['type'] == 4:
            suffix = self.report[self.option]['storage']['last']['value']
            img_names = self.save_images(data, type_data=str(suffix))
            self.save_gt(gt, img_names)
            self.save_dt(dt, img_names)

        else:
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
        if  self.report[option]['type'] == 4:
            self.report[option]['storage']['last']['value'] += 1
        else:
            self.report[option]['storage']['last']['value'] = self.limit

    def make_report(self, option, verbose=True):
        _mess = self.report[option]['message']
        _note = self.report[option]['note']
        _value = self.report[option]['storage']['last']['value']
        message = "{}: \n{} = {} \n({})".format(option,
                                                _mess,
                                                _value,
                                                _note)
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

        # if self.counter > 5:
        #     self.stop_generator = True
        #     self.test_failed = True

        if self.stop_generator is True:
            return False


        return True

    #======================================================
    #========================Process=======================
    def preprocess_input(self):
        # set param
        self.stop_generator = False
        self.stop_type_4 = False

        # check option
        assert (self.option in self.valid_option_list), 'Invalid option'

        # Create a folder to store result
        if self.result_image_path is not None:
            if not os.path.exists(self.result_image_path):
                os.makedirs(os.path.join(self.result_image_path, 'images'))
                os.makedirs(os.path.join(self.result_image_path, 'gt'))
                os.makedirs(os.path.join(self.result_image_path, 'dt'))

        self.height, self.width, _ = self.data.shape
    
    def get_generator(self, option):
        return self.report.get(option).get('generator')
    
    @abc.abstractmethod
    def create_original_input(self):
        # format dt
        data = self.data
        # format gt
        gt = self.gt
        return data, gt

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
            # save test image if necessary
            if self.save_all_tested_images:
                self.save_tested_image(img)

            predicted_result = inference_function(img)
            converted_result = convert_output_function(predicted_result)
            results.append(converted_result)

        # format predicted result
        dt = self.format_dt(results=results, gt=gt)

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
        dt = self.fit(
            inference_function, 
            convert_output_function, 
            data, 
            gt
        )

        metric = self.evaluate(gt, dt)

        # STEP 3: RUN TEST WITH CORRESPONDING OPTION
        # if model fails in evaluation of the original image, stop testing
        if self.check(metric, threshold, criterion):
            # Get the corresponding generator
            image_generator = self.get_generator(option)

            # if the option is belong to the group determining the deadpoint
            if self.report[self.option]['type'] != 4:
                while True:
                    self.backup_data(data, gt, dt)

                    # Create data which has format the corresponding option
                    data, gt = self.create_input(image_generator)
                    
                    # Conduct inference and format model results
                    dt = self.fit(
                        inference_function, 
                        convert_output_function, 
                        data, 
                        gt
                    )

                    metric = self.evaluate(gt, dt)

                    # Check end condition
                    if self.check(metric, threshold, criterion) is False:
                        self.update_report(option)
                        self.log(data, gt, dt)
                        break
            
            # if the option is belong to the group counting failed images
            elif self.report[self.option]['type'] == 4:
                while True:
                    # Create data which has format the corresponding option
                    data, gt = self.create_input(image_generator)
                    
                    # Conduct inference and format model results
                    dt = self.fit(
                        inference_function,
                        convert_output_function,
                        data,
                        gt
                    )

                    metric = self.evaluate(gt, dt)

                    # logging if model fails
                    if self.check(metric, threshold, criterion) is False:
                        self.update_report(option)
                        self.log(data, gt, dt)

                    # Check end condition
                    if self.stop_type_4 is True:
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
              save_all_tested_images=False,
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
        self.save_all_tested_images = save_all_tested_images

        self.test_transformation(
            inference_function,
            convert_output_function,
            option,
            criterion,
            threshold
        )
                
        return self.make_report(option, verbose)