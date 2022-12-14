import os
import sys
import random
import string

import cv2

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def read_image(img_path, image_color):
    BGR_img = cv2.imread(img_path)
    if image_color == 'rgb':
        img = cv2.cvtColor(BGR_img, cv2.COLOR_BGR2RGB)
    elif image_color == 'bgr':
        img = BGR_img

    return img

def create_random_name(length_of_random_name=5, img_ext='.jpg'):
    _name = ''.join(random.choices(
            string.ascii_uppercase + string.digits, 
            k=length_of_random_name
        ))
    _extension = img_ext
    return _name + _extension