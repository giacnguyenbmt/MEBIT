import albumentations as A

from . import base_transforms as T

def blur(blur_limit, *args, **kwargs):
    data = {'image': None}
    for key in data.keys():
        if key in kwargs:
            data[key] = kwargs[key]

    transform = A.Compose([
        A.Blur(blur_limit=(blur_limit, blur_limit + 1), p=1.0),
    ])
    transformed = transform(**data)
    return transformed

def brightness(brightness_limit, *args, **kwargs):
    data = {'image': None}
    for key in data.keys():
        if key in kwargs:
            data[key] = kwargs[key]

    transform = A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=[brightness_limit, 
                                brightness_limit+0.001],
            contrast_limit=0, 
            brightness_by_max=True, 
            always_apply=False, 
            p=1.0),
    ])
    transformed = transform(**data)
    return transformed

def contrast(contrast_limit, *args, **kwargs):
    data = {'image': None}
    for key in data.keys():
        if key in kwargs:
            data[key] = kwargs[key]

    transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0, 
            contrast_limit=[contrast_limit, contrast_limit + 0.001], 
            brightness_by_max=True, 
            always_apply=False, 
            p=1.0),
    ])
    transformed = transform(**data)
    return transformed

def crop(x_min, y_min, x_max, y_max, *args, **kwargs):
    data = {
        'image': None,
        'masks': None,
        'keypoints': None,
        'bboxes': None
    }
    for key in data.keys():
        if key in kwargs:
            data[key] = kwargs[key]

    transform = A.Compose(
        [A.Crop(x_min, y_min, x_max, y_max)], 
        keypoint_params=A.KeypointParams(format='xy', 
                                        remove_invisible=False),
        bbox_params=A.BboxParams(format='coco')
    )
    transformed = transform(**data)
    return transformed

def resize(ratio, *args, **kwargs):
    data = {
        'image': None,
        'masks': None,
        'keypoints': None,
        'bboxes': None
    }
    for key in data.keys():
        if key in kwargs:
            data[key] = kwargs[key]

    old_h, old_w, _ = data['image'].shape
    h = int(old_h * ratio)
    w = int(old_w * ratio)

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
    transformed = transform(**data)
    return transformed

def flip_rorate90(rotate_k=1, flip=False, *args, **kwargs):
    data = {
        'image': None,
        'masks': None,
        'keypoints': None,
        'bboxes': None
    }
    for key in data.keys():
        if key in kwargs:
            data[key] = kwargs[key]

    transform_list = [
        T.Rotate90(k=rotate_k, always_apply=True, p=1.0)
    ]
    if flip:
        transform_list.append(A.HorizontalFlip(p=1.0))

    transform = A.Compose(
        transform_list, 
        keypoint_params=A.KeypointParams(format='xy', 
                                            remove_invisible=False),
        bbox_params=A.BboxParams(format='coco')
    )
    transformed = transform(**data)
    return transformed

def rotate(limit=90, 
           interpolation=1, 
           border_mode=4, 
           value=None, 
           mask_value=None, 
           method='largest_box', 
           crop_border=False, 
           always_apply=True, 
           p=1.0, 
           *args, 
           **kwargs):
    data = {
        'image': None,
        'masks': None,
        'keypoints': None,
        'bboxes': None
    }
    for key in data.keys():
        if key in kwargs:
            data[key] = kwargs[key]

    transform = A.Compose(
        [A.Rotate(
            limit=[limit, limit], 
            interpolation=interpolation, 
            border_mode=border_mode, 
            value=value, 
            mask_value=mask_value, 
            rotate_method=method, 
            crop_border=crop_border, 
            always_apply=always_apply, 
            p=p
        )], 
        keypoint_params=A.KeypointParams(format='xy', 
                                            remove_invisible=False),
        bbox_params=A.BboxParams(format='coco')
    )
    transformed = transform(**data)
    return transformed
