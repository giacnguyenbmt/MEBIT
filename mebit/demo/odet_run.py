import os

import numpy as np
import tensorflow as tf

from ..object_detection import ODetEvaluation

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White
# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
# MODEL_NAME = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
model_path = "./"
PATH_TO_CKPT = model_path + MODEL_NAME + '/frozen_inference_graph.pb'

def download_model():
    import six.moves.urllib as urllib
    import tarfile

    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())

def load_graph():
    if not os.path.exists(PATH_TO_CKPT):
        download_model()

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    return detection_graph

def select_boxes(boxes, classes, scores, score_threshold=0.0, target_class=10):
    """

    :param boxes:
    :param classes:
    :param scores:
    :param target_class: default traffic light id in COCO dataset is 10
    :return:
    """

    sq_scores = np.squeeze(scores)
    sq_classes = np.squeeze(classes)
    sq_boxes = np.squeeze(boxes)

    # sel_id = np.logical_and(sq_classes == target_class, sq_scores > score_threshold)
    sel_id = sq_scores > score_threshold

    return sq_boxes[sel_id], sq_scores[sel_id], sq_classes[sel_id]
    # return sq_boxes, sq_classes, sq_boxes

class TLClassifier(object):
    def __init__(self):

        self.detection_graph = load_graph()
        self.extract_graph_components()
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)

        # run the first session to "warm up"
        dummy_image = np.zeros((100, 100, 3))
        self.detect_multi_object(dummy_image,0.1)
        self.traffic_light_box = None
        self.classified_index = 0

    def extract_graph_components(self):
        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
    
    def detect_multi_object(self, image_np, score_threshold):
        """
        Return detection boxes in a image

        :param image_np:
        :param score_threshold:
        :return:
        """

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.

        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        sel_boxes, sel_scores, sel_classes = select_boxes(boxes=boxes, classes=classes, scores=scores,
                                                          score_threshold=score_threshold, target_class=10)

        return sel_boxes, sel_scores, sel_classes

def inference_function(input):
    boxes, scores, classes = tlc.detect_multi_object(input, score_threshold=0.2)
    ih, iw, _ = input.shape
    formated_boxes = np.empty_like(boxes)

    formated_boxes[:, 0] = boxes[:, 1] * iw
    formated_boxes[:, 1] = boxes[:, 0] * ih
    formated_boxes[:, 2] = (boxes[:, 3] - boxes[:, 1]) * iw
    formated_boxes[:, 3] = (boxes[:, 2] - boxes[:, 0]) * ih
    return formated_boxes, scores, classes

def convert_output_function(predicted_sample):
    boxes, scores, classes = predicted_sample
    dt = {
        'boxes': boxes,
        'scores': scores,
        'classes': classes
    }
    return dt


img = 'data/object_detection/000000162752.jpg'
gt = 'data/object_detection/coco_trafficlight.json'

option = ["blurring",
          "increasing_brightness",
          "increasing_contrast",
          "decreasing_brightness",
          "decreasing_contrast",
          "down_scale",
          "crop",
          "rotate90",]

tlc = TLClassifier()

evaluator = ODetEvaluation.from_input_path(img, gt)

for i in range(len(option)):
    evaluator.stats(inference_function,
                    convert_output_function,
                    option[i],
                    'ap_50',
                    0.5,
                    "result_folder",
                    False,
                    verbose=True)