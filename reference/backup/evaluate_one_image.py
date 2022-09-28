from trafficlight import TrafficLight
from metrics_ver2 import Evaluation

option = ["blurring", 
          "increasing_brightness", 
          "increasing_contrast",
          "decreasing_brightness", 
          "decreasing_contrast", 
          "down_scale",
          "crop"]

def inference_function(input):
    bbox_coords = [[0, 0, input.shape[1] - 1, input.shape[0] - 1]]
    labels, confs = trafficlight.predict(input, bbox_coords)
    return (labels, confs)

def convert_function(predicted_sample):
    return predicted_sample[0][0]

if __name__ == "__main__":
    image = "../test_tflight_1_eval/img/WhatsApp Image 2022-07-15 at 10.11.55 AM.jpeg"
    gt_name = "../test_tflight_1_eval/gt/WhatsApp Image 2022-07-15 at 10.11.55 AM.txt"

    trafficlight = TrafficLight()
    trafficlight.load_model('trafficlight.onnx')
    evaluation = Evaluation(img_path=image, gt_path=gt_name, image_color='bgr')
    for i in range(len(option)):
        # if i != 6:
        #     continue
        result = evaluation.clsf_stats(inference_function, 
                                       convert_function,
                                       option[i],
                                       "accuracy",
                                       0.5,
                                       verbose=True)


