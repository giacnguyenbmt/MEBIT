from trafficlight import TrafficLight
from MEBIT import ClsfEvaluation

option = [
    "blurring",
    "increasing_brightness",
    "increasing_contrast",
    "decreasing_brightness",
    "decreasing_contrast",
    "down_scale",
    "crop",
    "left_rotation",
    "right_rotation",
    "compactness"
]

def inference_function(input):
    bbox_coords = [[0, 0, input.shape[1] - 1, input.shape[0] - 1]]
    labels, confs = trafficlight.predict(input, bbox_coords)
    print("===============")
    print(confs[0].round(2))
    return (labels, confs)

def convert_function(predicted_sample):
    return predicted_sample[0][0]

if __name__ == "__main__":
    trafficlight = TrafficLight()
    trafficlight.load_model('trafficlight.onnx')

    # image = "../data/test_tflight_1_eval/img/WhatsApp Image 2022-07-15 at 10.11.55 AM.jpeg"
    # gt_name = "../data/test_tflight_1_eval/gt/WhatsApp Image 2022-07-15 at 10.11.55 AM.txt"

    # evaluation = ClsfEvaluation.from_input_path(
    #     img_path=image, 
    #     gt_path=gt_name, 
    #     image_color='bgr'
    #     )

    # for i in range(7):
    #     result = evaluation.stats(inference_function,
    #                               convert_function,
    #                               option[i],
    #                               "accuracy",
    #                               0.5,
    #                               "result_folder",
    #                               False,
    #                               verbose=True)

    #====================================================================#
    image = "../data/rotate_compacness/expanding_traffic_light.png"
    gt_name = "../data/rotate_compacness/expanding_traffic_light.json"
    # image = "../data/rotate_compacness/red.jpg"
    # gt_name = "../data/rotate_compacness/red.json"
    # image = "../data/rotate_compacness/yellow.jpg"
    # gt_name = "../data/rotate_compacness/yellow.json"
    # image = "../data/rotate_compacness/green.jpg"
    # gt_name = "../data/rotate_compacness/green.json"
    image = "../data/rotation_compactness_data/image/17800.png"
    gt_name = "../data/rotation_compactness_data/gt/17800.json"
    image = "../data/projected_street_view/img/g02.jpg"
    gt_name = "../data/projected_street_view/gt/g02.json"

    evaluation = ClsfEvaluation.from_coco_input_path(
        img_path=image, 
        gt_path=gt_name, 
        image_color='bgr'
        )

    for i in range(8, 9):
        result = evaluation.stats(inference_function,
                                  convert_function,
                                  option[i],
                                  "accuracy",
                                  0.0,
                                  "result_folder",
                                  True,
                                  verbose=True)