import cv2
import numpy as np
import onnxruntime as ort

class_name = ['red', 'yellow', 'green']

class TrafficLight():  
    def __init__(self):
        super(TrafficLight, self).__init__()

    def load_model(self, model_path):
        self.sess = ort.InferenceSession(model_path)
        # self.sess = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])

    def predict(self, img, bbox_coords, conf_thresh=(0.8,0.8,0.8),verbose=0):
        labels, confs = [], []
        for x,y,w,h in bbox_coords:
            bbox_crop = img[y:y+h,x:x+w]
            bbox_tf = self.parse(bbox_crop, (75, 75))
            
            predicted_proba = self.sess.run(None, {"input_1": np.array([bbox_tf])})[0]
            label = 'n'
            for _, predicted in enumerate(predicted_proba):
                if predicted[1] > conf_thresh[1]:
                    label = 'y'
                elif predicted[2] > conf_thresh[2]:
                    label = 'g'
                elif predicted[0] > conf_thresh[0]:
                    label = 'r'
            labels.append(label)
            confs.append(predicted)
        return labels, confs
    
    def draw(self, img, bbox_coords, labels, confs):
        for _,((x,y,w,h),label,conf) in enumerate(zip(bbox_coords,labels,confs)):
            color = ()
            label_txt = ''
            if (label == 'r'): 
                color = (0,0,255)
                label_txt = 'red'
            elif (label == 'y'): 
                color = (0,255,255)
                label_txt = 'yellow'
            elif (label == 'g'): 
                color = (0,255,0)
                label_txt = 'green'
            elif (label == 'n'): 
                color = (50,50,50)
                label_txt = 'none'
            conf_txt = 'r'+str(round(conf[0],2))+' y'+str(round(conf[1],2))+' g'+str(round(conf[2],2))
            img = cv2.rectangle(img, (x,y),(x+w,y+h),color,4)
            img = cv2.putText(img,label,(x,y-30),0,0.8,color, thickness=2)
            img = cv2.putText(img,conf_txt,(x,y-10),0,0.6,color, thickness=2)
        return img
        
    def parse(self, img, input_size):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, input_size, interpolation = cv2.INTER_AREA)
        img = img.astype(np.float32)
        img = img / 255.0
        return img

if __name__ == '__main__':
    bbox_coords = [(74, 134, 151, 367),
                  (298, 606, 110, 276),
                  (548, 611, 83, 158),
                  (757, 293, 70, 136),
                  (650, 339, 73, 129),
                  (613, 159, 50, 105),
                  (1107, 392, 126, 219),
                  (1132, 616, 81, 140),
                  (1535, 579, 103, 199),
                  (1356, 268, 64, 126),
                  (1340, 101, 59, 104),
                  (1615, 53, 171, 367),
                  (1787, 296, 113, 233)]

    trafficlight = TrafficLight()
    trafficlight.load_model('trafficlight.onnx')

    cap = cv2.VideoCapture('videoplayback.mp4')
    size = (int(cap.get(3)), int(cap.get(4)))
    result = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            print('Video is ended')
            break

        if (i>20): break
        
        labels, confs = trafficlight.predict(frame, bbox_coords)
        
        frame = trafficlight.draw(frame, bbox_coords, labels, confs)

        result.write(frame)

        # cv2.imshow('frame',frame)
        # if cv2.waitKey(1) & 0xFF == 27:
        #     break

        print(f'Done Frame{i}')
        i += 1
    cap.release()
    result.release()