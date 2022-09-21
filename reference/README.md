# Traffic Light Classifier
## Description
- A model and its implementation to classify traffic light
- Model input: 75px x 75px resized croped image contain a traffic light
- Model output: [red_conf yellow_conf green_conf]
    - Each value is float-type confidence value for each light [0..1]
    - The practical threshold should be 0.8 for each value
## Author
- Duong Tran-Thanh(duong.jt.19@gmail.com)
- Tung Le-Thanh (bluelul.email@gmail.com)
- Date: 18/07/2022
## How to run
```python
python trafficlight.py
```
This script will generate `output.mp4` result video, processed from `videoplayback.mp4` video
## Environment
```
python == 3.9.12
tensorflow == 2.9.1
opencv == 4.5.5

cudatoolkit == 11.2 
cudnn == 8.1.0
```
## Model Spec
- MSE = 0.03 on test dataset
- ACC > 0.98 on test dataset
- Speed: 450 fps on T4, 
- Size: 948kB
## Appendix
- We include a mini model and its un-clean implementation in `\appendix` folder, only detect whether a light is on or off
- This mini model is used to support the color classifier model in some specific cases
- Spec: acc 0.9955, f1 0.994, size 219kB

## Demo Output
![Demo Output](demoimg/demooutput.png)

## Works on night
![Works on night](demoimg/nightdemo.png)