# HowBeautifulAmI
Python application for face detection and prediction age, gender and face beauty.

## Dependencies
Install python 3+ dependencies
- opencv-python  
- numpy  
- argparse

## How to Use
```
usage: python howbeautifulami.py [-i] --input_image
```
<p align="center"><img width="40%" src="examples/charlize.jpg"/><img width="40%" src="examples/michael.jpg"/></p>   

Results for **Charlize Theron** photo
```
Image: images/charlize.jpg
Detected faces: 1
Gender: Woman
Age: 37.3
Beauty: 8.6
```
and for **Michael Fassbender** photo
```
Image: images/michael.jpg
Detected faces: 1
Gender: Man
Age: 41.1
Beauty: 8.0
```

## License
**MIT**

## References
* Lapuschkin et al. - Understanding and Comparing Deep Neural Networks for Age and Gender Classification - [Data and Models](https://github.com/sebastian-lapuschkin/understanding-age-gender-deep-learning-models).
* Linzaer - [Ultra-lightweight face detection model](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB).
* Rothe et al. - [IMDB-WIKI – 500k+ face images with age and gender labels](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/).
* A diverse benchmark database for multi-paradigm facial beauty prediction - [SCUT-FBP5500](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release).
* Microsoft - [ONNX Runtime](https://github.com/microsoft/onnxruntime).
* Caffe to ONNX: [unofficial converter](https://github.com/asiryan/caffe-onnx).
