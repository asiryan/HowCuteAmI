# HowBeatifulAmI
Python application for face detection and prediction age, gender and face beauty.

## Dependencies
Install python 3+ dependencies
- opencv-python  
- numpy  
- onnxruntime  
- argparse

## ONNX Models
| Model | Description | | Download | ONNX version | Opset version | Dataset |
|:-------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| face_detector_ultra_light_320 | Face detection | [1 MB](https://drive.google.com/file/d/1VfCqZgSFW7alMUfzxCnEH8r_NunOrqF2/view?usp=sharing)| 1.5 | 5 | - |
| face_detector_ultra_light_640 | Face detection | [2 MB](https://drive.google.com/file/d/16NfgL14WXYT2LPQiBYpOr42m-pqkeaaI/view?usp=sharing)| 1.5 | 5 | - |
| vgg_ilsvrc_16_age_chalearn_iccv2015 | Age classification | [513 MB](https://drive.google.com/file/d/1V75U1kUJ0udBLs6bg3lGqBk3ym8q9guV/view?usp=sharing) | 1.5 | 5 | ChaLearn LAP 2015 |
| vgg_ilsvrc_16_age_imdb_wiki | Age classification | [513 MB](https://drive.google.com/file/d/1ECle8EvsXiIid_vMa1_vwMJk6abhzrPF/view?usp=sharing)| 1.5 | 5 | IMDB-WIKI |
| vgg_ilsvrc_16_gender_imdb_wiki | Gender classification | [512 MB](https://drive.google.com/file/d/1epLM5ghucLcnGZg-NCIf1r16lotN004I/view?usp=sharing)| 1.5 | 5 | IMDB-WIKI |
| vgg_ilsvrc_16_gender_imdb_wiki | Gender classification | [512 MB](https://drive.google.com/file/d/1epLM5ghucLcnGZg-NCIf1r16lotN004I/view?usp=sharing)| 1.5 | 5 | IMDB-WIKI |
| beauty_alexnet | Beauty prediction | [217 MB](https://drive.google.com/file/d/1uXoP3XDx8s5oyo6VszOnG9qU_1P9Pik-/view?usp=sharing)| 1.5 | 5 | SCUT-FBP5500 |
| beauty_resnet18 | Beauty prediction | [43 MB](https://drive.google.com/file/d/1gFGDBdKdiW1LWHMiPLMx6Tt9Y_0l7sOM/view?usp=sharing)| 1.5 | 5 | SCUT-FBP5500 |
| beauty_resnext50 | Beauty prediction | [88 MB](https://drive.google.com/file/d/1gR8Rr6BcH1BlC_gRVbeou7oxJRQ4-wWK/view?usp=sharing)| 1.5 | 5 | SCUT-FBP5500 |
