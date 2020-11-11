import cv2
import numpy as np
import onnxruntime
from utils.box_utils import predict
from utils.image_utils import resizeImage, cropImage, scaleRectangle
import argparse

# onnx models and params
face_model = "models/face_detector_ultra_light_640_freeze.onnx"
age_model = "models/vgg_ilsvrc_16_age_imdb_wiki.onnx"
gender_model = "models/vgg_ilsvrc_16_gender_imdb_wiki.onnx"
beauty_model = "models/beauty_resnet18.onnx"
factor = 0.5

# graphics 
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2
point = (5, 50)

# prepare gender, age and beauty inference sessions
face_session = onnxruntime.InferenceSession(face_model)
gender_session = onnxruntime.InferenceSession(gender_model)
age_session = onnxruntime.InferenceSession(age_model)
beauty_session = onnxruntime.InferenceSession(beauty_model)

# face detection
def faceDetection(raw_img):
    input_name = face_session.get_inputs()[0].name
    h, w, _ = raw_img.shape
    img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 480))
    img_mean = np.array([127, 127, 127])
    img = (img - img_mean) / 128
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)
    confidences, boxes = face_session.run(None, {input_name: img})
    boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)
    return boxes, labels, probs

# face beauty
def faceBeauty(raw_img):
    img = resizeImage(raw_img, (224, 224))
    img_mean = np.array([104, 117, 123])
    img = (img - img_mean) / 255
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    input_name = beauty_session.get_inputs()[0].name
    outputs = beauty_session.run(None, {input_name: img})[0][0]
    beauty = round(2.0 * sum(outputs), 1)
    return beauty

# face gender
def faceGender(raw_img):
    img = resizeImage(raw_img, (224, 224))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    input_name = gender_session.get_inputs()[0].name
    outputs = gender_session.run(None, {input_name: img})[0][0]
    gender = 'Woman' if (np.argmax(outputs) == 0) else 'Man'
    return gender

# face age
def faceAge(raw_img):
    img = resizeImage(raw_img, (224, 224))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    input_name = age_session.get_inputs()[0].name
    outputs = age_session.run(None, {input_name: img})[0][0]
    age = round(sum(outputs * list(range(0, 101))), 1)
    return age

# parse args
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, required=False, help="input image")
    args = parser.parse_args()
    return args

# main
def main(args):
    # load image
    image_name = args.image
    if image_name is None: image_name = "images/charlize.jpg"
    image = cv2.imread(image_name)
    shape = image.shape
    print(f'Image: {image_name}')

    # face detection
    boxes, labels, probs = faceDetection(image)
    print(f'Detected faces: {boxes.shape[0]}\n------------------')

    # face analytics
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        c1, c2 = (box[0], box[1]), (box[2], box[3])
        c1, c2 = scaleRectangle(shape, c1, c2, factor)
        img = cropImage(image, (c1[1], c1[0]), (c2[1], c2[0]))

        gender = faceGender(img)
        age = faceAge(img)
        beauty = faceBeauty(img)
        labels = 'Gender: {}'.format(gender) + '\n' + 'Age: {}'.format(age) + '\n' + 'Beauty: {}'.format(beauty)
        print(f'{labels}\n------------------\n')
        
        for i, label in enumerate(labels.split('\n')):
            y = point[1] + i * point[1]
            cv2.putText(img, label, (point[0], y), font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.rectangle(image, c1, c2, color, thickness=2)

    # save produced image
    cv2.imwrite("processed.jpg", image)

if __name__ == '__main__':
    main(parse_args())
    
