import cv2
import math
import argparse
import numpy as np

# detect face
def highlightFace(net, frame, conf_threshold=0.95):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]

    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append(scale([x1,y1,x2,y2]))
            
    return faceBoxes

# scale current rectangle to box
def scale(box):
    width = box[2] - box[0]
    height = box[3] - box[1]
    maximum = max(width, height)
    dx = int((maximum - width)/2)
    dy = int((maximum - height)/2)
    
    bboxes = [box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy]
    return bboxes

# crop image
def cropImage(image, box):
    num = image[box[1]:box[3], box[0]:box[2]]
    return num

# main
parser=argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str, required=False, help="input image")
args=parser.parse_args()

faceProto="models/opencv_face_detector.pbtxt"
faceModel="models/opencv_face_detector_uint8.pb"
ageProto="models/age_googlenet.prototxt"
ageModel="models/age_googlenet.caffemodel"
genderProto="models/gender_googlenet.prototxt"
genderModel="models/gender_googlenet.caffemodel"
beautyProto="models/beauty_resnet.prototxt"
beautyModel="models/beauty_resnet.caffemodel"

MODEL_MEAN_VALUES=(104, 117, 123)
#MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']
color = (0,255,255)

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)
beautyNet=cv2.dnn.readNet(beautyModel,beautyProto)

frame=cv2.imread(args.image if args.image else 'images/charlize.jpg')
    
faceBoxes=highlightFace(faceNet,frame)
if not faceBoxes:
    print("No face detected")

for faceBox in faceBoxes:

    # face detection net
    face = cropImage(frame, faceBox)
    face = cv2.resize(face, (224, 224))

    # gender net
    blob=cv2.dnn.blobFromImage(face, 1.0, (224,224), MODEL_MEAN_VALUES, swapRB=False)
    genderNet.setInput(blob)
    genderPreds=genderNet.forward()
    gender=genderList[genderPreds[0].argmax()]
    print(f'Gender: {gender}')

    # age net
    ageNet.setInput(blob)
    agePreds=ageNet.forward()
    age=ageList[agePreds[0].argmax()]
    print(f'Age: {age[1:-1]} years')

    # beauty net
    blob=cv2.dnn.blobFromImage(face, 1.0/255, (224,224), MODEL_MEAN_VALUES, swapRB=False)
    beautyNet.setInput(blob)
    beautyPreds=beautyNet.forward()
    beauty=round(2.0 * sum(beautyPreds[0]), 1)
    print(f'Beauty: {beauty}/10.0')

    cv2.rectangle(frame, (faceBox[0],faceBox[1]), (faceBox[2],faceBox[3]), color, int(round(frame.shape[0]/400)), 8)
    cv2.putText(frame, f'{gender}, {age}, {beauty}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1.25, color, 2, cv2.LINE_AA)
    cv2.imshow("howbeautifulami", frame)
