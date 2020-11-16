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
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

# resize image with original ratio
def resizeImage(img, size):
    jpg = img
    shape = jpg.shape[:2]
    r = min(size[0] / shape[0], size[1] / shape[1])
    new_size = int(round(shape[0] * r)), int(round(shape[1] * r))
    border = int((size[0] - new_size[0]) / 2), int((size[1] - new_size[1]) / 2)
    jpg = cv2.resize(jpg, (new_size[1], new_size[0]))
    num = np.zeros((size[0], size[1], 3), np.uint8) + 255
    num[border[0]:new_size[0]+border[0], border[1]:new_size[1]+border[1]] = jpg 
    return num

# crop image function
def cropImage(image, c1, c2):
    num = np.zeros((c2[0]-c1[0], c2[1]-c1[1], 3), np.uint8)
    num = image[c1[0]:c2[0], c1[1]:c2[1]]
    return num

# scale reactangle function
def scaleRectangle(shape, c1, c2, factor = 1):
    squareImage = shape[0] * shape[1]
    squareRectangle = (c2[0] - c1[0]) * (c2[1] - c1[1])
    gain = (squareRectangle / squareImage)**factor
    dx = int(c1[0] * gain)
    dy = int(c1[1] * gain)
    do = min(dx, dy)
    c1 = c1[0] - do, c1[1] - do
    c2 = c2[0] + do, c2[1] + do
    return c1, c2


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

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']
factor = 0.5

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)
beautyNet=cv2.dnn.readNet(beautyModel,beautyProto)

video=cv2.VideoCapture(args.image if args.image else 0)
padding=20
while cv2.waitKey(1)<0 :
    hasFrame,frame=video.read()
    if not hasFrame:
        cv2.waitKey()
        break
    
    resultImg,faceBoxes=highlightFace(faceNet,frame)
    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:

        # image cropping
        c1, c2 = scaleRectangle(frame.shape, (faceBox[1], faceBox[0]), (faceBox[3], faceBox[2]), factor)
        face = cropImage(frame, c1, c2)
        face = resizeImage(face, (224, 224))
        ##cv2.imshow("Cropped face", face)

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

        cv2.putText(resultImg, f'{gender}, {age}, {beauty}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow("howbeautifulami", resultImg)
        cv2.imwrite("howbeautifulami.jpg", image)
