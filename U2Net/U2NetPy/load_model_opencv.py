import cv2
import os
import numpy as np
from glob import glob


model_name = 'u2netp'
full_model_path = os.path.join(os.getcwd(), 'saved_models', model_name + '.onnx')
opencv_net = cv2.dnn.readNetFromONNX(full_model_path)

dir = r"E:\Workspaces\zbq\projects\U-2-Net-master\test_image"

for img_path in glob(dir+'\*.jpg'):

    # read the image
    input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    input_img = input_img.astype(np.float32)
    input_img = cv2.resize(input_img, (256, 256))
    # define preprocess parameters
    mean = np.array([0.485, 0.456, 0.406]) * 255.0
    scale = 1 / 255.0
    std = [0.229, 0.224, 0.225]
    # prepare input blob to fit the model input:
    # 1. subtract mean
    # 2. scale to set pixel values from 0 to 1
    input_blob = cv2.dnn.blobFromImage(
        image=input_img,
        scalefactor=scale,
        size=(400, 400),  # img target size
        mean=mean,
        swapRB=True,  # BGR -> RGB
        crop=True  # center crop
    )
    # 3. divide by std
    input_blob[0] /= np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    print(input_blob.shape)

    # img /= 255.0
    # img -= [0.485, 0.456, 0.406]
    # img /= [0.229, 0.224, 0.225]

    # set OpenCV DNN input
    opencv_net.setInput(input_blob)
    # OpenCV DNN inference
    out = opencv_net.forward()
    cv2.imshow("out", out[0][0])
    print("OpenCV DNN prediction: \n")
    print("* shape: ", out.shape)
    cv2.waitKey(0)
    # # get the predicted class ID
    # imagenet_class_id = np.argmax(out)
    # # get confidence
    # confidence = out[0][imagenet_class_id]
    # print("* class ID: {}, label: {}".format(imagenet_class_id, imagenet_labels[imagenet_class_id]))
    # print("* confidence: {:.4f}".format(confidence))