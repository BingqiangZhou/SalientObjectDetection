import time
start = time.time()
import os
import cv2 as cv

import numpy as np
from PIL import Image
from glob import glob
from skimage import io
from copy import deepcopy
from crf import crf2d

from sod import U2Net

model_name = 'u2netp'
# model_name = 'u2net'
# model_name = "u2net_half"

def image_resize_by_fixed_max_side(image, max_side=400):
    h, w = image.shape[:2]
    ratio = max_side / max(h, w)
    image = cv.resize(image, (int(w * ratio), int(h * ratio)))
    return image

def predict_an_image(dir, image_name):
    # dir = r'E:\Datasets\sod_datasets\DUT-OMRON'
    # image_name = 'im005'
    # image_name = 'im010'
    image = np.array(Image.open(os.path.join(dir, image_name+'.jpg')))
    # label = np.array(Image.open(os.path.join(dir, image_name+'.png')))
    pred = U2Net(model_name, use_gpu=True).forward(image)
    # print(image.shape, label.shape)
    return pred

net = U2Net(model_name, use_gpu=True)
end = time.time()
print("load time: ", end - start)

def predict_image_from_dir(dir):
    image_list = glob(dir+'/*.jpg')
    for image_path in image_list:
        original_image = np.array(Image.open(image_path))
        h, w = original_image.shape[:2]
        # if max(h, w) > 400:
            # image = image_resize_by_fixed_max_side(image, max_side=400)

        if original_image.shape[-1] == 4:
            original_image = cv.cvtColor(original_image, cv.COLOR_RGBA2RGB)
        elif original_image.ndim == 2:
            original_image = cv.cvtColor(original_image, cv.COLOR_GRAY2RGB)
        # image = cv.resize(original_image, (288, 288))
        image = cv.resize(original_image, (400, 400))

        # label = np.array(Image.open(os.path.join(dir, image_name+'.png')))
        pred_start = time.time()
        pred = net.forward(image)
        pred_end = time.time()
        # cv.imshow("image", original_image[:, :, ::-1])
        # cv.imshow("pred", pred)
        # pred_mask = np.zeros((*image.shape[:2], 2))
        # pred_mask[:, :, 0] = 1 - pred
        # pred_mask[:, :, 1] = pred
        # cv.imshow("pred0", pred_mask[:, :, 0])
        # cv.imshow("pred1", pred_mask[:, :, 1])
        # pred_mask = crf2d(image, pred_mask)
        # pred_mask = (pred_mask*255).astype(np.uint8)
        # cv.imshow("mask", pred_mask)
        # print(np.unique(pred_mask))
        # cv.waitKey(0)
        print(image_path, (h, w), "-->", image.shape[:2], pred_end-pred_start)
        # sum_pred = np.zeros_like(pred)
        # for pred_map in pred_list:
        #     sum_pred += pred_map
        # # sum_pred /= 3
        # pred_mask = np.zeros_like(pred)
        # pred_mask[pred >= 0.3] = 255
        # pred_mask[pred < 0.3] = 0
        # pred_mask = (sum_pred * 255).astype(np.uint8)
        # e = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        # pred_mask = cv.morphologyEx(pred_mask, cv.MORPH_DILATE, e)
        # pred_mask = cv.morphologyEx(pred_mask, cv.MORPH_OPEN, e)
        # pred_mask = cv.GaussianBlur(pred_mask, (3, 3), 0)
        # pred_mask = cv.convertScaleAbs(pred_mask)
        # _, pred_mask = cv.threshold(pred_mask, 0, 255, cv.THRESH_OTSU)
        # _, pred_mask = cv.threshold(pred_mask, 0, 255, cv.THRESH_TRIANGLE)
        # _, pred_mask_0 = cv.threshold(pred_mask, 125, 255, cv.THRESH_BINARY)
        # _, pred_mask_1 = cv.threshold(pred_mask, 100, 255, cv.THRESH_BINARY)
        # _, pred_mask_2 = cv.threshold(pred_mask, 75, 255, cv.THRESH_BINARY)
        # _, pred_mask_3 = cv.threshold(pred_mask, 50, 255, cv.THRESH_BINARY)
        # _, pred_mask_4 = cv.threshold(pred_mask, 25, 255, cv.THRESH_BINARY)
        # _, pred_mask_5 = cv.threshold(pred_mask, 0, 255, cv.THRESH_BINARY)
        # pred_mask_6 = pred_mask_0 + pred_mask_1 + pred_mask_2 + pred_mask_3 + pred_mask_4 + pred_mask_5
        # pred_mask_6 = pred_mask_1 + pred_mask_3 + pred_mask_5
        # pred_mask_6 = (pred_mask_2 + pred_mask_3 + pred_mask_4)
        # _, pred_mask = cv.threshold(pred_mask_6, 0, 255, cv.THRESH_OTSU)
        # contours, hierarchy = cv.findContours(pred_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # print(len(contours))
        # pred_mask = cv.drawContours(image, contours, -1, (0, 0, 255))
        # cv.imshow("c", pred_mask)
        # cv.waitKey(0)
        
        pred_mask = cv.resize(pred, (w, h))
        pred_mask = cv.convertScaleAbs(pred_mask)
        _, pred_mask = cv.threshold(pred_mask, 0, 255, cv.THRESH_OTSU)
        pred_mask[pred_mask > 1e-2] = 255
        out_path = image_path.replace(".jpg", ".tif")
        out_mask_path = image_path.replace(".jpg", "_mask.tif")
        cv.imwrite(out_path, pred)
        cv.imwrite(out_mask_path,  pred_mask)
        del pred, image
        # break

dir = r"E:\Workspaces\zbq\projects\U-2-Net-master\test_image"
# dir = r"E:\Workspaces\zbq\projects\U-2-Net-master\val_image"
predict_image_from_dir(dir)

# dir = r'E:\Workspaces\zbq\projects\U-2-Net-master\test_image'
# image_name = '12'
# i1 = io.imread(os.path.join(dir, image_name+'.jpg'))
# print(type(Image.open(os.path.join(dir, image_name+'.jpg'))))
# i2 = np.array(Image.open(os.path.join(dir, image_name+'.jpg')))

# print(np.any((i1 - i2)), i2.shape)