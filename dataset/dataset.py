import os
import cv2 as cv
from glob import glob
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class SODDataset(Dataset):
    def __init__(self, root, get_salient_map=True, transfrom=None):
        super().__init__()

        assert os.path.exists(root), f"{root} is not exist."

        self.images_dir = os.path.join(root, "images")
        self.gts_dir = os.path.join(root, "gts")

        self.image_path_list = glob(f"{self.images_dir}/*.jpg")
        self.gt_path_list = glob(f"{self.gts_dir}/*.png")

        assert len(self.image_path_list) == len(self.gt_path_list), f"the number of image and gt is not match."
        
        self.saliency = None
        if get_salient_map:
            self.saliency = cv.saliency.StaticSaliencyFineGrained_create()

        self.transfrom = transfrom

    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        image = Image.open(image_path)

        path, file_name = os.path.split(image_path)
        gt_path = os.path.join(self.gts_dir, file_name.replace("jpg", "png"))
        gt = Image.open(gt_path)

        if self.saliency is None:
            if self.transfrom is not None:
                image, gt = self.transfrom(image, gt)
            return image, gt
        
        success, salient_map = self.saliency.computeSaliency(np.array(image))
        temp = (salient_map*255).astype(np.uint8)
        threshold, salient_binary_map = cv.threshold(temp, 0, 255, cv.THRESH_OTSU)
        salient_map = salient_binary_map * salient_map
        if self.transfrom is not None:
            gt = Image.fromarray(np.array(gt)//255)
            salient_map = Image.fromarray(salient_map.astype(np.uint8))
            image, gt, salient_map = self.transfrom(image, gt, salient_map)
        return image, gt, salient_map
    
    def __len__(self):
        return len(self.image_path_list)