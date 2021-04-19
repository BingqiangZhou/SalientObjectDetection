import time
import torch
import random
import torchvision

class Transfroms(torchvision.transforms.Compose):
    def __call__(self, *images):
        img_list = list(images) # 元组无法修改改为适用
        for transform in self.transforms:
            seed_time = int(round(time.time() * 1000)) # ms
            # seed = torch.random.initial_seed()
            for i in range(len(img_list)):
                random.seed(seed_time)              # 适配低版本torchvision
                torch.random.manual_seed(seed_time) # 适配高版本torchvision
                img_list[i] = transform(img_list[i])
        return img_list