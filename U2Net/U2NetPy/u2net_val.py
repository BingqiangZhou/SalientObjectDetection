import os
import tqdm
import glob
import torch
import argparse
import numpy as np
import torchmetrics
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader, random_split

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from nets import get_net

# ------- config --------

parser = argparse.ArgumentParser(description='val u2net')
parser.add_argument('-n', '--model_name', type=str, choices=['u2net', 'u2netp', 'u2net_groupconv', 'u2net_dsconv'], default="u2net_groupconv")
parser.add_argument('-d', '--device_id_str', type=str, default='0')
parser.add_argument('-i', '--num_val_images', type=int, default=1000)
parser.add_argument('-p', '--is_parallel_model', type=bool, default=True)
args = parser.parse_args()

device_ids = [i for i in range(len(args.device_id_str.split(',')))]
os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id_str

model_name = args.model_name
is_parallel_model = args.is_parallel_model
num_val_images = args.num_val_images

print("args:", args)

# ------- set the directory of training dataset --------

data_dir = os.path.join('bingqiangzhou/projects/sod/datas/val' + os.sep)
tra_image_dir = os.path.join('images' + os.sep)
tra_label_dir = os.path.join('gts' + os.sep)

image_ext = '.jpg'
label_ext = '.png'

model_dir = os.path.join(os.getcwd(), 'models', model_name + os.sep)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

train_num = 0

tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)

tra_lbl_name_list = []
for img_path in tra_img_name_list:
	img_name = os.path.splitext(os.path.split(img_path)[-1])[0]
	tra_lbl_name_list.append(data_dir + tra_label_dir + img_name + label_ext)

print("---")
print("val images: ", len(tra_img_name_list))
print("val labels: ", len(tra_lbl_name_list))
print("---")
train_num = len(tra_img_name_list)

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        # RescaleT(288),
        RescaleT(320),
        RandomCrop(288),
        ToTensorLab(flag=0)]))

num_images = len(salobj_dataset)
# num_val_images = 1000
print(f"random sample {num_val_images} images")
datasets = random_split(salobj_dataset, [num_val_images, num_images - num_val_images])
salobj_dataloader = DataLoader(datasets[0], batch_size=1, shuffle=True, num_workers=1)

# ------- define model --------
# define the net
net = get_net(model_name, False, only_return_d0=True)
model_paths = glob.glob(model_dir + os.sep + "*.pth")

def remove_module_str_in_state_dict_key(state_dict):
    new_state_dict = {}
    for key in state_dict.keys():
        if key[:7] == 'module.':
            new_state_dict.update({key[7:]: state_dict[key]}) # remove "module." in key
        else:
            new_state_dict.update({key: state_dict[key]})
    return new_state_dict

print("---")
print("start val...")
num_models = len(model_paths)
print("num of models: ", len(model_paths))
print("---")

mean_iou_list = []
for index, load_model_path in enumerate(model_paths):
    if torch.cuda.is_available():
        state_dict = torch.load(load_model_path)
        if is_parallel_model:
            state_dict = remove_module_str_in_state_dict_key(state_dict)
        net.load_state_dict(state_dict)
        print("load parameters from: ", load_model_path)
        net.cuda()
    else:
        state_dict = torch.load(load_model_path, map_location='cpu')
        if is_parallel_model:
            state_dict = remove_module_str_in_state_dict_key(state_dict)
        net.load_state_dict(state_dict)
        print("load parameters from: ", load_model_path)

    net.eval()

    iou_list = []
    for i, data in enumerate(tqdm.tqdm(salobj_dataloader, desc=f"valing - {index+1}/{num_models}")):

        inputs, labels = data['image'], data['label']

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        with torch.no_grad():
            d0 = net(inputs_v)
        
        iou = torchmetrics.functional.iou(d0, labels_v.long())
        # print(f"image {i}: iou {iou}.")
        iou_list.append(iou.item())

        # del temporary outputs
        del d0
    mean_iou = np.mean(iou_list)
    mean_iou_list.append(mean_iou)
    print("mean iou:", mean_iou, 
            ", number of images which iou more than 90%:", np.sum(np.array(iou_list)>0.9))
    print('---')

best_index = np.argmax(mean_iou_list)
print(f"best mean iou model: {model_paths[best_index]}, mean iou: {mean_iou_list[best_index]}" )