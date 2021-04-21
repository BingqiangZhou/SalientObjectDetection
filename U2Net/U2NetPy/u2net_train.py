import os
import glob
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from nets import get_net
from loss import muti_bce_loss_fusion

# ------- config --------

parser = argparse.ArgumentParser(description='train u2net')
parser.add_argument('-n', '--model_name', type=str, choices=['u2net', 'u2netp', 'u2net_groupconv', 'u2net_dsconv'], default="u2net_groupconv")
parser.add_argument('-d', '--device_id_str', type=str, default='0')
parser.add_argument('-e', '--epoch_num', type=int, default=100000)
parser.add_argument('-b', '--batch_size', type=int, default=16)
parser.add_argument('-s', '--save_frq', type=int, default=2000)
parser.add_argument('-p', '--pretrain', type=bool, default=True)
parser.add_argument('-m', '--pretrain_model_path', type=str, default=None)
args = parser.parse_args()

device_ids = [i for i in range(len(args.device_id_str.split(',')))]
os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id_str

epoch_num = args.epoch_num
batch_size_train = args.batch_size
save_frq = args.save_frq # save the model every 2000 iterations
num_workers = args.batch_size // 4 * len(device_ids)
num_workers = 1 if num_workers < 1 else num_workers
pretrain = args.pretrain
model_name = args.model_name
pretrain_model_path = args.pretrain_model_path

print("train net: ", model_name)
print("use device: ", args.device_id_str)
print("epoch_num: ", epoch_num)
print("batch_size: ", batch_size_train)
print("num_workers: ", num_workers)

# ------- set the directory of training dataset --------

data_dir = os.path.join('bingqiangzhou/projects/sod/datas/test' + os.sep)
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
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")
train_num = len(tra_img_name_list)

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(320),
        RandomCrop(288),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers)

# ------- define model --------
# define the net
net, load_model_path = get_net(model_name)
if os.path.exists(pretrain_model_path):
    load_model_path = pretrain_model_path

if torch.cuda.is_available():
    if pretrain:
        net.load_state_dict(torch.load(load_model_path))
        print("load parameters from: ", load_model_path)
    net.cuda()
else:
    if pretrain:
        net.load_state_dict(torch.load(load_model_path, map_location='cpu'))
        print("load parameters from: ", load_model_path)

if len(device_ids) > 1:
    net = torch.nn.DataParallel(net, device_ids=device_ids)

# ------- define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0

for epoch in range(0, epoch_num):
    net.train()

    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels = data['image'], data['label']

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

        loss.backward()
        optimizer.step()

        # # print statistics
        running_loss += loss.data.item()
        running_tar_loss += loss2.data.item()

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
        epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
        # torch.save(net.state_dict(), model_dir + model_name+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

        if ite_num % save_frq == 0:
            torch.save(net.state_dict(), model_dir + model_name +"_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # resume train
            ite_num4val = 0