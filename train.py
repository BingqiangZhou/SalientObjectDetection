import torch
from torch.utils import data
import torchvision.transforms as transforms

from dataset import SODDataset
from nerworks import SODNet
from utils import Transfroms, calc_iou, softmax_to_label

t = Transfroms([
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize([320, 320]),
    transforms.ToTensor(),
])

dataset = SODDataset("../sod/data", transfrom=t)
nums_data = len(dataset)
index = int(nums_data * 0.8)
test_dataset = data.Subset(dataset, range(index))
val_dataset = data.Subset(dataset, range(index, nums_data))

model = SODNet(in_channels=4, out_channels=2, down_method="pooling")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_func = torch.nn.CrossEntropyLoss()

gpu = 0
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

model = model.to(device)

test_dataloader = data.DataLoader(test_dataset, batch_size=4, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=4, shuffle=False)

nums_epoch = 10
for i in range(nums_epoch):
    for index, data in enumerate(test_dataloader):
        image, gt, salient_map = data
        gt = (gt * 255).long().squeeze_(dim=1)
        image = image.to(device)
        gt = gt.to(device)
        salient_map = salient_map.to(device)
        # print(torch.unique(gt), gt.shape)
        x = torch.cat((image, salient_map), dim=1)
        optimizer.zero_grad()
        y, pbr_y = model(x)
        # loss = loss_func(y, gt)
        loss = loss_func(y, gt) + loss_func(pbr_y, gt)
        loss.backward()
        optimizer.step()
        
        y_label = softmax_to_label(y)
        pbr_y_label = softmax_to_label(pbr_y)
        y_iou = calc_iou(y_label, gt)
        pbr_y_iou = calc_iou(pbr_y_label, gt)
        print(f"train: epoch {i+1}, step {index + 1}, loss: {loss}, iou: {y_iou}, pbr iou: {pbr_y_iou}.")
    torch.save(model.state_dict(), f"./models/checkpoints.pth")

    for index, data in enumerate(val_dataloader):
        image, gt, salient_map = data
        gt = (gt * 255).long().squeeze_(dim=1)
        image = image.to(device)
        gt = gt.to(device)
        salient_map = salient_map.to(device)
        # print(torch.unique(gt), gt.shape)
        x = torch.cat((image, salient_map), dim=1)
        optimizer.zero_grad()
        with torch.no_grad():
            y, pbr_y = model(x)
            loss = loss_func(y, gt) + loss_func(pbr_y, gt)
            y_label = softmax_to_label(y)
            pbr_y_label = softmax_to_label(pbr_y)
            y_iou = calc_iou(y_label, gt)
            pbr_y_iou = calc_iou(pbr_y_label, gt)
        print(f"val: step {index + 1}, loss: {loss}, iou: {y_iou}, pbr iou: {pbr_y_iou}.")
        # loss = loss_func(y, gt)
        