
import torch

def softmax_to_label(y):
    # return torch.argmax(torch.nn.functional.softmax(y, dim=1), dim=1)
    return torch.argmax(y, dim=1)

def calc_iou(y, gt):
    n = gt.shape[0]
    iou = 0
    for k in range(n):
        u = y[k] + gt[k]
        i = y[k] + gt[k]
        u[u > 1] = 1
        i = i - u
        iou += torch.sum(i).float() / torch.sum(u)
    return iou / n
