import os
import torch
import numpy as np

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB


class U2Net():
    def __init__(self, model_name='u2netp', use_gpu=True):
        if('u2netp' in model_name):
            print("...load U2NEP---4.7 MB")
            self.net = U2NETP(3,1)
        elif('u2net' in model_name):
            print("...load U2NET---173.6 MB")
            self.net = U2NET(3,1)
        
        self.use_gpu = use_gpu
        model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + '.pth')
        if torch.cuda.is_available() and self.use_gpu:
            self.net.load_state_dict(torch.load(model_dir))
            self.net.cuda()
        else:
            self.net.load_state_dict(torch.load(model_dir, map_location='cpu'))
        self.net.eval()
    
    # normalize the predicted SOD probability map
    def __normPRED(self, d):
        ma = torch.max(d)   
        mi = torch.min(d)
        dn = (d-mi)/(ma-mi)
        return dn

    def forward(self, image):
        image = image/np.max(image)
        image[:,:,0] = (image[:,:,0]-0.485)/0.229
        image[:,:,1] = (image[:,:,1]-0.456)/0.224
        image[:,:,2] = (image[:,:,2]-0.406)/0.225
        image = image.transpose((2, 0, 1))
        image = image[None, :, :, :]

        image_torch = torch.from_numpy(image).type(torch.FloatTensor)
        if torch.cuda.is_available() and self.use_gpu :
            image_torch = image_torch.cuda()
        
        with torch.no_grad():
            output = self.net(image_torch)
        
        pred = output[:,0,:,:]
        return self.__normPRED(pred).cpu().numpy()[0,:,:]