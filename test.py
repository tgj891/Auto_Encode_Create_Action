import net as net
import torch
from PIL import Image
import numpy as np
from torch.autograd import Variable
import os

if __name__ == '__main__':
    net = net.Generator_Net()
    if torch.cuda.is_available():
        net = net.cuda()
    net.eval()

    parampath = r"C:\Users\liev\Desktop\myproject\create_action\netparam.pkl"
    if os.path.exists(parampath):
        net.load_state_dict(torch.load(parampath))

    # filepath = r"C:\Users\liev\Desktop\data\move\okimg\0.jpg"
    filepath = r"C:\Users\liev\Desktop\data\move\input\158.jpg"
    imgdata = (np.array(Image.open(filepath), dtype=np.float32) / 255. - 0.5)*2
    imgdata = np.transpose(imgdata, [2, 0, 1])
    imgdata = np.expand_dims(imgdata, axis=0)
    imgdata = torch.FloatTensor(imgdata)
    imgdata = Variable(imgdata)
    if torch.cuda.is_available():
        imgdata = imgdata.cuda()

    count = 100
    imglist = []
    tempdata = imgdata
    for i in range(count):
        out = net(tempdata)
        tempdata = out
        imglist.append(tempdata)

    for i, img in enumerate(imglist):
        img = img.cpu().data.numpy()
        img = np.array((img[0]/2 + 0.5)*255, dtype=np.uint8)
        img = img.transpose((1, 2, 0))  # 轴变换
        img = Image.fromarray(img)
        img.save("./people_%d.jpg"%i, "JPEG")
