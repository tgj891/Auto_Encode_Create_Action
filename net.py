import torch, math
import torch.nn as nn
import data as mydata
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
import os,shutil
import torch.nn.functional as F
import numpy as np
from PIL import Image

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

class Generator_Net(nn.Module):
    def __init__(self):
        super(Generator_Net, self).__init__()
        #128*128 1152
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
            , nn.BatchNorm2d(64)
            , nn.ReLU(inplace=True)
            , nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
            , nn.BatchNorm2d(64)
            , nn.ReLU(inplace=True)
        )
        #64*64 2304
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            , nn.BatchNorm2d(128)
            , nn.ReLU(inplace=True)
            , nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)
            , nn.BatchNorm2d(128)
            , nn.ReLU(inplace=True)
        )
        #32*32 9216
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
            , nn.BatchNorm2d(256)
            , nn.ReLU(inplace=True)
            , nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            , nn.BatchNorm2d(256)
            , nn.ReLU(inplace=True)
            , nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            , nn.BatchNorm2d(256)
            , nn.ReLU(inplace=True)
            , nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)
            , nn.BatchNorm2d(256)
            , nn.ReLU(inplace=True)
        )
        #vgg19 前三层
        #16*16 32768
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
            , nn.BatchNorm2d(512)
            , nn.ReLU(inplace=True)
            , nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
            , nn.BatchNorm2d(512)
            , nn.ReLU(inplace=True)
        )
        #8*8
        self.fc1 = nn.Sequential(
            nn.Linear(512*8*8, 1024)
            , nn.ReLU(inplace=True)
            , nn.Dropout()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024)
            , nn.ReLU(inplace=True)
            , nn.Dropout()
        )

        #65536
        self.fc3 = nn.Sequential(
            nn.Linear(1024, 512*8*8)
            , nn.ReLU(inplace=True)
            , nn.Dropout()
        )
        # 8*8
        self.rconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1)
            , nn.BatchNorm2d(512)
            , nn.ReLU(inplace=True)
            , nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
            , nn.BatchNorm2d(256)
            , nn.ReLU(inplace=True)
        )
        # 16*16
        self.rconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1)
            , nn.BatchNorm2d(256)
            , nn.ReLU(inplace=True)
            , nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1)
            , nn.BatchNorm2d(256)
            , nn.ReLU(inplace=True)
            , nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1)
            , nn.BatchNorm2d(256)
            , nn.ReLU(inplace=True)
            , nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
            , nn.BatchNorm2d(128)
            , nn.ReLU(inplace=True)
        )
        #32*32
        self.rconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
            , nn.BatchNorm2d(128)
            , nn.ReLU(inplace=True)
            , nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
            , nn.BatchNorm2d(64)
            , nn.ReLU(inplace=True)
        )

        #64*64
        self.rconv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
            , nn.BatchNorm2d(64)
            , nn.ReLU(inplace=True)
            , nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        )
        #128*128
        self.apply(weights_init)

    def forward(self, x):
        #encode
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv4 = conv4.view(-1, 512*8*8)
        fc1 = self.fc1(conv4)
        fc2 = self.fc2(fc1)

        #decode
        fc3 = self.fc3(fc2)
        fc3 = fc3.view(-1, 512, 8, 8)
        rconv1 = self.rconv1(fc3)
        rconv2 = self.rconv2(rconv1)
        rconv3 = self.rconv3(rconv2)
        rconv4 = self.rconv4(rconv3)
        out = F.tanh(rconv4)

        return out

if __name__ == '__main__':
    mydataset = mydata.Mydataset(
        r"C:\Users\liev\Desktop\data\move\input"
        , r"C:\Users\liev\Desktop\data\move\label"
    )
    dataloader = data.DataLoader(mydataset, batch_size=20, shuffle=True)
    parampath = r"C:\Users\liev\Desktop\myproject\create_action\netparam.pkl"
    tempparampath = r"C:\Users\liev\Desktop\myproject\create_action\netparam_temp.pkl"

    net = Generator_Net()
    if torch.cuda.is_available():
        net = net.cuda()
    net.train()

    if os.path.exists(parampath):
        net.load_state_dict(torch.load(parampath))

    optimer = optim.Adam(net.parameters(),lr=0.0002, betas=(0.5, 0.999))
    lossfun = nn.MSELoss()

    for epoch in range(10000000):
        xs, ys = mydataset.get_batch(dataloader)
        xs, ys = Variable(xs), Variable(ys)
        if torch.cuda.is_available():
            xs, ys = xs.cuda(), ys.cuda()

        out = net(xs)
        loss = lossfun(out, ys)

        optimer.zero_grad()
        loss.backward()
        optimer.step()

        if epoch % 10 == 0:
            if os.path.exists(tempparampath):
                shutil.copyfile(tempparampath, parampath)
            torch.save(net.state_dict(), tempparampath)

            img = out.cpu().data.numpy()
            img = np.array((img[0] / 2 + 0.5) * 255, dtype=np.uint8)
            img = img.transpose((1, 2, 0))  # 轴变换
            img = Image.fromarray(img)
            img.save("./train.jpg", "JPEG")

            print("epoch:", epoch, "loss:", loss.cpu().data.numpy())