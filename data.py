import numpy as np
import torch
from torch.utils import data
import os
from PIL import Image

def pic_handle():
    filepath = r"C:\Users\liev\Desktop\data\move\woman"
    # savepath = r"C:\Users\liev\Desktop\data\move\label"

    filelist = os.listdir(filepath)
    for i, file in enumerate(filelist):
        strs = file.split(".")
        num = int(strs[0])
        filedir = os.path.join(filepath, file)
        img = Image.open(filedir)
        # img = img.resize((128, 128))
        img.save(os.path.join(filepath, "%d.jpg"%(num-1)))

class Mydataset(data.Dataset):
    def __init__(self, inputimgpath, labelimgpath):
        self.inputimgpath = inputimgpath
        self.labelimgpath = labelimgpath
        self.dataset = os.listdir(inputimgpath)

    def __getitem__(self, index):
        filename = self.dataset[index].strip()
        filepath = os.path.join(self.inputimgpath, filename)
        imgdata = (np.array(Image.open(filepath), dtype=np.float32)/255. - 0.5)*2
        imgdata = np.transpose(imgdata, [2, 0, 1])

        labelfilepath = os.path.join(self.labelimgpath, filename)
        labelimgdata = (np.array(Image.open(labelfilepath), dtype=np.float32) / 255. - 0.5)*2
        labelimgdata = np.transpose(labelimgdata, [2, 0, 1])

        return imgdata, labelimgdata

    def __len__(self):
        return len(self.dataset)

    def get_batch(self, loader):
        ite = iter(loader)
        xs, ys = ite.next()
        return xs, ys

if __name__ == '__main__':
    pic_handle()
    # mydataset = Mydataset(
    #     r"C:\Users\liev\Desktop\data\move\input"
    #     ,r"C:\Users\liev\Desktop\data\move\label"
    # )
    # dataloader = data.DataLoader(mydataset, batch_size=5, shuffle=True)
    # print(mydataset.get_batch(dataloader))