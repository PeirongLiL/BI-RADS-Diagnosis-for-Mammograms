from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
def clahe(img, clip):
    #contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=clip)
    cl = clahe.apply(np.array(img, dtype=np.uint8))
    return cl
class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None,transform2=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
        self.transform2=transform2

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item]).convert('L')

        # RGB为彩色图片，L为灰度图片
        cl = Image.fromarray(clahe(img, 1.0))
        # print(cl1.size)
        # cl2 = Image.fromarray(clahe(img, 2.0))
        label = self.images_class[item]
        img = self.transform(cl)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        # labels = torch.as_tensor(labels)
        labels=torch.tensor(np.array(labels))
        # return images[:,1,:,:].unsqueeze(1), labels
        # return images, labels

        return images, labels
