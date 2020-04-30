from torchvision.datasets import VisionDataset
from PIL import Image

import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class PACS(VisionDataset):
    def __init__(self, root, domain, transform=None, target_transform=None):
        super(PACS, self).__init__(root, transform=transform, target_transform=target_transform)
        
        self.dataset = {}
        self.counter = 0
        self.classes = os.listdir(root + '/' + domain)
        self.classes_dict = {}
        class_counter = 0
        
        for class_ in self.classes:
            self.classes_dict[class_] = class_counter
            class_path = root+'/'+domain+'/'+class_
            images = os.listdir(class_path)
            for image in images:
                image_path = root+'/'+domain+'/'+class_+'/'+image
                self.dataset[self.counter] = (pil_loader(image_path),class_counter)
                self.counter += 1
            class_counter += 1

    def __getitem__(self, index):
        image, label = self.dataset[index]

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label
    
    def __len__(self):
        return self.counter

    def __getsplit__(self, train_size = 0.5):
        images, labels = [], []
        sss = StratifiedShuffleSplit(1,train_size=train_size)

        for item in self.dataset.values():
            images.append(item[0])
            labels.append(item[1])

        for x, y in sss.split(images,labels):
            train_indexes = x
            test_indexes = y 

        return train_indexes, test_indexes