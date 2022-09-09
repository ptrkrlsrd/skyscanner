import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
import pandas as pd
from skimage import io, transform

class CatDog(Dataset):
    def __init__(self, root, transform=None):
        self.images = os.listdir(root)
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        file = self.images[index]
        img = np.array(Image.open(os.path.join(self.root, file)))

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        if "dog" in file:
            label = 1
        elif "cat" in file:
            label = 0
        else:
            label = -1

        return img, label

class SkyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.image_frame.iloc[idx, 0])
        image = np.array(io.imread(img_name))
        photogene = self.image_frame.iloc[idx, 1]

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, photogene
