import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from skimage import io

def write_csv(root_folder):
    df = pd.DataFrame()
    file_names = []
    labels = []
    label_names = []
    for idx, label_folder in enumerate(os.listdir(root_folder)):
        file_names.extend(os.listdir(label_folder))
        labels.append(idx)
        label_names.append(label_folder)
    df['file'] = file_names
    df['lables'] = labels
    df['label_name'] = label_names
    print(df.head(5))
    return df


class Custom_Image_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.annotations = write_csv(root_dir)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        class_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 2])
        file_path = os.path.join(class_path, self.annotations.iloc[idx, 0])
        img = io.imread(file_path)
        y_label = torch.tensor(int(self.annotations.iloc[idx,1]))

        if self.transform:
            image =self.transform(image)
        return (image, y_label)




