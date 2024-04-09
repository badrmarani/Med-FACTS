import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


def map_image_to_intensity_range(image, min_o, max_o, percentiles=0):
    # If percentile = 0 uses min and max. Percentile >0 makes normalisation more robust to outliers.
    if image.dtype in [np.uint8, np.uint16, np.uint32]:
        assert min_o >= 0, 'Input image type is uintXX but you selected a negative min_o: %f' % min_o
    if image.dtype == np.uint8:
        assert max_o <= 255, 'Input image type is uint8 but you selected a max_o > 255: %f' % max_o
    min_i = np.percentile(image, 0 + percentiles)
    max_i = np.percentile(image, 100 - percentiles)
    if (max_i - min_i) != 0:
        image = (np.divide((image - min_i), (max_i - min_i)) * (max_o - min_o) + min_o).copy()
        image[image > max_o] = max_o
        image[image < min_o] = min_o
    else:
        image = image
    return image

def normalize_image(img):
    img = np.float32(np.array(img))
    m = np.mean(img)
    s = np.std(img)
    if s == 0:
        s = 1e-06
    img = np.divide((img - m), s)
    return img

class CheXpert(Dataset):
    def __init__(self, csv_path, images_path, stage, target_class_name, args=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[(self.df["AP/PA"] == "AP") | (self.df["AP/PA"] == "PA")]
        self.df = self.df[self.df["Frontal/Lateral"] == "Frontal"]
        self.df["Path"] = self.df["Path"].str.replace(
            "CheXpert-v1.0-small/train",
            os.path.join(images_path, stage)
        )
        
        self.image_size = 320

        self.target_class_name = target_class_name
        self.stage = stage
        self.images_path = images_path

    @property
    def transform(self):
        if self.stage == "train":
            return transforms.Compose([
                transforms.ColorJitter(contrast=(0.8, 1.4), brightness=(0.8, 1.1)),
                transforms.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05), scale=(0.95, 1.05), fill=128),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),  
            ])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        path = self.df.iloc[index]["Path"]

        image = Image.open(path).convert("L")
        image = self.transform(image)
        image = normalize_image(image)
        image = map_image_to_intensity_range(image, -1, 1, percentiles=0.95)
        image = torch.from_numpy(image)
        
        # Get labels from the dataframe for current image
        label = self.df.iloc[index][self.target_class_name]
        label = torch.tensor(label)
        spurious_attr = self.df.iloc[index]["Contamination"]
        spurious_attr = torch.tensor(spurious_attr)

        # print("path", path)
        # print("image:", image.shape, type(image))
        # print("label:", label)
        # print("spurious_attr:", spurious_attr)

        return image, label, spurious_attr, path

def get_chexpert_datasets(args):
    images_path = args.train_dataset
    target_class_name = images_path.split("/")[4]
    datasets = []
    for stage in ["train", "valid", "test"]:
        csv_path = os.path.join(images_path, f"{stage}_df.csv")
        ds = CheXpert(
            csv_path=csv_path,
            images_path=images_path,
            stage=stage,
            target_class_name=target_class_name,
        )
        datasets.append(ds)
    return datasets
