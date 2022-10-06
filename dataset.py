import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch

class Covid_dataset(Dataset):
    def __init__(self, non_covid_anno, covid_anno, all_image_path, transform=None, target_transform=None):
        self.non_covid = pd.read_csv(non_covid_anno)["image_name"].tolist()
        self.covid = pd.read_csv(covid_anno)["image_name"].tolist()
        self.transform = transform
        self.target_transform = target_transform
        self.all_image_path = all_image_path
        self.non_covid_size = len(self.non_covid)

    def __len__(self):
        return len(self.covid) + len(self.non_covid)

    def __getitem__(self, idx):
        #未感染新冠的为0
        label = 0
        if (idx >= self.non_covid_size):
            label = 1
            _image_path = self.covid[idx - self.non_covid_size]
        else:
            _image_path = self.non_covid[idx]

        total_img_path = os.path.join(self.all_image_path, _image_path)
        image = read_image(total_img_path)
        if self.transform:
            image = self.transform(image)

        return image, label

# mydataset = Covid_dataset("/home/jas0n/PycharmProjects/covid_ct/COVID-CT/Data-split/non_covid.csv",
#                           "/home/jas0n/PycharmProjects/covid_ct/COVID-CT/Data-split/covid.csv",
#                           "/home/jas0n/PycharmProjects/covid_ct/COVID-CT/all_image")
# train_data, test_data = torch.utils.data.random_split(mydataset, [397,349])
# print(test_data)
