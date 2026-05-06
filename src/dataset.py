import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class MultimodalXRayDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.image_paths = self.dataframe['File_Path'].values
        self.tabular_data = self.dataframe[tabular_features].values.astype(np.float32)
        self.labels = np.stack(self.dataframe['Target_Vector'].values).astype(np.float32)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (320, 320))
            
        if self.transform:
            image = self.transform(image)
            
        tab_features = torch.tensor(self.tabular_data[idx])
        label = torch.tensor(self.labels[idx])
        return (image, tab_features), label
