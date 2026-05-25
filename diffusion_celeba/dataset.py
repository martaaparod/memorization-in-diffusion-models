import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as T


transform_celeba = T.Compose([
    T.Resize(64),
    T.CenterCrop(64),
    T.ToTensor(),
])

class CelebADataset(Dataset):
    def __init__(self, path='img_align_celeba', transform=transform_celeba):
        '''
        Args:
        path: str, path to folder containing .jpg CelebA images
        transform: preprocessing of images
        '''
        self.path = path
        self.transform = transform
        self.files = sorted(os.listdir(self.path))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.path, self.files[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


