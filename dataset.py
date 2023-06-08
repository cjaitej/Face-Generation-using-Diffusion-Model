from torch.utils.data import Dataset
from PIL import Image
from utils import *

class FaceDataset(Dataset):
    def __init__(self, data_folder, transform):
        self.transform = transform
        with open(data_folder, 'r') as f:
            self.images = f.read().split('\n')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = Image.open(self.images[i])
        image = self.transform(image)
        return image

if __name__ == '__main__':
    dataset = FaceDataset('data_set.txt')
    for i in dataset:
        print(i.shape)