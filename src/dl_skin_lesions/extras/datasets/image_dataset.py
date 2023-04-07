from torch.utils.data import Dataset
import pandas as pd

from torchvision.transforms import transforms

class ImageDataset(Dataset):
    def __init__(self, dataset, csv, image_size):
        self.dataset = dataset
        self.idx_to_id = {i : id for i, id in enumerate(self.dataset.keys())}

        self.csv = csv
        self.csv['label'] = pd.Categorical(csv['dx']).codes
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])


    def __len__(self):
        return len(self.idx_to_id)

    def __getitem__(self, idx):

        image_name = self.get_image_name(idx)
        image = self.dataset[image_name]()
        label = self.csv[self.csv['image_id'] == image_name]['label'].values[0]

        if self.transform:
            image = self.transform(image)

        return image, int(label)

    def get_image_name(self, idx):
        return self.idx_to_id[idx]
    
