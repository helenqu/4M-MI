from torchvision.datasets.vision import VisionDataset
from pathlib import Path
import numpy as np

class MIDataset(VisionDataset):
    def __init__(self, data_paths, transform=None, target_transform=None, transforms=None):
        #TODO: transform is not used
        super().__init__(Path(data_paths[0]).parent, transform=transform, target_transform=target_transform, transforms=transforms)

        images = []
        thetas = []
        noise = []
        cifar_labels = []
        ids = []

        for data_path in data_paths:
            data = np.load(data_path)
            images.append(data['X'])
            thetas.append(data['Y'])
            noise.append(data['Noise'])
            cifar_labels.append(data['labels'])
            ids.append([f"{int(data_path.stem.split('_')[-1])}_{i}" for i in range(data['X'].shape[0])]) # unique id = "fileidx_sampleidx"

        self.images = np.concatenate(images, axis=0)
        self.thetas = np.concatenate(thetas, axis=0)
        self.noise = np.concatenate(noise, axis=0)
        self.cifar_labels = np.concatenate(cifar_labels, axis=0)
        self.ids = np.concatenate(ids, axis=0)

    def __getitem__(self, index):
        return {
            'rgb1': self.images[index, 0, ...],
            'rgb2': self.images[index, 1, ...],
            'theta': self.thetas[index],
            'cifar_label': self.cifar_labels[index],
            'id': self.ids[index]
        }
    
    def __len__(self):
        return len(self.images)
    