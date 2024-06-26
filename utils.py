from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import zoom


def preprocess_2d_image(x):
    x = zoom(x, (.5, .5))
    x = np.expand_dims(x, axis=0)
    x = np.concatenate([x, x, x], axis=0)
    x = x.astype('float32') / 256.
    return x


def convert_3d_image_to_2d(image_3d):
    return [preprocess_2d_image(image_3d[:, :, i])
            for i in range(image_3d.shape[-1])]


class TestImageDataset(Dataset):
    def __init__(self, x):
        self.x = x
        self.n = len(self.x)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.x[idx]


def predict_probabilities(image_3d, model, device, batch_size=4):
    images = convert_3d_image_to_2d(image_3d)
    test_loader = DataLoader(TestImageDataset(images), batch_size=batch_size, shuffle=False)
    seg_pred, dist_pred = [], []

    for x in test_loader:
        x = x.to(device)
        s, d = model(x)
        s = F.sigmoid(s).detach().cpu().numpy()
        s = np.squeeze(s, axis=1)
        d = d.detach().cpu().numpy()
        d = np.squeeze(d, axis=-1)
        seg_pred.append(s)
        dist_pred.append(d)
    return np.concatenate(seg_pred, axis=0), np.concatenate(dist_pred, axis=0)
