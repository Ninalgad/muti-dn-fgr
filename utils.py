from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import zoom

# from tqdm import tqdm


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
    predictions = []

    for x in test_loader:
        x = torch.tensor(x, dtype=torch.float32).to(device)
        p = F.sigmoid(model(x)).detach().cpu().numpy()
        predictions.append(p)
    predictions = np.concatenate(predictions, axis=0)
    predictions = np.squeeze(predictions, axis=1)

    return predictions
