from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn import functional as F
from scipy.ndimage import zoom, convolve


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


def process_output(x, dtype="float16"):
    x = F.sigmoid(x).detach().cpu().numpy().astype(dtype)
    for dim in x.shape:
        if dim == 1:
            x = np.squeeze(x, axis=dim)
    return x


def predict_probabilities(images, model, device, batch_size=2):
    images = convert_3d_image_to_2d(images)
    test_loader = DataLoader(TestImageDataset(images), batch_size=batch_size, shuffle=False)
    seg_pred, dist_pred = [], []

    for x in test_loader:
        x = x.to(device)
        s, d = model(x)

        s, d = process_output(s), process_output(d)
        seg_pred.append(s)
        dist_pred.append(d)

    return np.concatenate(seg_pred, axis=0), np.concatenate(dist_pred, axis=0)


def smooth(y):
    box = [0.1665, 0.333, 0.333, 0.1665]
    y_smooth = convolve(y, box, origin=0, mode="constant")
    return y_smooth
