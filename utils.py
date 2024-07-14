from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn import functional as F
from scipy.ndimage import zoom
import torch


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


def postprocess_model_output(x):
    return F.sigmoid(x).detach().cpu().numpy()


def aug_predict(model, inp, device):
    # x: (n, 3, d, d)
    pred_s, pred_d = 0, 0
    x = torch.tensor(inp, dtype=torch.float32).to(device)

    s, d = model(x)
    pred_s += postprocess_model_output(s)
    pred_d += postprocess_model_output(d)

    s, d = model(torch.flip(x, [-1]))
    s = torch.flip(s, [-1])
    pred_s += postprocess_model_output(s)
    pred_d += postprocess_model_output(d)

    s, d = model(torch.flip(x, [-2]))
    s = torch.flip(s, [-2])
    pred_s += postprocess_model_output(s)
    pred_d += postprocess_model_output(d)

    s, d = model(torch.flip(x, [-1, -2]))
    s = torch.flip(s, [-1, -2])
    pred_s += postprocess_model_output(s)
    pred_d += postprocess_model_output(d)

    return pred_s / 4., pred_d / 4.


def predict_probabilities(images, model, device, batch_size=2):
    images = convert_3d_image_to_2d(images)
    test_loader = DataLoader(TestImageDataset(images), batch_size=batch_size, shuffle=False)
    seg_pred, dist_pred = [], []

    for x in test_loader:
        # x = x.to(device)
        # s, d = model(x)
        # s = np.squeeze(F.sigmoid(s).detach().cpu().numpy().astype("float16"), axis=1)
        # d = np.squeeze(F.sigmoid(d).detach().cpu().numpy().astype("float16"), axis=-1)

        s, d = aug_predict(model, x, device)
        s = np.squeeze(s.astype("float16"), axis=1)
        d = np.squeeze(d.astype("float16"), axis=-1)

        seg_pred.append(s)
        dist_pred.append(d)

    return np.concatenate(seg_pred, axis=0), np.concatenate(dist_pred, axis=0)
