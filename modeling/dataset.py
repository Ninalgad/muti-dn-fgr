# %%writefile /content/src/dataset.py
from torch.utils.data import Dataset
import numpy as np

from utils import load_mask, load_ultra


class BalancedTrainingDataset(Dataset):
    def __init__(self, df, transform=None):
        self.pos_df = df[df.label != ""]
        self.neg_df = df[df.label == ""]
        self.transform = transform

    def __len__(self):
        return len(self.pos_df)

    def augment(self, img, tar):
        if self.transform is not None:
            img, tar = np.transpose(img, (1, 2, 0)), np.transpose(tar, (1, 2, 0))
            transformed = self.transform(image=img, mask=tar)
            img = transformed["image"]
            tar = transformed["mask"]
            img, tar = np.transpose(img, (2, 0, 1)), np.transpose(tar, (2, 0, 1))
        return img, tar

    def load_sample(self, row):
        img, tar = load_ultra(row.image), load_mask(row.label)
        img, tar = self.augment(img, tar)
        v = row['volume-label']
        return img, tar, v

    def __getitem__(self, idx):
        px, py, pfd = self.load_sample(self.pos_df.iloc[idx])
        nx, ny, nfd = self.load_sample(self.neg_df.sample().iloc[0])

        batch = {"image_pos": px, "s_label_pos": py, "v_label_pos": pfd,
                 "image_neg": nx, "s_label_neg": ny, "v_label_neg": nfd}
        return batch


class DenosingDataset(Dataset):
    def __init__(self, df, size=None, transform=None, noising_transform=None):
        self.df = df
        if noising_transform is None:
            noising_transform = noising_transform

        self.noise_transform = noising_transform
        self.transform = transform
        if size is None:
            size = len(df)
        self.size = min(len(df), size)

    def __len__(self):
        return self.size

    def augment(self, img, tar):
        if self.transform is not None:
            img, tar = np.transpose(img, (1, 2, 0)), np.transpose(tar, (1, 2, 0))
            transformed = self.transform(image=img, mask=tar)
            img = transformed["image"]
            tar = transformed["mask"]
            img, tar = np.transpose(img, (2, 0, 1)), np.transpose(tar, (2, 0, 1))
        return img, tar

    def load_sample(self, row):
        img, tar = load_ultra(row.image), load_mask(row.label)
        img, tar = self.augment(img, tar)
        v = row['volume-label']
        return img, tar, v

    def __getitem__(self, idx):
        idx = np.random.choice(len(self.df))
        image, label, frame_dist = self.load_sample(self.df.iloc[idx])
        if self.transform is not None:
            image = self.transform(image=image)['image']

        image, noise = self.noise_transform.apply(image, label)

        batch = dict()
        batch['image'] = image
        batch['n_label'] = noise
        batch['v_label'] = frame_dist
        return batch
